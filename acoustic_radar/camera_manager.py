import time

# INTEGRATION CHANGE (unified system): picamera2 is imported lazily rather
# than at module import time.
#
# Rationale: main.py must be able to run the ACOUSTIC subsystem on a machine
# where picamera2 is not installed (a dev laptop, or a Pi whose camera stack
# is broken). A hard top-level import made `import camera_manager` itself
# fail, which took the whole unified application down with it — exactly the
# failure isolation the integration is required to provide.
#
# Behaviour on a working Pi is unchanged: the import still happens, just at
# CameraManager() construction time instead of module load, and any failure
# surfaces as a normal exception from __init__ (which callers already handle)
# instead of an ImportError at the top of the program.
Picamera2 = None
PICAMERA2_IMPORT_ERROR = None


def _load_picamera2():
    """Import Picamera2 on first use. Returns the class or raises RuntimeError."""
    global Picamera2, PICAMERA2_IMPORT_ERROR
    if Picamera2 is not None:
        return Picamera2
    try:
        from picamera2 import Picamera2 as _P
    except Exception as exc:          # ImportError, or libcamera load failure
        PICAMERA2_IMPORT_ERROR = exc
        raise RuntimeError(
            f"picamera2 is not available ({exc}). The camera subsystem "
            f"cannot start; the acoustic subsystem is unaffected.") from exc
    Picamera2 = _P
    return Picamera2


def picamera2_available() -> bool:
    """True if picamera2 can be imported. Used for graceful degradation."""
    try:
        _load_picamera2()
        return True
    except RuntimeError:
        return False


class CameraManager:

    FAR_CAMERA_ID = 0
    NEAR_CAMERA_ID = 1

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        buffer_count=4,
        debounce_interval=0.5,
        warmup_frames=8,
        max_fps=None,
        failover_after_failures=15,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        # PERF-7 (see TWO_CAMERAS_AUDIT.md Section 33): `fps` sets the
        # SLOWEST allowed frame rate; `max_fps` sets the FASTEST. Leaving
        # max_fps=None reproduces the original behaviour exactly — both
        # FrameDurationLimits bounds equal, hard-locking the sensor to
        # `fps` and therefore hard-capping the main loop (and so the
        # displayed FPS) at that number no matter how fast the code runs.
        self.max_fps = max_fps
        self.buffer_count = buffer_count
        self.debounce_interval = debounce_interval
        self.warmup_frames = warmup_frames

        self.active_camera = self.FAR_CAMERA_ID
        self.last_switch_time = 0.0
        self.picams = {}

        # ⚠️ ALL timing in this class uses time.monotonic(), never
        # time.time(). A Raspberry Pi has no battery-backed RTC: it boots
        # believing it is 1970 and the wall clock JUMPS by decades the
        # moment NTP syncs — typically a few seconds after startup, i.e.
        # exactly while this station is initialising.
        #
        # With the wall clock, that jump either disables the switch debounce
        # entirely (a huge forward delta always exceeds the interval) or
        # freezes switching (a backward delta stays under it indefinitely).
        # This file previously used time.time() throughout; the project's
        # own tracker had already been fixed the same way (see
        # AdvancedADASTracker._compute_dt, "AUDIT BUG #5"), and this class
        # was simply missed.

        # Capture health (see get_frame / _handle_capture_failure).
        self.consecutive_capture_failures = 0
        self.total_capture_failures = 0
        self.total_frames = 0
        self.failover_after_failures = failover_after_failures
        self.failure_log_interval = 2.0     # seconds between repeated warnings
        self._last_failure_log = 0.0

        opened = []
        for camera_id in (self.FAR_CAMERA_ID, self.NEAR_CAMERA_ID):
            try:
                self._open_camera(camera_id)
                opened.append(camera_id)
            except Exception as exc:
                print(f"[CameraManager] failed to open camera {camera_id} "
                      f"at startup: {exc}")

        if not opened:
            raise RuntimeError(
                "CameraManager: no cameras could be opened at startup.")

        if self.active_camera not in self.picams:
            self.active_camera = opened[0]
            print(f"[CameraManager] camera {self.FAR_CAMERA_ID} did not "
                  f"open — defaulting active camera to {opened[0]}.")

    def _open_camera(self, camera_id):

        picam = _load_picamera2()(camera_num=camera_id)

        # PERF-7: FrameDurationLimits is (min_duration, max_duration) in
        # microseconds — the fastest and slowest frame the sensor may
        # produce. The original code passed the SAME value for both,
        # which hard-locks the sensor to exactly `fps`. Because
        # get_frame() -> capture_array() blocks until the sensor delivers
        # a frame, that lock is also a hard ceiling on how fast the whole
        # main loop can iterate: with fps=30 the displayed FPS can never
        # exceed 30 no matter how fast the code becomes.
        #
        # With max_fps set, the MAX duration still corresponds to `fps`
        # (so the longest allowed exposure is unchanged — low-light
        # behaviour is not made worse than before), while the MIN
        # duration lets the sensor run faster when there is enough light.
        # Frame rate then floats between fps and max_fps instead of being
        # pinned. max_fps=None keeps the original locked behaviour.
        slowest_us = int(1000000 / self.fps)
        fastest_us = (int(1000000 / self.max_fps)
                      if self.max_fps else slowest_us)

        config = picam.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            },
            buffer_count=self.buffer_count,
            controls={
                "FrameDurationLimits": (fastest_us, slowest_us)
            },
        )

        picam.configure(config)

        picam.start()

        # AUDIT BUG #6 fix (unchanged from session 2): log warmup failures
        # instead of silently swallowing them.
        warmup_failures = 0
        for _ in range(self.warmup_frames):
            try:
                picam.capture_array()
            except Exception as exc:
                warmup_failures += 1
                print(f"[CameraManager] warmup capture failed on camera "
                      f"{camera_id}: {exc}")
        if self.warmup_frames > 0 and warmup_failures == self.warmup_frames:
            print(f"[CameraManager] WARNING: all {self.warmup_frames} "
                  f"warmup captures failed on camera {camera_id} — it may "
                  f"not be producing valid frames.")

        # Only registered as available once fully opened, configured,
        # started, and warmed up — if any step above raised, this line
        # never runs and the camera is correctly treated as unavailable.
        self.picams[camera_id] = picam

    def _close_camera(self, camera_id):
        picam = self.picams.pop(camera_id, None)
        if picam is None:
            return
        try:
            picam.stop()
        except Exception:
            pass
        time.sleep(0.2)
        try:
            picam.close()
        except Exception:
            pass

    def get_frame(self):
        # INTEGRATION FIX (BUG C1): this used to be a bare
        #     return self.picams[self.active_camera].capture_array()
        # which had two failure modes that killed the whole application:
        #
        #   1. KeyError — self.active_camera is not a key of self.picams.
        #      Reachable in normal operation: _close_camera() pops the entry,
        #      and __init__ only guarantees that SOME camera opened, not the
        #      active one (it does fix up active_camera at startup, but any
        #      later close leaves the dict and active_camera inconsistent).
        #   2. Any exception from capture_array() — a USB/CSI hiccup, a
        #      timeout, a camera unplugged mid-run — propagated straight out
        #      of the frame loop.
        #
        # Both now return None, which every existing caller already handles
        # (`if frame_rgb is None: continue`), and a persistent failure fails
        # over to the other camera if one is open.
        picam = self.picams.get(self.active_camera)
        if picam is None:
            self._handle_capture_failure(
                f"active camera {self.active_camera} is not open")
            return None
        try:
            frame = picam.capture_array()
        except Exception as exc:
            self._handle_capture_failure(
                f"capture failed on camera {self.active_camera}: {exc}")
            return None
        if frame is None:
            self._handle_capture_failure(
                f"camera {self.active_camera} returned no frame")
            return None
        self.consecutive_capture_failures = 0
        self.total_frames += 1
        return frame

    def _handle_capture_failure(self, message):
        """Count a failed capture and fail over once the failure persists."""
        self.consecutive_capture_failures += 1
        self.total_capture_failures += 1

        # Rate-limit the message: a dead camera would otherwise print at the
        # full frame rate and drown every other log line.
        now = time.monotonic()
        if now - self._last_failure_log >= self.failure_log_interval:
            self._last_failure_log = now
            print(f"[CameraManager] {message} "
                  f"(consecutive failures: {self.consecutive_capture_failures})")

        if self.consecutive_capture_failures < self.failover_after_failures:
            return

        # Fail over to any OTHER open camera. Bypasses the debounce on
        # purpose: this is not a range-driven switch, it is the only way to
        # keep producing frames at all.
        alternatives = [cid for cid in self.picams if cid != self.active_camera]
        if not alternatives:
            return
        target = alternatives[0]
        print(f"[CameraManager] failing over from camera "
              f"{self.active_camera} to {target} after "
              f"{self.consecutive_capture_failures} consecutive failures.")
        self.active_camera = target
        self.last_switch_time = time.monotonic()
        self.consecutive_capture_failures = 0

    def available_cameras(self):
        """IDs of cameras that are currently open. Used by the unified UI."""
        return sorted(self.picams.keys())

    def get_active_camera(self):
        return self.active_camera

    def switch_to_far(self):
        return self._switch(self.FAR_CAMERA_ID)

    def switch_to_near(self):
        return self._switch(self.NEAR_CAMERA_ID)

    def _switch(self, target_id):
        if self.active_camera == target_id:
            return False

        # INTEGRATION FIX (BUG C2): availability is now checked BEFORE the
        # debounce. Previously a switch request to a camera that never
        # opened was reported as a plain debounce rejection, so a
        # permanently-missing camera looked identical to "asked again too
        # soon" and the real cause was never logged.
        if target_id not in self.picams:
            now = time.monotonic()
            if now - self._last_failure_log >= self.failure_log_interval:
                self._last_failure_log = now
                print(f"[CameraManager] cannot switch to camera {target_id}: "
                      f"it is not currently open/running.")
            return False

        if time.monotonic() - self.last_switch_time < self.debounce_interval:
            return False

        self.active_camera = target_id
        self.last_switch_time = time.monotonic()
        return True

    def release(self):
        # Defensive: __del__ runs even on a PARTIALLY-constructed object,
        # i.e. when __init__ raised before (or while) setting self.picams
        # — a bad keyword argument, an import problem, or the very first
        # camera failing to open. Reading self.picams unguarded there
        # raises AttributeError inside __del__, which Python prints as a
        # confusing "Exception ignored in ..." traceback that appears
        # BEFORE, and therefore masks, the real underlying error.
        picams = getattr(self, "picams", None)
        if not picams:
            return
        for camera_id in list(picams.keys()):
            self._close_camera(camera_id)

    def __del__(self):
        # Never let interpreter shutdown / GC surface a secondary error
        # that hides the primary one.
        try:
            self.release()
        except Exception:
            pass

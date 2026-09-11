import time

import camera_identity

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


def global_camera_info():
    """
    libcamera's view of every attached camera. Raises if unavailable.

    Separated from CameraManager so the role-resolution path can be driven
    with recorded hardware output in tests.
    """
    return _load_picamera2().global_camera_info()


class CameraManager:
    """
    Owns both cameras and the role -> device binding.

    ⚠️ `FAR_CAMERA_ID` / `NEAR_CAMERA_ID` USED TO BE CLASS CONSTANTS equal
    to 0 and 1, and they were the station's ONLY notion of camera identity
    (audit finding C1). They are gone. A device index is assigned by
    libcamera's enumeration order and says nothing about which lens is
    fitted, so the role is now resolved from each camera's stable
    device-tree Id at construction time and exposed per INSTANCE as
    `wide_id` / `tele_id`.
    """

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
        role_hints=None,
        expected_sensor_model=camera_identity.EXPECTED_SENSOR_MODEL,
        info_provider=None,
        lens_mm=None,
    ):
        #: role -> nominal lens focal length in mm, from GeometryConfig.
        #: Empty disables the runtime focal derivation below (the
        #: configured THEORETICAL default then stands unchanged).
        self.lens_mm = dict(lens_mm or {})
        self._role_hints = dict(role_hints or {})
        self._expected_sensor_model = expected_sensor_model

        # ── Enumerate and validate BEFORE opening anything ──
        #
        # Count and sensor model can be checked from libcamera's metadata
        # alone, so a missing or unexpected camera fails here rather than
        # after two pipelines have been started.
        provider = info_provider or global_camera_info
        self.cameras_info = camera_identity.validate_cameras(
            list(provider()), expected_sensor_model)

        # Roles are NOT known yet — see _resolve_roles(), called once both
        # cameras are running, because the decisive evidence is in the
        # images themselves.
        self.roles = {}
        self.role_of = {}
        self.wide_id = None
        self.tele_id = None
        self.role_source = "unresolved"
        self.role_detail = ""

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

        # Set once the roles are known (see _resolve_roles). TELE is the
        # default: it is the long lens, and the switch to WIDE only happens
        # once the drone is close enough to overflow it.
        self.active_camera = None
        self.last_switch_time = 0.0
        self.picams = {}

        #: role -> horizontal focal length in pixels of the delivered frame,
        #: recomputed from each sensor's ACTUAL crop once it is running.
        #: Empty until _open_camera succeeds. See _derive_focal_px.
        self.focal_px_by_role = {}
        self.focal_source_by_role = {}

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

        # ⚠️ CAN THE DRIVER TELL US WHEN THE SENSOR ACTUALLY EXPOSED A FRAME?
        #
        # `capture_array()` returns pixels and nothing else, so a caller
        # cannot distinguish "the sensor just produced this" from "this had
        # been sitting in the request queue for three frame periods". Those
        # two are the difference between a real frame rate and a processing
        # rate, and between fresh video and stale video.
        #
        # `capture_request()` returns the same pixels PLUS the metadata, which
        # carries SensorTimestamp. It costs nothing extra — capture_array is
        # implemented on top of the same mechanism — but it does hand us a
        # buffer we are responsible for releasing, so every path below is
        # wrapped in try/finally.
        #
        # If that path ever raises we fall back to capture_array PERMANENTLY:
        # a metric is never worth losing frames over.
        self._request_capture_ok = True
        self._sensor_ts_supported = None      # None = not yet known

        # Capture health (see get_frame / _handle_capture_failure).
        self.consecutive_capture_failures = 0
        self.total_capture_failures = 0
        self.total_frames = 0
        self.failover_after_failures = failover_after_failures
        self.failure_log_interval = 2.0     # seconds between repeated warnings
        self._last_failure_log = 0.0

        # ── Open by INDEX; roles are decided afterwards ──
        indices = [int(c.get("Num", i))
                   for i, c in enumerate(self.cameras_info)]
        opened = []
        for camera_id in indices:
            try:
                self._open_camera(camera_id)
                opened.append(camera_id)
            except Exception as exc:
                print(f"[CameraManager] failed to open camera index "
                      f"{camera_id} at startup: {exc}")

        if not opened:
            raise RuntimeError(
                "CameraManager: no cameras could be opened at startup.")

        # ⚠️ Fail-closed. _resolve_roles() raises rather than guessing, and
        # it runs before any frame is served, so a station whose cameras
        # cannot be told apart produces an error and no video — never video
        # with the roles silently transposed. The acoustic subsystem is a
        # separate thread and is unaffected; camera_worker catches this and
        # reports the camera subsystem OFFLINE.
        self._resolve_roles(opened)

        # Focal length is per-ROLE, so it can only be derived now.
        for camera_id in opened:
            self._derive_focal_px(camera_id, self.picams[camera_id])

        self.active_camera = self.tele_id
        if self.active_camera not in self.picams:
            self.active_camera = opened[0]
            print(f"[CameraManager] the TELE camera (index {self.tele_id}) "
                  f"did not open — defaulting active camera to "
                  f"{self.role_of.get(opened[0], opened[0])} "
                  f"(index {opened[0]}).")

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
        # ⚠️ The focal length is NOT derived here any more. It is per-ROLE,
        # and the role is not known until every camera is running and the
        # optical check has run — see _resolve_roles(). __init__ calls
        # _derive_focal_px() for each open camera immediately afterwards.
        self.picams[camera_id] = picam

    def _probe_frame(self, camera_id):
        """One frame for the optical role check, or None."""
        picam = self.picams.get(camera_id)
        if picam is None:
            return None
        try:
            return picam.capture_array()
        except Exception as exc:
            print(f"[CameraManager] could not grab a probe frame from "
                  f"index {camera_id} ({exc}) — the optical role check "
                  f"cannot run on it.")
            return None

    def _optical_roles(self, opened):
        """
        Work out which camera holds the long lens BY LOOKING (finding C1).

        Returns {role: index} or None. Never raises: an inconclusive scene
        is an expected outcome, not an error.
        """
        if len(opened) < 2:
            return None
        lens_wide = self.lens_mm.get(camera_identity.WIDE)
        lens_tele = self.lens_mm.get(camera_identity.TELE)
        if not lens_wide or not lens_tele or lens_tele <= lens_wide:
            return None
        ratio = float(lens_tele) / float(lens_wide)

        a, b = opened[0], opened[1]
        frame_a, frame_b = self._probe_frame(a), self._probe_frame(b)
        if frame_a is None or frame_b is None:
            return None

        try:
            role_a, role_b, detail = camera_identity.classify_roles_by_optics(
                frame_a, frame_b, ratio)
        except Exception as exc:
            print(f"[CameraManager] optical role check failed to run "
                  f"({exc}) — falling back to the configured hint.")
            return None

        self.role_detail = detail
        if role_a is None:
            print(f"[CameraManager] optical role check INCONCLUSIVE: {detail}")
            return None
        print(f"[CameraManager] optical role check: index {a} = {role_a}, "
              f"index {b} = {role_b}  [{detail}]")
        return {role_a: a, role_b: b}

    def _resolve_roles(self, opened):
        """
        Decide the role -> index mapping, or refuse.

        ═══════════════════════════════════════════════════════════
        TWO INDEPENDENT SOURCES, AND THEY CHECK EACH OTHER
        ═══════════════════════════════════════════════════════════

        1. THE OPTICAL MEASUREMENT (`_optical_roles`) reads the answer out
           of the images. The 25 mm lens magnifies 4.17x more than the
           6 mm one, so the telephoto view is the centre of the wide view
           blown up — a difference far too large to mistake. This is the
           primary source because it is a MEASUREMENT of the actual
           hardware rather than a statement about it.

        2. THE CONFIGURED HINT (`camera_role_id_hint`) binds a role to a
           device-tree Id, i.e. to a physical CSI socket. This is what
           makes the mapping STABLE across reboots and reordering.

        Used together they answer both halves of finding C1: the optics say
        WHICH lens, the Id says WHICH SOCKET, and each catches the other
        being wrong. A disagreement is a hard error — it means the recorded
        socket assignment no longer matches the hardware, which is exactly
        the silent lens swap this whole mechanism exists to prevent.
        """
        optical = self._optical_roles(opened)

        hinted = None
        if all(self._role_hints.get(r) for r in camera_identity.ROLES):
            hinted = camera_identity.resolve_roles(
                self.cameras_info, self._role_hints,
                self._expected_sensor_model)

        if optical and hinted:
            if optical != hinted:
                raise camera_identity.CameraIdentityError(
                    f"THE CONFIGURED CAMERA ROLES CONTRADICT THE OPTICS.\n"
                    f"  configured (by device Id): {hinted}\n"
                    f"  measured   (by lens magnification): {optical}\n"
                    f"  {self.role_detail}\n"
                    f"  One of these is wrong, and the station will not "
                    f"guess which. Either a camera was moved to the other "
                    f"CSI socket without updating "
                    f"geometry.camera_role_id_hint, or the lenses were "
                    f"swapped between the two bodies.\n"
                    f"  The measurement describes the hardware as it is "
                    f"NOW; if the cameras were re-cabled, update the hint "
                    f"to match. See HARDWARE_TEST_REQUIRED.md, test HW-1.")
            self.roles = hinted
            self.role_source = "device Id, CONFIRMED by lens magnification"
        elif hinted:
            self.roles = hinted
            self.role_source = ("device Id (optical confirmation "
                                "unavailable this run)")
        elif optical:
            self.roles = optical
            self.role_source = "lens magnification (measured from the images)"
        else:
            # Neither source could answer. resolve_roles() raises with the
            # full operator message, including the exact JSON to paste.
            camera_identity.resolve_roles(
                self.cameras_info, self._role_hints,
                self._expected_sensor_model)
            raise AssertionError("unreachable")   # pragma: no cover

        missing = [r for r in camera_identity.ROLES if r not in self.roles]
        if missing:
            raise camera_identity.CameraIdentityError(
                f"camera role(s) {', '.join(missing)} were not resolved")

        self.wide_id = self.roles[camera_identity.WIDE]
        self.tele_id = self.roles[camera_identity.TELE]
        self.role_of = {index: role for role, index in self.roles.items()}
        print(f"[CameraManager] camera roles resolved by {self.role_source}: "
              f"WIDE=index {self.wide_id}, TELE=index {self.tele_id}")

    def _derive_focal_px(self, camera_id, picam):
        """
        Recompute this camera's THEORETICAL focal length from the mode it
        actually ended up in.

        ⚠️ WHY THIS IS NOT A CONSTANT (audit finding C2). A focal length in
        PIXELS is not a property of the lens; it is a property of the lens
        AND the sensor mode. Binning, cropping and ISP downscaling all
        change how much of the sensor one output pixel covers, so the same
        6 mm lens is ~930 px in the 1332x990 mode and ~611 px in a
        full-field-of-view mode — a 1.5x difference that lands directly in
        every distance in metres.

        The configured defaults assume the 1332x990 mode. This asks the
        driver what actually happened instead, so a libcamera version that
        picks a different mode cannot silently invalidate the optics.

        ⚠️ THE RESULT IS STILL THEORETICAL. Reading the real crop removes
        the mode assumption; it does not measure the lens. A nominal 25 mm
        lens is only approximately 25 mm, the projection is assumed
        distortion-free, and none of that is checked here. This value must
        never be reported as CALIBRATED — see HARDWARE_TEST_REQUIRED.md
        HW-2 for the measurement that would earn that label.

        Failure is not fatal: the configured THEORETICAL default stands and
        we say so. A camera that delivers frames is worth more than a
        derived constant.
        """
        role = self.role_of.get(camera_id)
        lens_mm = (self.lens_mm or {}).get(role) if role else None
        if not role or not lens_mm:
            return
        try:
            props = picam.camera_properties or {}
            array_w = int(props["PixelArraySize"][0])
            metadata = picam.capture_metadata() or {}
            # ScalerCrop is (x, y, w, h) in full-pixel-array coordinates:
            # exactly the sensor region the ISP sampled for this stream.
            crop_w = int(metadata["ScalerCrop"][2])
            out_w = int(picam.camera_configuration()["main"]["size"][0])
        except Exception as exc:
            print(f"[CameraManager] could not read the sensor crop for "
                  f"{role} ({exc}) — keeping the configured THEORETICAL "
                  f"focal length, which assumes the 1332x990 mode.")
            return

        if not (0 < crop_w <= array_w) or out_w <= 0:
            print(f"[CameraManager] implausible crop for {role} "
                  f"(crop {crop_w} of array {array_w}, output {out_w}) — "
                  f"keeping the configured THEORETICAL focal length.")
            return

        focal = camera_identity.theoretical_focal_px(
            lens_mm=float(lens_mm),
            sensor_crop_width_px=crop_w,
            output_width_px=out_w)
        self.focal_px_by_role[role] = focal
        self.focal_source_by_role[role] = "THEORETICAL(runtime-mode)"
        print(f"[CameraManager] {role}: {lens_mm:.0f} mm lens, sensor crop "
              f"{crop_w}/{array_w} px -> output {out_w} px gives a "
              f"THEORETICAL focal of {focal:.0f} px (NOT a calibration)")

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
        frame, _sensor_ns = self.get_frame_and_sensor_ns()
        return frame

    def get_frame_and_sensor_ns(self):
        """
        (frame, sensor_timestamp_ns) — the timestamp is None when unavailable.

        ⚠️ WHY THE TIMESTAMP EXISTS. Without it, "the loop ran 70 times this
        second" and "the sensor produced 70 frames this second" are the same
        measurement, and they are NOT the same thing. picamera2 holds
        `buffer_count` completed requests; a loop that speeds up (because, say,
        the detector started running on every second frame instead of every
        frame) drains that backlog and briefly completes iterations faster than
        the sensor can possibly produce frames.

        SensorTimestamp is the instant the sensor exposed the frame, so:

            new-frame rate   = 1 / diff(SensorTimestamp)      <- physical
            loop rate        = 1 / diff(loop wall clock)      <- processing
            frame age        = now - SensorTimestamp          <- staleness

        It is CLOCK_BOOTTIME nanoseconds on the Pi, which is why callers must
        compare it against time.clock_gettime(CLOCK_BOOTTIME) and not against
        time.monotonic(). See camera_worker._sensor_now_ns().
        """
        picam = self.picams.get(self.active_camera)
        if picam is None:
            self._handle_capture_failure(
                f"active camera {self.active_camera} is not open")
            return None, None

        frame = None
        sensor_ns = None

        # getattr, for the same reason release() uses it: this method's
        # contract is that it NEVER raises, and it can be reached on a
        # partially-constructed object (an __init__ that failed part way, or
        # a test that builds one with __new__). An AttributeError here would
        # break that contract for a diagnostic flag.
        if getattr(self, "_request_capture_ok", True):
            request = None
            try:
                request = picam.capture_request()
                try:
                    frame = request.make_array("main")
                    metadata = request.get_metadata() or {}
                finally:
                    # Releasing is not optional: a leaked request permanently
                    # removes one buffer from a pool of `buffer_count`, and
                    # leaking `buffer_count` of them stalls capture for good.
                    request.release()
                ts = metadata.get("SensorTimestamp")
                sensor_ns = int(ts) if ts is not None else None
                if getattr(self, "_sensor_ts_supported", False) is None:
                    self._sensor_ts_supported = sensor_ns is not None
                    if not self._sensor_ts_supported:
                        print("[CameraManager] the driver does not report "
                              "SensorTimestamp — real capture rate and frame "
                              "age cannot be measured on this system.")
            except Exception as exc:
                # One failure disables the path for the whole session. The
                # metric is a diagnostic; frames are the product.
                self._request_capture_ok = False
                frame = None
                print(f"[CameraManager] capture_request() unavailable "
                      f"({exc}) — falling back to capture_array(); the real "
                      f"sensor frame rate will not be measurable.")

        if frame is None:
            try:
                frame = picam.capture_array()
            except Exception as exc:
                self._handle_capture_failure(
                    f"capture failed on camera {self.active_camera}: {exc}")
                return None, None

        if frame is None:
            self._handle_capture_failure(
                f"camera {self.active_camera} returned no frame")
            return None, None
        self.consecutive_capture_failures = 0
        self.total_frames += 1
        return frame, sensor_ns

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

    def role_for(self, camera_id):
        """Role of a device index, or CAM<n> if it is not one of ours."""
        return self.role_of.get(camera_id, f"CAM{camera_id}")

    def switch_to_tele(self):
        """Select the long lens. Was `switch_to_far`."""
        return self._switch(self.tele_id)

    def switch_to_wide(self):
        """Select the wide lens. Was `switch_to_near`."""
        return self._switch(self.wide_id)

    # ⚠️ FAR/NEAR are retained ONLY as names, mapped to the resolved role
    # indices, so the legacy standalone loop in TWO_CAMERAS_FIXED.main()
    # keeps working. They no longer imply an index: "FAR" is the long lens
    # (TELE) and "NEAR" is the wide one (WIDE), whichever device each
    # turned out to be. Prefer the role names in new code.
    switch_to_far = switch_to_tele
    switch_to_near = switch_to_wide

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

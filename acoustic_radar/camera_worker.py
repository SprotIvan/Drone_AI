#!/usr/bin/env python3
"""
camera_worker.py — Runs capture + YOLO/Hailo + tracking on its own thread.

As with the acoustic side, the detection and tracking code is NOT
reimplemented. This wraps the existing, working pieces:

    CameraManager ──► HailoInference ──► AdvancedADASTracker ──► VisualObservation
    (TWO_CAMERAS_FIXED.py / camera_manager.py, both essentially untouched)

What this file adds:

  • a thread, so camera work never blocks audio or the UI;
  • CameraSwitchPolicy — the near/far decision, moved out of the old main()
    loop and given temporal confirmation on top of its distance hysteresis;
  • supervision, so a Hailo failure degrades to "camera offline" instead of
    killing the process;
  • translation into the immutable VisualObservation the fusion layer reads.

═══════════════════════════════════════════════════════════════════
FRAME OWNERSHIP (why there is no extra copy per frame)
═══════════════════════════════════════════════════════════════════

`Picamera2.capture_array()` allocates a NEW array per call, so a frame this
worker has published is never written to again by the worker. The published
`VisualObservation.frame` is therefore a reference, not a copy — exactly as
many bytes per frame as the original single-threaded program moved.

The HUD makes the only copy, and only when it must (it draws chrome onto
the image). With display_scale != 1.0 the resize already produces a new
buffer, so in the default configuration there is no additional copy at all.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from fusion_config import StationConfig
from station_logging import EventLogger
from target_state import (LatestValue, SubsystemHealth, SubsystemState,
                          VisualObservation, VisualTrack, now)

log = logging.getLogger("station.camera")


# ═══════════════════════════════════════════════════════════════
#  Camera selection policy
# ═══════════════════════════════════════════════════════════════

class CameraSwitchPolicy:
    """
    Decides which camera should be active.

    Preserves the original logic exactly — distance-based with a 1.5 m /
    2.0 m hysteresis band, falling back to pixel height when the active
    camera has no focal calibration — and adds ONE improvement:

        the condition must hold for `confirm_frames` consecutive frames.

    Why: the original switched on a single frame's estimate. A bounding box
    that jitters by a few pixels changes the derived distance by tens of
    centimetres, so a target sitting near 1.5 m could request a switch on
    one frame and the opposite switch on the next. CameraManager's 0.5 s
    debounce limited how *fast* that could happen but did not stop it —
    the result was a camera that flipped roughly twice a second. Requiring
    agreement across consecutive frames removes the cause rather than
    rate-limiting the symptom.

    ⚠️ Acoustic range is deliberately NOT an input here. See
    CameraSwitchConfig.allow_acoustic_fallback for the reasoning: the
    acoustic estimator's floor is 2.0 m and its accuracy is ±40–60%, so it
    cannot resolve a 1.5 m/2.0 m decision even in principle.
    """

    def __init__(self, config: StationConfig, events: EventLogger):
        self.config = config.switching
        self.events = events
        self.frozen = False
        self._near_streak = 0
        self._far_streak = 0
        self.last_reason = ""

    def freeze(self, frozen: bool) -> None:
        """Hold the current camera (used while calibrating)."""
        self.frozen = frozen
        self._near_streak = self._far_streak = 0

    def evaluate(self, manager, current_camera: int,
                 distance_m: Optional[float],
                 max_box_height_px: float) -> Optional[str]:
        """
        Returns a human-readable description of a switch that happened, or
        None. Never raises: a switching failure must not stop the frame loop.
        """
        if self.frozen:
            return None

        from camera_manager import CameraManager
        cfg = self.config

        want_near = want_far = False
        basis = ""

        if distance_m is not None:
            # PRIMARY PATH — metric distance with hysteresis.
            want_near = distance_m < cfg.switch_to_near_below_m
            want_far = distance_m > cfg.switch_to_far_above_m
            basis = f"{distance_m:.2f} m"
        elif max_box_height_px > 0:
            # FALLBACK — uncalibrated optics, pixel height only.
            want_near = max_box_height_px > cfg.fallback_near_height_px
            want_far = max_box_height_px < cfg.fallback_far_height_px
            basis = f"{max_box_height_px:.0f} px"
        else:
            # No target at all: decay both streaks so a stale streak cannot
            # trigger a switch later.
            self._near_streak = self._far_streak = 0
            return None

        self._near_streak = self._near_streak + 1 if want_near else 0
        self._far_streak = self._far_streak + 1 if want_far else 0

        need = cfg.confirm_frames

        if (current_camera == CameraManager.FAR_CAMERA_ID
                and self._near_streak >= need):
            if manager.switch_to_near():
                self._near_streak = 0
                self.last_reason = f"FAR -> NEAR ({basis})"
                return self.last_reason
            # Suppressed by the debounce or the camera is not open. Keep the
            # streak so the switch fires as soon as it is permitted.
        elif (current_camera == CameraManager.NEAR_CAMERA_ID
                and self._far_streak >= need):
            if manager.switch_to_far():
                self._far_streak = 0
                self.last_reason = f"NEAR -> FAR ({basis})"
                return self.last_reason

        return None


# ═══════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Owns the cameras, the Hailo device and the tracker.

    Public surface used by main.py:
        start() / stop() / join()
        latest      LatestValue[VisualObservation]
        health      LatestValue[SubsystemHealth]
        freeze_switching(bool), calibrate_focal() — keyboard actions
    """

    def __init__(self, config: StationConfig, events: EventLogger,
                 hef_path: str):
        self.config = config
        self.events = events
        self.hef_path = hef_path

        self.latest: LatestValue[VisualObservation] = LatestValue()
        self.health: LatestValue[SubsystemHealth] = LatestValue(
            SubsystemHealth(SubsystemState.STARTING, "not started"))

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._manager = None
        self._detector = None
        self._tracker = None
        self._policy = CameraSwitchPolicy(config, events)

        self._fps_ema = 0.0
        self._infer_ema = 0.0
        self._frame_count = 0
        self._detector_available = False
        # Requested from the UI thread, serviced on the camera thread.
        self._calibrate_request = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="camera",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("camera thread did not stop within %.1fs "
                            "(blocked in capture) — leaving it to the daemon "
                            "shutdown", timeout)

    def _set_health(self, state: SubsystemState, detail: str = "",
                    error: Optional[str] = None) -> None:
        current = self.health.get() or SubsystemHealth()
        self.health.publish(current.with_state(state, detail, error))

    # ── Keyboard actions (called from the UI thread) ───────────

    def freeze_switching(self, frozen: bool) -> None:
        self._policy.freeze(frozen)

    def request_focal_calibration(self) -> None:
        """Ask the camera thread to print a focal-length calibration."""
        self._calibrate_request.set()

    @property
    def switching_frozen(self) -> bool:
        return self._policy.frozen

    # ── Thread body ────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._setup()
        except BaseException as exc:
            msg = str(exc) or exc.__class__.__name__
            log.error("camera subsystem could not start: %s",
                      msg.splitlines()[0])
            self._set_health(SubsystemState.OFFLINE, "init failed", msg)
            return

        try:
            self._loop()
        except BaseException as exc:
            log.exception("camera loop crashed")
            self._set_health(SubsystemState.OFFLINE, "loop crashed", str(exc))
        finally:
            self._teardown()

    # ── Setup ──────────────────────────────────────────────────

    def _sync_optics(self, tc) -> None:
        """
        Make fusion_config the single source of truth for the optics.

        `TWO_CAMERAS_FIXED.estimate_distance_m()` reads that module's own
        CAMERA_FOCAL_PX / DRONE_REAL_WIDTH_M globals. GeometryConfig carries
        the same numbers so the fusion layer can derive the camera's useful
        range. Two copies of a calibration constant WILL drift: someone sets
        the focal length in fusion_config.json after a re-calibration, the
        range gate updates, and every distance label on screen silently
        keeps using the old value.

        Pushing config -> module at startup means the JSON override governs
        both. Values absent from the config leave the module's own constants
        untouched.
        """
        geo = self.config.geometry
        for camera_id, focal in geo.camera_focal_px.items():
            if focal is not None:
                tc.CAMERA_FOCAL_PX[camera_id] = float(focal)
        if geo.drone_real_width_m is not None:
            tc.DRONE_REAL_WIDTH_M = float(geo.drone_real_width_m)
        log.debug("optics synced from config: focal=%s width=%s",
                  tc.CAMERA_FOCAL_PX, tc.DRONE_REAL_WIDTH_M)

    def _setup(self) -> None:
        import TWO_CAMERAS_FIXED as tc
        from camera_manager import CameraManager

        self._sync_optics(tc)
        v = self.config.visual
        log.info("opening cameras (%dx%d @ %d fps)...",
                 v.frame_width, v.frame_height, v.fps)
        self._manager = CameraManager(
            width=v.frame_width, height=v.frame_height, fps=v.fps,
            max_fps=v.max_fps, buffer_count=v.buffer_count,
            debounce_interval=self.config.switching.debounce_interval_s,
            warmup_frames=v.warmup_frames)
        opened = self._manager.available_cameras()
        log.info("cameras opened: %s (active %d)", opened,
                 self._manager.get_active_camera())
        if len(opened) < 2:
            log.warning("only %d of 2 cameras available — switching will be "
                        "limited to what is open", len(opened))

        # ── Detector: optional. No Hailo must not stop the camera. ──
        try:
            self._detector = tc.HailoInference(
                self.hef_path,
                conf_thresh=v.detector_conf_threshold,
                iou_thresh=v.detector_iou_threshold)
            self._detector_available = True
            log.info("HEF loaded: %s (conf %.2f, iou %.2f)", self.hef_path,
                     v.detector_conf_threshold, v.detector_iou_threshold)
        except Exception as exc:
            self._detector = None
            self._detector_available = False
            log.error("YOLO/Hailo unavailable — the camera will show live "
                      "video with NO detection: %s", exc)

        self._tracker = tc.AdvancedADASTracker()

        detail = (f"{len(opened)} camera(s)"
                  + ("" if self._detector_available else ", NO DETECTOR"))
        self._set_health(
            SubsystemState.ONLINE if self._detector_available
            else SubsystemState.DEGRADED, detail)

    # ── Main loop ──────────────────────────────────────────────

    def _loop(self) -> None:
        import cv2
        import TWO_CAMERAS_FIXED as tc

        v = self.config.visual
        consecutive_failures = 0
        last_time = time.monotonic()

        log.info("camera loop running")

        while not self._stop.is_set():
            frame_start = time.monotonic()

            frame_raw = self._manager.get_frame()
            if frame_raw is None:
                consecutive_failures += 1
                self.events.rate("camera-noframe", logging.WARNING,
                                 "no frame from camera (x%d)",
                                 consecutive_failures, interval=3.0)
                if consecutive_failures >= 30:
                    self._set_health(SubsystemState.OFFLINE,
                                     "camera stopped delivering frames")
                    # Keep the thread alive and keep retrying: a USB/CSI
                    # camera can come back, and the acoustic side is
                    # unaffected either way.
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
                continue

            if consecutive_failures:
                consecutive_failures = 0
                self._set_health(
                    SubsystemState.ONLINE if self._detector_available
                    else SubsystemState.DEGRADED, "recovered")

            self._frame_count += 1

            # ⚠️ Preserved exactly from the original main(): the detector is
            # fed the RAW camera array and the display gets a colour-swapped
            # copy. See VisualConfig.swap_detector_channels for why this is
            # not "fixed" here without a measurement to justify it.
            frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            detector_input = frame_bgr if v.swap_detector_channels else frame_raw

            # ── Detection (duty-cycled exactly as before) ──
            detections: List[np.ndarray] = []
            scores: List[float] = []
            has_confirmed = any(t.state == tc.TrackState.Confirmed
                                for t in self._tracker.tracks)
            run_detector = (self._detector_available
                            and ((self._frame_count % v.frame_skip == 0)
                                 or not has_confirmed))

            infer_ms: Optional[float] = None
            if run_detector:
                t0 = time.monotonic()
                try:
                    detections, scores = self._detector.predict_with_scores(
                        detector_input)
                except Exception as exc:
                    self.events.rate("infer-error", logging.ERROR,
                                     "inference failed: %s", exc)
                    self._set_health(SubsystemState.DEGRADED,
                                     "inference errors", str(exc))
                    detections, scores = [], []
                infer_ms = (time.monotonic() - t0) * 1000.0
                self._infer_ema = (0.9 * self._infer_ema + 0.1 * infer_ms
                                   if self._infer_ema else infer_ms)

            # ── Tracking ──
            try:
                self._tracker.process_frame(frame_bgr, detections,
                                            run_association=run_detector,
                                            detection_scores=scores)
            except Exception as exc:
                self.events.rate("tracker-error", logging.ERROR,
                                 "tracker failed: %s", exc)

            active = self._manager.get_active_camera()
            tracks, max_box_h, max_box_w = self._collect_tracks(active)

            # ── Camera selection ──
            switch_distance = tc.estimate_distance_m(max_box_w, active)
            try:
                switched = self._policy.evaluate(self._manager, active,
                                                 switch_distance, max_box_h)
                if switched:
                    log.info("camera switched: %s", switched)
                    active = self._manager.get_active_camera()
            except Exception as exc:
                self.events.rate("switch-error", logging.WARNING,
                                 "camera switch failed: %s", exc)

            if self._calibrate_request.is_set():
                self._calibrate_request.clear()
                self._print_calibration(active, max_box_w)

            # ── Publish ──
            dt = frame_start - last_time
            last_time = frame_start
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = (0.9 * self._fps_ema + 0.1 * inst
                                 if self._fps_ema else inst)

            self._seq += 1
            self.latest.publish(VisualObservation(
                frame=frame_bgr,
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
                tracks=tuple(tracks),
                detector_ran=run_detector,
                detection_count=len(detections),
                active_camera=active,
                camera_name=self._camera_name(active),
                available_cameras=tuple(self._manager.available_cameras()),
                inference_ms=(self._infer_ema if self._infer_ema else None),
                tracker_ms=None,
                loop_fps=self._fps_ema,
                timestamp=now(),
                seq=self._seq))

            self.events.rate(
                "camera-heartbeat", logging.DEBUG,
                "camera: %.1f fps, %d track(s), infer %.0f ms, cam %d",
                self._fps_ema, len(tracks), self._infer_ema, active,
                interval=5.0)

    # ── Track extraction ───────────────────────────────────────

    def _collect_tracks(self, active_camera: int
                        ) -> Tuple[List[VisualTrack], float, float]:
        """
        Convert tracker state into immutable records.

        Also returns the largest box height and width across live tracks —
        the width drives the camera-switch distance and the calibration key,
        exactly as in the original main().
        """
        import TWO_CAMERAS_FIXED as tc

        out: List[VisualTrack] = []
        max_h = max_w = 0.0
        t_now = time.monotonic()

        for track in self._tracker.tracks:
            if track.state == tc.TrackState.Deleted:
                continue
            state = track.merged_x
            if state is None or len(state) < 4:
                continue
            if not np.all(np.isfinite(state[:4])):
                continue
            cx, cy, w, h = (float(state[0]), float(state[1]),
                            float(state[2]), float(state[3]))
            if w <= 1 or h <= 1:
                continue

            if track.state in (tc.TrackState.Confirmed, tc.TrackState.Tentative):
                max_h = max(max_h, h)
                max_w = max(max_w, w)

            det_age = (None if track.last_detection_time is None
                       else t_now - track.last_detection_time)
            # A detection score older than the visual stale window is no
            # longer evidence about *now*; report None ("coasting") rather
            # than a stale number the UI would show as current confidence.
            score = track.last_detection_score
            if det_age is not None and det_age > self.config.visual.stale_after_s:
                score = None

            out.append(VisualTrack(
                track_id=int(track.id),
                bbox=(cx - w / 2.0, cy - h / 2.0, w, h),
                state=str(track.state),
                quality_score=float(track.quality_score),
                detection_score=score,
                detection_age_s=det_age,
                distance_m=tc.estimate_distance_m(w, active_camera)))

        return out, max_h, max_w

    def _camera_name(self, camera_id: int) -> str:
        from camera_manager import CameraManager
        if camera_id == CameraManager.FAR_CAMERA_ID:
            return "FAR/IMX477"
        if camera_id == CameraManager.NEAR_CAMERA_ID:
            return "NEAR/IMX708"
        return f"CAM{camera_id}"

    def _print_calibration(self, camera_id: int, box_width_px: float) -> None:
        """The 'c' key, preserved from the original with its guard rails."""
        import TWO_CAMERAS_FIXED as tc

        if tc.DRONE_REAL_WIDTH_M is None:
            log.warning("calibration: DRONE_REAL_WIDTH_M is not set")
            return
        if box_width_px <= 1.0:
            log.warning("calibration: no drone box visible — get a stable "
                        "detection first")
            return
        if box_width_px < tc.MIN_CALIBRATION_BOX_PX:
            log.warning("calibration REJECTED: box is %.1f px, below the "
                        "%.0f px minimum. A small box has inflated edges and "
                        "produces a focal length that is wrong by a factor "
                        "of ~2 — move closer.",
                        box_width_px, tc.MIN_CALIBRATION_BOX_PX)
            return
        focal = (box_width_px * tc.CALIBRATION_DISTANCE_M
                 / tc.DRONE_REAL_WIDTH_M)
        log.info("calibration: camera %d (%s), box %.1f px at %.1f m, "
                 "real width %.2f m", camera_id, self._camera_name(camera_id),
                 box_width_px, tc.CALIBRATION_DISTANCE_M, tc.DRONE_REAL_WIDTH_M)
        log.info("calibration:   -> set CAMERA_FOCAL_PX[%d] = %.1f "
                 "(and geometry.camera_focal_px in fusion_config.json)",
                 camera_id, focal)

    # ── Teardown ───────────────────────────────────────────────

    def _teardown(self) -> None:
        for name, obj in (("detector", self._detector),
                          ("cameras", self._manager)):
            if obj is None:
                continue
            try:
                obj.release()
            except Exception as exc:
                log.debug("error releasing %s: %s", name, exc)

        if self._manager is not None:
            log.info("camera frames delivered: %d, capture failures: %d",
                     getattr(self._manager, "total_frames", 0),
                     getattr(self._manager, "total_capture_failures", 0))
        self._set_health(SubsystemState.OFFLINE, "stopped")
        log.info("camera worker stopped")

    # ── Diagnostics ────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._fps_ema

    @property
    def inference_ms(self) -> float:
        return self._infer_ema


if __name__ == "__main__":
    # Tests the switching policy without any hardware, using a stub manager.
    import logging as _logging

    from fusion_config import load as load_config
    from station_logging import setup

    cfg = load_config()
    cfg.logging.to_file = False
    setup(cfg.logging)
    events = EventLogger(_logging.getLogger("station.test"), cfg.logging)

    class StubManager:
        """Mimics CameraManager's switching surface, including the debounce."""
        FAR_CAMERA_ID = 0
        NEAR_CAMERA_ID = 1

        def __init__(self, debounce=0.5):
            self.active = 0
            self.debounce = debounce
            self.last = 0.0
            self.switches = 0

        def get_active_camera(self):
            return self.active

        def _switch(self, target):
            if self.active == target:
                return False
            if time.monotonic() - self.last < self.debounce:
                return False
            self.active = target
            self.last = time.monotonic()
            self.switches += 1
            return True

        def switch_to_near(self):
            return self._switch(self.NEAR_CAMERA_ID)

        def switch_to_far(self):
            return self._switch(self.FAR_CAMERA_ID)

    import sys
    sys.modules.setdefault("camera_manager", type(sys)("camera_manager"))
    sys.modules["camera_manager"].CameraManager = StubManager

    print("=" * 66)
    print("camera_worker.py — camera switching policy test")
    print("=" * 66)

    print("\nTEST 7: distance fluctuating around the switching thresholds.")
    print("Case A: small jitter at 1.5 m (sigma 5 cm). The existing 1.5/2.0 m")
    print("        hysteresis band alone already handles this.")
    print("Case B: a drone hovering mid-band at 1.75 m with realistic box")
    print("        jitter (sigma 35 cm), so single frames land BOTH below")
    print("        1.5 m and above 2.0 m. This is the case the hysteresis")
    print("        band cannot catch, and where the original single-frame")
    print("        decision oscillates.\n")

    for case, (mean, sigma) in (("A", (1.5, 0.05)), ("B", (1.75, 0.35))):
        for confirm_frames, label in (
                (1, "confirm_frames=1 (original behaviour)"),
                (5, "confirm_frames=5 (this integration)")):
            cfg.switching.confirm_frames = confirm_frames
            policy = CameraSwitchPolicy(cfg, events)
            mgr = StubManager(debounce=0.0)   # debounce off, to isolate policy
            rng = np.random.default_rng(1)
            for _ in range(300):
                d = mean + float(rng.normal(0.0, sigma))
                policy.evaluate(mgr, mgr.get_active_camera(), d, 100.0)
            print(f"   case {case}  {label:<40} -> "
                  f"{mgr.switches:3d} switch(es) / 300 frames")
        print()

    print("\nA genuine approach must still switch exactly once:\n")
    cfg.switching.confirm_frames = 5
    policy = CameraSwitchPolicy(cfg, events)
    mgr = StubManager(debounce=0.0)
    for d in np.linspace(3.0, 0.8, 60):
        policy.evaluate(mgr, mgr.get_active_camera(), float(d), 100.0)
    print(f"   3.0 m -> 0.8 m approach                  -> "
          f"{mgr.switches} switch(es), now on camera {mgr.active}")
    for d in np.linspace(0.8, 3.0, 60):
        policy.evaluate(mgr, mgr.get_active_camera(), float(d), 100.0)
    print(f"   0.8 m -> 3.0 m departure                 -> "
          f"{mgr.switches} switch(es) total, now on camera {mgr.active}")
    print()

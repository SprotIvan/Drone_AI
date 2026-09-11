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
from latency import BUDGET
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

        cfg = self.config

        # ⚠️ Was `CameraManager.FAR_CAMERA_ID` / `NEAR_CAMERA_ID`, i.e. the
        # constants 0 and 1 (audit finding C1). The comparison is now
        # against the indices the roles actually resolved to on this
        # machine, read from the manager that resolved them.
        wide_id = manager.wide_id
        tele_id = manager.tele_id

        want_wide = want_tele = False
        basis = ""

        if distance_m is not None:
            # PRIMARY PATH — metric distance with hysteresis.
            want_wide = distance_m < cfg.switch_to_near_below_m
            want_tele = distance_m > cfg.switch_to_far_above_m
            basis = f"{distance_m:.2f} m"
        elif max_box_height_px > 0:
            # FALLBACK — uncalibrated optics, pixel height only.
            want_wide = max_box_height_px > cfg.fallback_near_height_px
            want_tele = max_box_height_px < cfg.fallback_far_height_px
            basis = f"{max_box_height_px:.0f} px"
        else:
            # No target at all: decay both streaks so a stale streak cannot
            # trigger a switch later.
            self._near_streak = self._far_streak = 0
            return None

        self._near_streak = self._near_streak + 1 if want_wide else 0
        self._far_streak = self._far_streak + 1 if want_tele else 0

        need = cfg.confirm_frames

        if current_camera == tele_id and self._near_streak >= need:
            if manager.switch_to_wide():
                self._near_streak = 0
                self.last_reason = f"TELE -> WIDE ({basis})"
                return self.last_reason
            # Suppressed by the debounce or the camera is not open. Keep the
            # streak so the switch fires as soon as it is permitted.
        elif current_camera == wide_id and self._far_streak >= need:
            if manager.switch_to_tele():
                self._far_streak = 0
                self.last_reason = f"WIDE -> TELE ({basis})"
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

        # ── Rate meters, each fed AT the place its work happens ──
        #
        # ⚠️ These used to be one number (`_fps_ema`) plus a counter driven
        # from the UI thread. That could not answer the question that actually
        # matters — "is the loop running fast because the SENSOR is fast, or
        # because it is draining a backlog?" — and the UI-thread counter was
        # structurally incapable of exceeding ui.max_ui_fps.
        self._sensor_fps_ema = 0.0        # from SensorTimestamp deltas
        self._det_fps_ema = 0.0           # actual Hailo invocation rate
        self._frame_age_ms_ema = 0.0      # now - SensorTimestamp
        self._capture_wait_ms_ema = 0.0   # time blocked inside capture
        self._last_sensor_ns: Optional[int] = None
        self._last_detect_t: Optional[float] = None
        self._last_frame_camera: Optional[int] = None
        self._boottime_ok = hasattr(time, "clock_gettime") and hasattr(
            time, "CLOCK_BOOTTIME")
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
            # ⚠️ LOG EVERY LINE, not just the first.
            #
            # This used to log `msg.splitlines()[0]`. For an ordinary
            # exception that is fine, but CameraIdentityError's whole value
            # is in the lines AFTER the first: the list of cameras that
            # were found, their Ids, and the exact JSON block to paste.
            # Truncating it left an operator looking at "camera subsystem
            # could not start: camera role(s) WIDE, TELE are not
            # configured" with the remedy silently discarded — which is
            # how a deliberately helpful failure turns into a dead end.
            lines = msg.splitlines() or [msg]
            log.error("camera subsystem could not start: %s", lines[0])
            for line in lines[1:]:
                log.error("  %s", line)
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

        ⚠️ ORDER MATTERS (audit findings C1/M3). This runs AFTER the
        cameras are open, because until the roles are resolved to device
        indices there is no correct index to key the module's dict by.
        `geo.camera_focal_px` is empty before bind_roles(), so calling this
        earlier would have pushed nothing and silently left the module's
        own stale constants in charge.
        """
        geo = self.config.geometry
        # Replace, do not merge: a leftover entry for an index that no
        # longer holds that role is exactly the stale copy this exists to
        # prevent.
        tc.CAMERA_FOCAL_PX = {
            camera_id: (None if focal is None else float(focal))
            for camera_id, focal in geo.camera_focal_px.items()}
        if geo.drone_real_width_m is not None:
            tc.DRONE_REAL_WIDTH_M = float(geo.drone_real_width_m)
        log.debug("optics synced from config: focal=%s width=%s",
                  tc.CAMERA_FOCAL_PX, tc.DRONE_REAL_WIDTH_M)

    def _setup(self) -> None:
        import TWO_CAMERAS_FIXED as tc
        from camera_manager import CameraManager

        geo = self.config.geometry
        v = self.config.visual
        log.info("opening cameras (%dx%d @ %d fps)...",
                 v.frame_width, v.frame_height, v.fps)

        # ⚠️ Fail-closed on identity (audit finding C1). CameraManager
        # resolves each role from the camera's stable device-tree Id and
        # RAISES if that is ambiguous, missing or unconfigured. The raise
        # propagates to _run(), which reports the camera subsystem OFFLINE
        # with the operator-ready message and leaves the acoustic
        # subsystem running. A wrong role is worse than no video: it is
        # video with every distance, cue and lens choice silently
        # transposed.
        self._manager = CameraManager(
            width=v.frame_width, height=v.frame_height, fps=v.fps,
            max_fps=v.max_fps, buffer_count=v.buffer_count,
            debounce_interval=self.config.switching.debounce_interval_s,
            warmup_frames=v.warmup_frames,
            role_hints=geo.camera_role_id_hint,
            expected_sensor_model=geo.expected_sensor_model,
            lens_mm=geo.lens_mm)

        # ── Adopt the focal lengths derived from the REAL sensor mode ──
        # Still theoretical, but no longer dependent on an assumption
        # about which mode libcamera picked. See _derive_focal_px.
        for role, focal in self._manager.focal_px_by_role.items():
            geo.camera_focal_px_by_role[role] = focal
            geo.camera_focal_source[role] = \
                self._manager.focal_source_by_role.get(
                    role, "THEORETICAL(runtime-mode)")

        # Role -> device index, once, now that the mapping is known.
        geo.bind_roles(self._manager.roles)
        self._sync_optics(tc)

        for line in geo.focal_status_lines():
            log.warning("OPTICS NOT CALIBRATED: %s", line)

        opened = self._manager.available_cameras()
        log.info("cameras opened: %s (active %s)",
                 {i: self._manager.role_for(i) for i in opened},
                 self._manager.role_for(self._manager.get_active_camera()))
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

    # ── Measurement helpers ────────────────────────────────────

    @staticmethod
    def _ema(previous: float, sample: float, alpha: float = 0.1) -> float:
        """
        Exponential average that SEEDS on the first sample.

        ⚠️ The previous form was `0.9*prev + 0.1*x if prev else x`, which seeds
        correctly but was fed a first interval of a few microseconds (the loop
        set `last_time` immediately before entering the loop and read the clock
        again immediately after). That seeded the camera FPS at ~10^5 and it
        took ~77 frames — about 2.5 s — of 0.9 decay to come back to reality.
        The first interval is now discarded by the callers instead.
        """
        return (1.0 - alpha) * previous + alpha * sample if previous else sample

    def _sensor_now_ns(self) -> Optional[int]:
        """
        'Now' on the SAME clock libcamera stamps frames with.

        libcamera's SensorTimestamp is CLOCK_BOOTTIME. time.monotonic() is
        CLOCK_MONOTONIC. On Linux the two differ by however long the machine
        has been suspended — zero on a Pi that never suspends, but comparing
        them is still wrong on principle and would silently produce a constant
        offset in every frame-age figure if the Pi ever did suspend.
        """
        if not self._boottime_ok:
            return None
        return int(time.clock_gettime(time.CLOCK_BOOTTIME) * 1e9)

    # ── Main loop ──────────────────────────────────────────────

    def _loop(self) -> None:
        import cv2
        import TWO_CAMERAS_FIXED as tc

        v = self.config.visual
        consecutive_failures = 0
        # None, not now(): the first iteration must not produce an interval.
        # See _ema() — a microsecond-long first interval used to seed the FPS
        # average at ~100000 and poison it for the first seconds of the run.
        last_time: Optional[float] = None

        log.info("camera loop running")

        while not self._stop.is_set():
            frame_start = time.monotonic()

            frame_raw, sensor_ns = self._manager.get_frame_and_sensor_ns()
            capture_done = time.monotonic()
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

            # ⚠️ BUG-002. WHICH CAMERA PRODUCED **THIS** FRAME.
            #
            # Read once, here, immediately after the capture returned — and
            # never re-read for the rest of this iteration. Later in the
            # loop the switching policy may select the other camera, and an
            # earlier version re-read `get_active_camera()` after that and
            # published the NEW id alongside THIS (old) camera's image.
            #
            # The consequence was not cosmetic: sensor_fusion projects the
            # acoustic cue with `visual.active_camera`'s boresight and focal
            # length (FAR 1274 px / 28 deg vs NEAR 502 px / 65 deg) and
            # evaluates the camera-range gate with it, so for one frame after
            # every switch the cue was drawn through the wrong optics onto
            # the wrong picture.
            #
            # A switch decided during this iteration takes effect on the NEXT
            # capture, which is exactly when the frame will actually come
            # from the other sensor.
            frame_camera = self._manager.get_active_camera()

            self._frame_count += 1

            # ── A. SENSOR rate and frame age — the physical measurements ──
            #
            # Recorded here, before any processing, from the driver's own
            # timestamp. `sensor_fps` is the ONLY number in this file that
            # describes the camera; everything else describes this software.
            self._capture_wait_ms_ema = self._ema(
                self._capture_wait_ms_ema, (capture_done - frame_start) * 1000.0)
            if frame_camera != self._last_frame_camera:
                # A switch restarts capture on the other sensor, so the
                # interval spanning it is a switch cost, not a frame period.
                # Counting it would drag the reported sensor rate down once
                # per switch for no reason.
                self._last_sensor_ns = None
                first_frame = self._last_frame_camera is None
                self._last_frame_camera = frame_camera
                if not first_frame:
                    self._reset_camera_state(frame_camera)
            if sensor_ns is not None:
                if self._last_sensor_ns is not None:
                    d_ns = sensor_ns - self._last_sensor_ns
                    if d_ns > 0:
                        self._sensor_fps_ema = self._ema(
                            self._sensor_fps_ema, 1e9 / d_ns)
                self._last_sensor_ns = sensor_ns
                now_ns = self._sensor_now_ns()
                if now_ns is not None:
                    self._frame_age_ms_ema = self._ema(
                        self._frame_age_ms_ema, (now_ns - sensor_ns) / 1e6)

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
                # ── B. HAILO INVOCATION RATE, counted where it happens ──
                #
                # ⚠️ This used to be counted in main.py's UI loop, inside the
                # `if due:` block. That block runs at most `ui.max_ui_fps`
                # times a second (20), and only looks at whichever
                # VisualObservation happens to be current — so the reported
                # "Hailo / YOLO fps" could never exceed 20 and silently
                # skipped every frame that arrived between UI ticks. It was
                # not a Hailo measurement at all; it was a UI measurement
                # wearing Hailo's name.
                if self._last_detect_t is not None:
                    gap = t0 - self._last_detect_t
                    if gap > 0:
                        self._det_fps_ema = self._ema(self._det_fps_ema,
                                                      1.0 / gap)
                self._last_detect_t = t0
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

                # ── WHERE THE 29 ms ACTUALLY GOES ──
                #
                # The detector reports its own four internal stages (see
                # HailoInference.stage_ms). Recording them here, in the
                # thread that made the call, is what distinguishes "the NPU
                # is saturated" from "the NPU is idle while numpy does
                # post-processing on the CPU" — two problems with opposite
                # fixes, which a single aggregate timing cannot tell apart.
                stages = getattr(self._detector, "stage_ms", None)
                if stages:
                    for stage, value in stages.items():
                        # None = the stage was skipped on this frame (NMS is,
                        # whenever nothing was detected). Skipped is not zero.
                        if value is not None:
                            BUDGET.record("hailo_" + stage, float(value))

            # ── Tracking ──
            try:
                self._tracker.process_frame(frame_bgr, detections,
                                            run_association=run_detector,
                                            detection_scores=scores)
            except Exception as exc:
                self.events.rate("tracker-error", logging.ERROR,
                                 "tracker failed: %s", exc)

            # Everything that DESCRIBES this frame uses frame_camera: the
            # optics that produced the pixels are the optics the distances,
            # the cue and the range gate must be computed with.
            tracks, max_box_h, max_box_w = self._collect_tracks(frame_camera)

            # ── Camera selection ──
            # Decided FROM this frame, applied to the NEXT one.
            switch_distance = tc.estimate_distance_m(max_box_w, frame_camera)
            try:
                switched = self._policy.evaluate(self._manager, frame_camera,
                                                 switch_distance, max_box_h)
                if switched:
                    # Deliberately does NOT update frame_camera — see above.
                    log.info("camera switched: %s (takes effect next frame)",
                             switched)
            except Exception as exc:
                self.events.rate("switch-error", logging.WARNING,
                                 "camera switch failed: %s", exc)

            if self._calibrate_request.is_set():
                self._calibrate_request.clear()
                self._print_calibration(frame_camera, max_box_w)

            # ── Publish ──
            # ── C. LOOP rate — PROCESSING throughput, not the frame rate ──
            if last_time is not None:
                dt = frame_start - last_time
                if dt > 0:
                    self._fps_ema = self._ema(self._fps_ema, 1.0 / dt)
            last_time = frame_start

            self._seq += 1
            self.latest.publish(VisualObservation(
                frame=frame_bgr,
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
                tracks=tuple(tracks),
                detector_ran=run_detector,
                detection_count=len(detections),
                active_camera=frame_camera,
                camera_name=self._camera_name(frame_camera),
                available_cameras=tuple(self._manager.available_cameras()),
                inference_ms=(self._infer_ema if self._infer_ema else None),
                tracker_ms=None,
                loop_fps=self._fps_ema,
                sensor_fps=(self._sensor_fps_ema
                            if self._sensor_fps_ema else None),
                detector_fps=self._det_fps_ema,
                frame_age_ms=(self._frame_age_ms_ema
                              if self._last_sensor_ns is not None else None),
                capture_wait_ms=self._capture_wait_ms_ema,
                timestamp=now(),
                seq=self._seq))

            self.events.rate(
                "camera-heartbeat", logging.DEBUG,
                "camera: sensor %.1f fps | loop %.1f fps | detector %.1f fps | "
                "infer %.0f ms | frame age %.0f ms | capture wait %.1f ms | "
                "%d track(s) | cam %d",
                self._sensor_fps_ema, self._fps_ema, self._det_fps_ema,
                self._infer_ema, self._frame_age_ms_ema,
                self._capture_wait_ms_ema, len(tracks), frame_camera,
                interval=5.0)

    # ── Camera switch: discard the other camera's state ────────

    def _reset_camera_state(self, new_camera: int) -> None:
        """
        Drop every piece of state that belonged to the PREVIOUS camera.

        ═══════════════════════════════════════════════════════════
        ⚠️ AUDIT FINDING C4 — WHY THIS MUST HAPPEN
        ═══════════════════════════════════════════════════════════

        `AdvancedADASTracker` is constructed once in _setup() and was never
        told that the camera changed. Two things then went wrong on the
        first frame after every switch, and neither raised an error:

        1. OPTICAL FLOW ACROSS TWO DIFFERENT LENSES.
           `EgoMotionEstimator.prev_gray` still held the LAST FRAME OF THE
           OTHER CAMERA, so `estimate_motion()` ran Lucas-Kanade between a
           WIDE frame and a TELE frame. Those are different fields of view
           pointing along slightly different axes; the flow field between
           them is not camera motion and the affine warp fitted to it is
           meaningless. That warp was then applied to every live track via
           `warp_all_models()`.

        2. TRACK COORDINATES IN THE WRONG PIXEL SCALE.
           An IMM track holds a position, velocity and box size in PIXELS.
           With the lenses on this station the angular scale changes by
           25/6 = 4.17x in one frame, so a track carried across the switch
           describes a target that appears to jump and resize violently.
           The filter's response to that is to either diverge or to spend
           several frames dragging the estimate across the frame, and
           `estimate_distance_m()` is reading the box width the whole time.

        Clearing both is the correct behaviour rather than a workaround:
        after a switch there genuinely is no prior observation of the
        target through THIS lens, so the honest state is no state. The
        detector re-acquires on the next frame; a Tentative track needs a
        few frames to confirm, which is the real, unavoidable cost of a
        switch and is why switching is debounced and confirmed rather than
        done per frame.
        """
        # ── Ego motion: prev_gray, prev_pts and the frame counter ──
        estimator = getattr(self._tracker, "ego_estimator", None)
        if estimator is not None:
            try:
                estimator.reset()
            except Exception as exc:
                self.events.rate("switch-reset-error", logging.ERROR,
                                 "ego-motion reset failed on camera "
                                 "switch: %s", exc)

        # ── Tracks: pixel state from the other lens ──
        dropped = 0
        try:
            dropped = len(self._tracker.tracks)
            self._tracker.tracks.clear()
        except Exception as exc:
            self.events.rate("switch-reset-error", logging.ERROR,
                             "track reset failed on camera switch: %s", exc)

        # The switch also invalidates the detector-rate interval, for the
        # same reason the sensor interval is dropped above: the gap spans a
        # pipeline restart, not a frame period.
        self._last_detect_t = None

        log.info("camera switch -> %s: cleared %d track(s) and the "
                 "ego-motion reference (optical flow must never cross two "
                 "different lenses)",
                 self._camera_name(new_camera), dropped)

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
        """
        Label for the HUD and the logs: the camera's ROLE plus its sensor.

        ⚠️ This used to return "FAR/IMX477" for index 0 and "NEAR/IMX708"
        for index 1 (audit findings C1 and C3). Both parts were wrong.
        There is no IMX708 on this station — both cameras are IMX477P — so
        the operator was shown a sensor that is not installed. And the
        label was chosen by INDEX, so if libcamera ever reordered the
        cameras the HUD would confidently mislabel which lens the picture
        came from. The name now comes from the resolved role binding.
        """
        if self._manager is not None:
            return f"{self._manager.role_for(camera_id)}/IMX477P"
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
        """PROCESSING rate of this loop. See `sensor_fps` for the camera."""
        return self._fps_ema

    @property
    def sensor_fps(self) -> float:
        """Physical new-frame rate from SensorTimestamp. 0.0 = not measurable."""
        return self._sensor_fps_ema

    @property
    def detector_fps(self) -> float:
        """Rate at which the Hailo forward pass actually runs."""
        return self._det_fps_ema

    @property
    def frame_age_ms(self) -> float:
        """Capture-to-publish latency. Large = frames were queued (stale)."""
        return self._frame_age_ms_ema

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
        """
        Mimics CameraManager's switching surface, including the debounce.

        ⚠️ Exposes `wide_id` / `tele_id`, the RESOLVED role indices, not the
        old FAR_CAMERA_ID / NEAR_CAMERA_ID class constants (finding C1). The
        indices below are an arbitrary but consistent stand-in for a
        resolved mapping; on real hardware they come from the device Ids.
        """
        tele_id = 0          # the long lens
        wide_id = 1          # the wide lens

        def __init__(self, debounce=0.5):
            self.active = self.tele_id
            self.debounce = debounce
            self.last = 0.0
            self.switches = 0

        def get_active_camera(self):
            return self.active

        def role_for(self, camera_id):
            return {self.tele_id: "TELE", self.wide_id: "WIDE"}.get(
                camera_id, f"CAM{camera_id}")

        def _switch(self, target):
            if self.active == target:
                return False
            if time.monotonic() - self.last < self.debounce:
                return False
            self.active = target
            self.last = time.monotonic()
            self.switches += 1
            return True

        def switch_to_wide(self):
            return self._switch(self.wide_id)

        def switch_to_tele(self):
            return self._switch(self.tele_id)

        switch_to_near = switch_to_wide
        switch_to_far = switch_to_tele

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

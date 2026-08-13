#!/usr/bin/env python3
"""
main.py — Unified drone detection station. ONE command, ONE window.

    python main.py

Starts, in order:
    1. logging
    2. configuration (fusion_config.json + radar_calibration.json)
    3. acoustic worker  — microphone, acoustic ML, DOA, ranging
    4. camera worker    — Picamera2, Hailo/YOLO26n, IMM tracker, switching
    5. sensor fusion    — priority state machine
    6. unified UI       — camera view with the acoustic radar overlaid

and shuts all of it down cleanly on 'q', ESC, window close, or Ctrl+C.

═══════════════════════════════════════════════════════════════════
THREADING
═══════════════════════════════════════════════════════════════════

    ┌──────────────┐   AcousticObservation   ┌────────────────┐
    │ audio thread │ ──────────────────────► │                │
    │   ~2 Hz      │      (LatestValue)      │  SensorFusion  │
    └──────────────┘                         │   + HUD        │
    ┌──────────────┐    VisualObservation    │  (main thread) │
    │ camera thread│ ──────────────────────► │                │
    │   ~30 Hz     │      (LatestValue)      └───────┬────────┘
    └──────────────┘                                 │ LedFrame
                                             ┌───────▼────────┐
                                             │   LED thread   │
                                             │  (subprocess   │
                                             │   writes only) │
                                             └────────────────┘

Neither sensor thread ever waits on the other or on the UI: they publish
into a one-slot mailbox and move on. The UI never waits on a sensor: it
renders whatever the newest snapshot is, including "nothing yet" and
"that subsystem is offline".

The LED ring hangs off the SAME FusedTarget the HUD renders, so the ring,
the radar widget and the camera cue cannot show three different opinions
about the target. It is a third mailbox for the same reason as the other
two: one LED write is a subprocess spawn (200-500 ms), which would stall
the UI thread and, through the GIL, the sensors behind it.

The UI runs on the MAIN thread because `cv2.imshow`/`waitKey` require it on
macOS and are only reliably safe there on Linux/X11 too.

Both workers are daemon threads, so a device call stuck in the kernel
(a wedged PortAudio read, a hung CSI capture) can never prevent the process
from exiting.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
# Allow `python main.py` to work from any working directory: the existing
# modules use flat imports (`from camera_manager import ...`), which need
# this directory on the path.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2  # noqa: E402

import fusion_config  # noqa: E402
import station_logging  # noqa: E402
from acoustic_worker import AcousticWorker  # noqa: E402
from camera_cue import BearingProjector  # noqa: E402
from camera_worker import CameraWorker  # noqa: E402
from hud import HUD  # noqa: E402
from respeaker_led import RespeakerLed, frame_for_target  # noqa: E402
from sensor_fusion import SensorFusion, SystemState  # noqa: E402
from station_logging import EventLogger  # noqa: E402
from target_state import SubsystemHealth, SubsystemState, now  # noqa: E402

DEFAULT_HEF = str(BASE_DIR / "bestty_yolo260508.hef")

log = logging.getLogger("station.main")


# ═══════════════════════════════════════════════════════════════
#  Startup banner
# ═══════════════════════════════════════════════════════════════

def print_banner(config: fusion_config.StationConfig, hef_path: str) -> None:
    log.info("=" * 62)
    log.info("DRONE DETECTION STATION — acoustic + visual sensor fusion")
    log.info("=" * 62)

    for line in config.describe_calibration():
        log.info("  %s", line)

    # The acoustic bearing, the camera cue and the LED ring all express the
    # same direction, but only if they share one frame of reference. A
    # change to doa_offset_deg rotates that frame under every boresight
    # already measured, and nothing about the resulting numbers looks wrong.
    import calibration
    for line in config.check_bearing_frames(calibration.load()):
        log.warning("  %s", line)

    gate = config.activation_distance_m(0)
    release = config.release_distance_m(0)
    if gate is not None:
        log.info("  camera-range gate: acquire <= %.0f m, release > %.0f m",
                 gate, release or gate)
        log.info("  ^ DERIVED from optics, not assumed. A %.2f m drone spans "
                 "%.0f px at %.0f m on the FAR camera, which is the smallest "
                 "box this configuration expects the detector to resolve.",
                 config.geometry.drone_real_width_m or 0.0,
                 config.fusion.min_detectable_box_px, gate)
    else:
        log.warning("  camera-range gate: NOT COMPUTABLE (optics "
                    "uncalibrated) — visual acquisition will run on acoustic "
                    "confirmation alone")

    log.info("  camera switch: NEAR below %.2f m, FAR above %.2f m, "
             "%d-frame confirmation",
             config.switching.switch_to_near_below_m,
             config.switching.switch_to_far_above_m,
             config.switching.confirm_frames)

    led = config.led
    if not led.enabled:
        log.info("  LED ring: disabled")
    elif led.led_count is None:
        log.warning("  LED ring: led.led_count NOT SET -> ring not driven. "
                    "Count the LEDs on your array and set it in "
                    "fusion_config.json (run: python respeaker_led.py --probe)")
    else:
        log.info("  LED ring: %d LEDs, %d-LED target sector, zero at %.0f deg",
                 led.led_count, led.sector_leds, led.led_zero_offset_deg)

    log.info("  HEF: %s", hef_path)
    log.info("-" * 62)


# ═══════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════

class Station:
    """Owns the workers, the fusion layer and the window."""

    def __init__(self, config: fusion_config.StationConfig, hef_path: str,
                 events: EventLogger, headless: bool = False,
                 no_audio: bool = False, no_camera: bool = False,
                 no_led: bool = False):
        self.config = config
        self.events = events
        self.headless = headless
        self.running = True

        self.projector = BearingProjector(config.geometry,
                                          config.visual.frame_width)
        self.fusion = SensorFusion(config, self.projector)
        self.hud = HUD(config, self.projector)

        self.acoustic: Optional[AcousticWorker] = None
        self.camera: Optional[CameraWorker] = None

        # The LED ring needs the DOA mount offset to convert a displayed
        # bearing back into the array's own frame — the ring is bolted to
        # the array, not to the compass. calibration.load() is the same
        # source the acoustic worker uses, so the two can never disagree.
        import calibration
        led_cfg = config.led
        if no_led:
            led_cfg = fusion_config.LedConfig(enabled=False)
        self.led = RespeakerLed(led_cfg, calibration.load())

        self._disabled = SubsystemHealth(SubsystemState.DISABLED,
                                         "disabled on the command line")

        if not no_audio:
            self.acoustic = AcousticWorker(config, events)
        if not no_camera:
            self.camera = CameraWorker(config, events, hef_path)

        self._ui_dt = 1.0 / max(config.ui.max_ui_fps, 1.0)
        self._last_render = 0.0
        self._ui_fps = 0.0
        self._frames = 0
        self._started = now()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        # Workers are started in parallel and each reports its own health.
        # Neither is allowed to block the other's startup, so a camera that
        # takes seconds to warm up does not delay the microphone.
        if self.acoustic is not None:
            self.acoustic.start()
        if self.camera is not None:
            self.camera.start()
        # Started last and never waited on: the ring is an indicator, and
        # an indicator must not be able to delay the sensors it indicates.
        self.led.start()

    def stop(self) -> None:
        self.running = False

    def shutdown(self) -> None:
        log.info("shutting down...")
        # The ring is stopped FIRST so it is restored to a safe state while
        # the process is still healthy. A ring left showing a red target
        # after the station exited would be actively misleading.
        self.led.stop(timeout=2.0)
        for worker in (self.camera, self.acoustic):
            if worker is not None:
                worker.stop()
        for worker in (self.camera, self.acoustic):
            if worker is not None:
                worker.join(timeout=3.0)

        if not self.headless:
            try:
                cv2.destroyAllWindows()
                # On some builds the window only actually closes after a
                # further waitKey pump.
                for _ in range(4):
                    cv2.waitKey(1)
            except Exception:
                pass

        uptime = now() - self._started
        log.info("session: %.0f s, %d UI frames (%.1f fps average)",
                 uptime, self._frames, self._frames / max(uptime, 1e-6))
        if self.acoustic is not None and self.acoustic.mean_block_ms:
            log.info("acoustic block time: mean %.0f ms (budget 500 ms)",
                     self.acoustic.mean_block_ms)
        if self.camera is not None:
            log.info("camera: %.1f fps, inference %.0f ms",
                     self.camera.fps, self.camera.inference_ms)
        log.info("LED ring: %s, %d hardware writes",
                 self.led.status.value, self.led.writes)
        log.info("stopped.")

    # ── Health helpers ─────────────────────────────────────────

    # ⚠️ A worker reports its own health, which means a worker that is STUCK
    # cannot report anything at all. Both sensor threads can block
    # indefinitely inside a driver call — PortAudio's read() or picamera2's
    # capture_array() — and a thread blocked in the kernel keeps its last
    # published health forever. Before this watchdog the lamps stayed green
    # and the state machine kept trusting a subsystem that had stopped
    # producing data minutes earlier.
    #
    # The Station is the only party that can notice, because it is the one
    # still running. Silence for longer than the sensor's own lost-timeout
    # is treated as failure regardless of what the worker last claimed.

    def _watchdog(self, health: SubsystemHealth, last, lost_after: float,
                  label: str) -> SubsystemHealth:
        if not health.ok or last is None:
            return health
        age = now() - last.timestamp
        if age <= lost_after:
            return health
        return health.with_state(
            SubsystemState.OFFLINE,
            f"no data for {age:.1f}s (thread stalled?)",
            f"{label} watchdog timeout")

    def _acoustic_health(self) -> SubsystemHealth:
        if self.acoustic is None:
            return self._disabled
        health = self.acoustic.health.get() or SubsystemHealth()
        return self._watchdog(health, self.acoustic.latest.get(),
                              self.config.acoustic.lost_after_s * 3.0,
                              "acoustic")

    def _led_lamp_state(self) -> Optional[SubsystemState]:
        """
        Lamp state for the LED ring, or None to hide the lamp entirely.

        Hidden when the ring is switched off, so a station that does not use
        it looks exactly as it did before this feature existed. Shown in
        every other case — including UNAVAILABLE, which is the state an
        operator most needs to see, because it means the hardware
        indication they may be relying on is not actually running.
        """
        # self.led.cfg, not self.config.led: --no-led substitutes a disabled
        # copy, and the lamp must follow what is actually running.
        if not self.led.cfg.enabled:
            return None
        return self.led.status.as_subsystem_state()

    def _visual_health(self) -> SubsystemHealth:
        if self.camera is None:
            return self._disabled
        health = self.camera.health.get() or SubsystemHealth()
        return self._watchdog(health, self.camera.latest.get(),
                              max(self.config.visual.lost_after_s * 3.0, 2.0),
                              "camera")

    # ── Main loop ──────────────────────────────────────────────

    def run(self) -> None:
        window = self.config.ui.window_name
        if not self.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if self.config.ui.fullscreen:
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)

        last_state: Optional[SystemState] = None
        # Start one UI period in the past so the first pass renders
        # immediately with a sane dt. Initialising to 0.0 made the first
        # `since_render` the entire monotonic clock (~10^5 s), which fed a
        # nonsense dt into the radar sweep and pinned the UI-FPS average
        # near zero for the first seconds.
        last_render = time.monotonic() - self._ui_dt
        last_visual_seq = -1

        # ⚠️ PERFORMANCE-CRITICAL LOOP. See the comment block below before
        # changing anything here.
        #
        # The first version of this loop rendered the full HUD on EVERY
        # iteration, paced only by cv2.waitKey(1). That is not a frame rate,
        # it is a spin: the HUD was recomposited (resize + radar + every
        # overlay) over and over on the *same* camera frame, hundreds of
        # times a second. On a Raspberry Pi 5 that saturates one core
        # permanently and starves the camera and audio threads through both
        # CPU contention and the GIL — measured effect on target hardware:
        # camera 37 fps -> 19 fps, and the audio thread missing blocks badly
        # enough that acoustic detection stopped working at all.
        #
        # `self._ui_dt` was computed and then never used, so the configured
        # max_ui_fps had no effect whatsoever.
        #
        # Now: the HUD is recomposited only when there is something new to
        # show, and never faster than max_ui_fps. Everything else is spent
        # inside cv2.waitKey(), which blocks in C with the GIL released and
        # therefore actively hands the CPU to the sensor threads.
        idle_interval = 0.25          # refresh at 4 Hz with no new frame,
                                      # so the radar sweep and acoustic
                                      # readouts still animate

        while self.running:
            now_t = time.monotonic()

            visual_obs = (self.camera.latest.get()
                          if self.camera is not None else None)
            new_frame = (visual_obs is not None
                         and visual_obs.seq != last_visual_seq)
            since_render = now_t - last_render

            due = since_render >= self._ui_dt and (new_frame
                                                   or since_render >= idle_interval)

            if due:
                acoustic_obs = (self.acoustic.latest.get()
                                if self.acoustic is not None else None)

                target = self.fusion.update(
                    acoustic_obs, visual_obs,
                    self._acoustic_health(), self._visual_health())

                # ── The one authoritative state, fanned out ──
                # LED ring, radar widget and camera cue are all rendered
                # from THIS `target` and nothing else. The ring is fed here,
                # immediately after the state is computed and before any
                # drawing, so it can never be driven by a repaint cycle:
                # frame_for_target() is a pure function of the fused state,
                # and submit() is a single publish into a one-slot mailbox
                # (all device I/O happens on the LED worker thread).
                self.led.submit(frame_for_target(target))

                if target.state is not last_state:
                    self._announce(target, last_state)
                    last_state = target.state

                if not self.headless:
                    # The FPS shown is the CAMERA pipeline rate — the number
                    # the original application displayed and the one that
                    # actually matters for detection. The UI's own refresh
                    # rate is reported separately and deliberately smaller.
                    camera_fps = (visual_obs.loop_fps
                                  if visual_obs is not None else 0.0)
                    frame = self.hud.render(
                        target, since_render, camera_fps=camera_fps,
                        ui_fps=self._ui_fps, led_state=self._led_lamp_state())
                    cv2.imshow(window, frame)

                inst = 1.0 / max(since_render, 1e-6)
                self._ui_fps = (0.9 * self._ui_fps + 0.1 * inst
                                if self._ui_fps else inst)
                last_render = now_t
                if visual_obs is not None:
                    last_visual_seq = visual_obs.seq
                self._frames += 1

            if self.headless:
                # No window to pump; just yield.
                time.sleep(0.005)
                continue

            # Sleep INSIDE waitKey, which releases the GIL — this is what
            # hands the CPU to the camera and audio threads.
            #
            # Two different waits, because there are two reasons to be here:
            #
            #   • rate-limited: the UI period has not elapsed. Sleep exactly
            #     the remainder — the next render is due at a known time.
            #   • waiting for a frame: the period HAS elapsed and there is
            #     simply nothing new to draw. The next render time is
            #     unknown, so poll.
            #
            # The second case used to fall through the same expression and
            # produce a negative remainder, clamped to 1 ms — i.e. a 1000 Hz
            # poll for as long as the camera was slow or dead, which is
            # exactly when CPU matters most. Polling at ~5 ms bounds the
            # added display latency well under one camera frame while
            # costing 200 wakeups/s instead of 1000.
            remaining = self._ui_dt - (time.monotonic() - last_render)
            wait_ms = int(round(remaining * 1000.0)) if remaining > 0.001 else 5
            if not self._handle_keys(window, max(1, min(20, wait_ms))):
                break

    def _announce(self, target, previous: Optional[SystemState]) -> None:
        """One clear line per state change — the operator-facing narrative."""
        state = target.state
        acoustic = target.acoustic
        if state is SystemState.ACOUSTIC_DETECTED:
            log.info("ACOUSTIC CONTACT: bearing %s, range %s, confidence %.0f%%",
                     target.bearing_text(),
                     acoustic.distance_text() if acoustic else "N/A",
                     (acoustic.p_smoothed * 100.0) if acoustic else 0.0)
        elif state is SystemState.ACOUSTIC_TRACKING:
            log.info("ACOUSTIC TRACKING: microphone is the primary sensor")
        elif state is SystemState.VISUAL_ACQUISITION:
            log.info("VISUAL ACQUISITION: target should be within camera "
                     "range — searching visually")
        elif state is SystemState.VISUAL_TRACKING:
            log.info("VISUAL TRACKING: camera is now the primary sensor "
                     "(YOLO confidence %s)", target.visual_confidence_text())
        elif state is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING:
            log.warning("VISUAL TRACK LOST — falling back to acoustic "
                        "tracking for reacquisition")
        elif state is SystemState.TARGET_LOST:
            log.warning("TARGET LOST: both sensors have lost contact")
        elif state is SystemState.SEARCHING and previous is not None:
            log.info("SEARCHING: no target")

    # ── Keyboard ───────────────────────────────────────────────

    def _handle_keys(self, window: str, wait_ms: int = 1) -> bool:
        """
        Pump the GUI event loop and handle a key. Returns False to exit.

        `wait_ms` is where the UI thread spends its idle time. cv2.waitKey
        blocks in C with the GIL released, so a longer wait here directly
        buys CPU for the camera and audio threads.
        """
        key = cv2.waitKey(max(1, int(wait_ms))) & 0xFF

        # Closing the window with the title-bar X must also stop the app.
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                log.info("window closed")
                return False
        except cv2.error:
            return False

        if key in (ord("q"), 27):            # q / ESC
            log.info("quit requested")
            return False
        if key == ord("h"):
            self.hud.show_help = not self.hud.show_help
        elif key == ord("f") and self.camera is not None:
            frozen = not self.camera.switching_frozen
            self.camera.freeze_switching(frozen)
            log.info("camera switching %s",
                     "FROZEN" if frozen else "RESUMED")
        elif key == ord("c") and self.camera is not None:
            self.camera.request_focal_calibration()
        elif key == ord("r"):
            self.fusion.reset()
        return True


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified acoustic + visual drone detection station")
    parser.add_argument("--config", default=str(fusion_config.CONFIG_PATH),
                        help="path to fusion_config.json")
    parser.add_argument("--hef", default=DEFAULT_HEF,
                        help="path to the YOLO26n .hef model")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="run every subsystem but open no window")
    parser.add_argument("--no-audio", action="store_true",
                        help="run without the acoustic subsystem")
    parser.add_argument("--no-camera", action="store_true",
                        help="run without the camera subsystem")
    parser.add_argument("--no-led", action="store_true",
                        help="run without the ReSpeaker LED ring indicator")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--duration", type=float, default=None,
                        help="exit after N seconds (for testing)")
    args = parser.parse_args(argv)

    config = fusion_config.load(args.config)
    if args.log_level:
        config.logging.level = args.log_level
    if args.fullscreen:
        config.ui.fullscreen = True

    station_logging.setup(config.logging, BASE_DIR)
    events = EventLogger(logging.getLogger("station"), config.logging)

    if args.no_audio and args.no_camera:
        log.error("--no-audio and --no-camera together leave nothing to run")
        return 2

    print_banner(config, args.hef)

    station = Station(config, args.hef, events, headless=args.headless,
                      no_audio=args.no_audio, no_camera=args.no_camera,
                      no_led=args.no_led)

    # Ctrl+C must run the same orderly shutdown as 'q'.
    def on_signal(signum, _frame):
        log.info("signal %s received", signum)
        station.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError, AttributeError):
            pass          # not available on this platform / not main thread

    station.start()

    exit_code = 0
    try:
        if args.duration is not None:
            # Test/CI mode: stop the station after a fixed wall-clock time.
            import threading
            stopper = threading.Timer(args.duration, station.stop)
            stopper.daemon = True
            stopper.start()
        station.run()
    except KeyboardInterrupt:
        log.info("interrupted")
    except Exception:
        log.exception("station crashed")
        exit_code = 1
    finally:
        station.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

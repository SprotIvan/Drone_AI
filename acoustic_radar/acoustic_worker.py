#!/usr/bin/env python3
"""
acoustic_worker.py — Runs the acoustic pipeline on its own thread.

The DSP/ML pipeline itself is NOT reimplemented here. This is a thin,
carefully-bounded wrapper around the existing, working `radar.RadarEngine`:

    audio device ──► RadarEngine.process_block() ──► AcousticObservation
                     (features, ONNX, detector FSM,       (immutable
                      DOA, ranging — all untouched)        snapshot)

Everything about how a drone is recognised, how bearing is estimated and
how range is derived stays exactly where it was. This file only adds:

  • a thread so audio never waits on the camera or the UI;
  • translation of RadarStatus into the immutable observation the fusion
    layer consumes;
  • supervision — an exception in the audio pipeline marks the subsystem
    OFFLINE and (optionally) retries, instead of killing the process.

═══════════════════════════════════════════════════════════════════
WHY THE AUDIO LOOP MUST NOT BE BLOCKED
═══════════════════════════════════════════════════════════════════

`stream.read(BLOCK_SAMPLES)` returns 0.5 s of audio. If the loop takes
longer than 0.5 s to come back for the next block, PortAudio's ring buffer
overruns and audio is *lost* — the drone signature in that gap is simply
never analysed. The original radar_gui.py did two things inside this loop
that could stall it: a synchronous WAV write on every alarm, and a Telegram
call. Both are now off this thread (see AudioLogger dispatch below and
IntegrationConfig).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from fusion_config import StationConfig
from latency import BUDGET
from station_logging import EventLogger
from target_state import (AcousticObservation, LatestValue, SubsystemHealth,
                          SubsystemState, now)

log = logging.getLogger("station.acoustic")


class AcousticWorker:
    """
    Owns the microphone stream and the RadarEngine.

    Public surface used by main.py:
        start()      spawn the thread (returns False if it cannot start)
        latest       LatestValue[AcousticObservation]
        health       LatestValue[SubsystemHealth]
        stop()       request shutdown
        join()       wait for the thread to finish
    """

    def __init__(self, config: StationConfig, events: EventLogger,
                 calibration_cfg: Optional[dict] = None):
        self.config = config
        self.events = events
        self.calibration_cfg = calibration_cfg

        self.latest: LatestValue[AcousticObservation] = LatestValue()
        self.health: LatestValue[SubsystemHealth] = LatestValue(
            SubsystemHealth(SubsystemState.STARTING, "not started"))

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._overflow_count = 0
        self._engine = None            # radar.RadarEngine
        self._uart = None
        self._audio_logger = None
        self._alerter = None
        self._block_times: list[float] = []

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="acoustic",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                # A blocked PortAudio read cannot be interrupted from
                # outside; the thread is a daemon, so the process still
                # exits. Say so rather than hanging silently.
                log.warning("audio thread did not stop within %.1fs "
                            "(blocked in device read) — leaving it to the "
                            "daemon shutdown", timeout)

    def _set_health(self, state: SubsystemState, detail: str = "",
                    error: Optional[str] = None) -> None:
        current = self.health.get() or SubsystemHealth()
        self.health.publish(current.with_state(state, detail, error))

    # ── Thread body ────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._setup()
        except BaseException as exc:            # SystemExit included
            # RadarEngine raises SystemExit for a missing/mismatched model.
            # In a single-script world that ended the program; here it must
            # only take the acoustic subsystem offline.
            msg = str(exc) or exc.__class__.__name__
            log.error("acoustic subsystem could not start: %s",
                      msg.splitlines()[0] if msg else exc)
            self._set_health(SubsystemState.OFFLINE, "init failed", msg)
            return

        try:
            self._loop()
        except BaseException as exc:
            log.exception("acoustic loop crashed")
            self._set_health(SubsystemState.OFFLINE, "loop crashed", str(exc))
        finally:
            self._teardown()

    # ── Setup ──────────────────────────────────────────────────

    def _setup(self) -> None:
        import calibration
        import features
        from audio_io import open_stream, resolve_input
        from radar import BLOCK_SAMPLES, BLOCK_SEC, RadarEngine

        cfg = (self.calibration_cfg if self.calibration_cfg is not None
               else calibration.load())

        log.info("loading acoustic model...")
        tuning = self.config.acoustic.detector_tuning()
        self._engine = RadarEngine(cfg=cfg, verbose=False, tuning=tuning)
        log.info("acoustic model loaded (P_START %.3f, P_HOLD %.3f)",
                 self._engine.threshold, self._engine.detector.hold_limit)
        # These three numbers are the whole answer to "how fast does the
        # alarm come up and how long does it survive a dropout". They are
        # logged as MEASURED values derived from the real block period, not
        # as the intent behind them.
        hop = float(self.config.acoustic.block_seconds or BLOCK_SEC)
        log.info("acoustic timing: block %.0f ms x %d confirmations = "
                 "%.0f ms to alarm; coasting %.1f s (%d blocks)",
                 hop * 1000.0, tuning.confirm_blocks,
                 tuning.confirm_seconds(hop) * 1000.0,
                 tuning.coast_seconds(hop), tuning.miss_tolerance)
        if tuning.coast_seconds(hop) > self.config.acoustic.lost_after_s:
            # The fusion layer would declare the acoustic sensor lost while
            # the engine is still legitimately holding the target.
            log.warning("acoustic.lost_after_s (%.1f s) is SHORTER than the "
                        "detector's coasting window (%.1f s) — the fusion "
                        "layer will drop the target before the engine does",
                        self.config.acoustic.lost_after_s,
                        tuning.coast_seconds(hop))

        inp = resolve_input(cfg, features.SAMPLE_RATE)
        self._engine.attach_input(inp)
        log.info("microphone: %s", inp.describe())
        log.info("DOA: %s", self._engine.doa.describe()
                 if self._engine.doa else "unavailable")

        if not self._engine.ranger.calibrated:
            log.warning("acoustic range NOT calibrated — distance will be "
                        "reported as N/A (run: python calibrate.py range)")
        else:
            self._warn_if_range_calibration_looks_synthetic(cfg)

        # The hop may be overridden to trade CPU for latency; radar.py
        # remains the source of the default so `python radar.py` standalone
        # is unaffected. The engine's sliding window handles any hop.
        self._block_seconds = float(
            self.config.acoustic.block_seconds or BLOCK_SEC)
        self._block_samples = (
            BLOCK_SAMPLES if self._block_seconds == BLOCK_SEC
            else int(features.SAMPLE_RATE * self._block_seconds))
        if self._block_seconds != BLOCK_SEC:
            log.warning("audio hop overridden to %.0f ms (radar.py default "
                        "%.0f ms) — verify block_total stays well under it, "
                        "or the device buffer will overrun",
                        self._block_seconds * 1000.0, BLOCK_SEC * 1000.0)
        self._stream = open_stream(inp, self._block_samples)

        # ── Optional side channels ──
        integ = self.config.integrations
        if integ.uart_enabled:
            try:
                from radar import UartSender
                self._uart = UartSender()
            except Exception as exc:
                log.info("UART unavailable: %s", exc)

        if integ.audio_logging_enabled:
            try:
                from audio_logger import AudioLogger
                self._audio_logger = AudioLogger()
                log.info("audio logging: %s (%d saved)",
                         self._audio_logger.save_dir,
                         self._audio_logger.detection_count)
            except Exception as exc:
                log.info("audio logging unavailable: %s", exc)

        if integ.telegram_enabled:
            self._alerter = self._make_alerter()

        self._set_health(SubsystemState.ONLINE, inp.describe())

    @staticmethod
    def _warn_if_range_calibration_looks_synthetic(_cfg: dict) -> None:
        """
        Detect a hand-written radar_calibration.json.

        Why this check exists: `calibrate.py range` ALWAYS writes the three
        keys range_ref_distance_m, range_ref_level_dbfs AND
        range_spreading_db together (see calibrate.py cmd_range). A file
        that has the first two but not the third was not produced by the
        calibration tool. Combined with suspiciously round values, that
        means the metres on screen rest on numbers nobody measured.

        This matters more than it looks. The whole point of ranging.py was
        to stop the system inventing distances — the previous version's
        `6.0 / peak_volume` produced confident, meaningless metres. A
        hand-edited calibration file reintroduces exactly that failure
        while looking calibrated. The operator must be told.

        ⚠️ The RAW file is inspected, not the loaded config: calibration
        .load() merges DEFAULTS, which supply range_spreading_db=22.0, so
        the key is always present in the merged dict and the check would
        never fire against it.
        """
        import json

        import calibration as calib_module

        try:
            raw = json.loads(
                calib_module.CALIBRATION_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return

        ref_d = raw.get("range_ref_distance_m")
        ref_l = raw.get("range_ref_level_dbfs")
        if ref_d is None or ref_l is None:
            return
        has_spreading = "range_spreading_db" in raw

        looks_round = all(float(v) == round(float(v)) for v in (ref_d, ref_l))

        if not has_spreading and looks_round:
            log.warning(
                "range calibration looks HAND-WRITTEN, not measured: "
                "ref %.0f m @ %.0f dBFS with no range_spreading_db key, "
                "which `calibrate.py range` always writes.",
                float(ref_d), float(ref_l))
            log.warning(
                "  -> every distance in metres shown by the acoustic "
                "sensor is therefore ARBITRARY. Run "
                "`python calibrate.py range` with a real drone at a known "
                "distance before trusting any of them.")

    def _make_alerter(self):
        """Telegram, only if a token is supplied out-of-band."""
        import os
        token = (self.config.integrations.telegram_token
                 or os.environ.get(self.config.integrations.telegram_token_env))
        if not token:
            log.warning("telegram enabled but no token supplied (set %s) — "
                        "skipping", self.config.integrations.telegram_token_env)
            return None
        try:
            from telegram_alert import TelegramAlerter
            alerter = TelegramAlerter(token)
            alerter.discover_chats()
            if alerter.ready:
                log.info("telegram: %d chat(s)", len(alerter.chat_ids))
                return alerter
            log.info("telegram: no chats registered yet")
        except Exception as exc:
            log.warning("telegram unavailable: %s", exc)
        return None

    # ── Main loop ──────────────────────────────────────────────

    def _loop(self) -> None:
        import features as _features

        log.info("acoustic loop running (%.1f Hz decision rate, %.0f ms "
                 "processing budget per block)",
                 1.0 / self._block_seconds, self._block_seconds * 1000.0)
        consecutive_errors = 0
        # The audio thread is the only place that can measure how long the
        # microphone takes to hand over a block. That wait IS latency even
        # though it is not our code: a drone that becomes audible just after
        # a read returns is not looked at until the next one completes.
        BUDGET.describe_config(
            self._block_seconds,
            self.config.acoustic.detector_tuning().confirm_blocks,
            float(_features.WINDOW_SEC))
        last_return = time.monotonic()

        with self._stream:
            while not self._stop.is_set():
                try:
                    block, overflowed = self._stream.read(self._block_samples)
                except Exception as exc:
                    consecutive_errors += 1
                    self.events.rate("audio-read-error", logging.ERROR,
                                     "audio read failed: %s", exc)
                    self._set_health(SubsystemState.DEGRADED,
                                     "audio read errors", str(exc))
                    if consecutive_errors >= 20:
                        raise
                    time.sleep(0.05)
                    continue
                consecutive_errors = 0

                # ⚠️ The clock starts AFTER the read returns.
                # stream.read() blocks until a full 0.5 s block exists, so
                # timing from before it would always measure ~500 ms of
                # waiting plus the work, and could never be "within budget".
                # What matters is whether the WORK fits inside the 0.5 s
                # period — if it does not, the device buffer overruns and
                # audio is permanently lost.
                t0 = time.monotonic()
                BUDGET.record("audio_wait", (t0 - last_return) * 1000.0)
                last_return = t0

                if overflowed:
                    # Data already read is still processed — the original
                    # radar.py comment explains why skipping it is worse.
                    self._overflow_count += 1
                    self.events.rate("audio-overflow", logging.WARNING,
                                     "audio buffer overflow (x%d) — the host "
                                     "is not keeping up with the microphone",
                                     self._overflow_count, interval=5.0)

                block_np = np.asarray(block)

                try:
                    status = self._engine.process_block(block_np)
                except Exception as exc:
                    # An inference failure must not kill the thread; skip
                    # this block and keep the stream draining.
                    self.events.rate("acoustic-infer-error", logging.ERROR,
                                     "acoustic inference failed: %s", exc)
                    self._set_health(SubsystemState.DEGRADED,
                                     "inference errors", str(exc))
                    continue

                self._seq += 1
                obs = self._to_observation(status)
                self.latest.publish(obs)

                if self.health.get().state is not SubsystemState.ONLINE:
                    self._set_health(SubsystemState.ONLINE, "recovered")

                self._dispatch_side_channels(status, block_np)
                self._log_events(obs)

                elapsed = time.monotonic() - t0
                BUDGET.record("block_total", elapsed * 1000.0)
                self._block_times.append(elapsed)
                if len(self._block_times) > 200:
                    del self._block_times[:100]

                # Processing must stay well inside the 0.5 s block period,
                # otherwise the device buffer overruns and audio is lost.
                if elapsed > 0.9 * self._block_seconds:
                    self.events.rate(
                        "audio-slow", logging.WARNING,
                        "acoustic processing took %.0f ms of its %.0f ms "
                        "budget — the host is close to dropping audio",
                        elapsed * 1000.0, self._block_seconds * 1000.0,
                        interval=10.0)

    # ── Translation ────────────────────────────────────────────

    def _to_observation(self, status) -> AcousticObservation:
        """
        RadarStatus (mutable, reused in place by the engine) → immutable
        snapshot.

        ⚠️ This copy is not optional. RadarEngine mutates a single
        RadarStatus instance every block; publishing a reference to it would
        let the UI read a record that changes underneath it mid-render.
        (radar_gui.py did copy the fields for the same reason — but silently
        dropped `overflow`, which is carried here.)
        """
        r = status.range
        return AcousticObservation(
            engine_state=status.state,
            p_drone=float(status.p_drone),
            p_smoothed=float(status.p_smoothed),
            threshold=float(status.threshold),
            hold_threshold=float(status.hold_threshold),
            confirmations=int(status.confirmations),
            confirm_needed=int(status.confirm_needed),
            misses=int(status.misses),
            miss_tolerance=int(status.miss_tolerance),
            gated=bool(status.gated),
            bearing_deg=(None if status.angle_deg is None
                         else float(status.angle_deg)),
            bearing_confidence=float(status.angle_confidence),
            bearing_source=str(status.angle_source),
            bearing_ambiguous=bool(status.angle_ambiguous),
            bearing_calibrated=bool(getattr(status, "angle_calibrated", False)),
            bearing_reason=str(getattr(status, "angle_reason", "")),
            distance_m=None if r.distance_m is None else float(r.distance_m),
            distance_lo_m=None if r.lo_m is None else float(r.lo_m),
            distance_hi_m=None if r.hi_m is None else float(r.hi_m),
            distance_reason=str(r.reason),
            level_dbfs=float(r.level_dbfs),
            noise_dbfs=float(r.noise_dbfs),
            excess_db=float(r.excess_db),
            timestamp=now(),
            seq=self._seq,
            overflow=bool(status.overflow))

    # ── Side channels ──────────────────────────────────────────

    def _dispatch_side_channels(self, status, block_np: np.ndarray) -> None:
        """
        UART / WAV capture / Telegram.

        Each is wrapped so a failure in an optional extra can never stop
        drone detection, which is the part that matters.
        """
        if self._uart is not None:
            try:
                self._uart.send(status)
            except Exception as exc:
                self.events.rate("uart-error", logging.WARNING,
                                 "UART send failed: %s", exc)

        if self._audio_logger is not None:
            try:
                self._audio_logger.feed(block_np)
                self._audio_logger.on_status(
                    status.state, status.angle_deg, status.range.format())
            except Exception as exc:
                self.events.rate("audiolog-error", logging.WARNING,
                                 "audio logging failed: %s", exc)

        if self._alerter is not None and self._alerter.ready:
            try:
                self._alerter.on_status(
                    status.state, status.angle_deg, status.range.format(),
                    status.p_smoothed)
            except Exception as exc:
                self.events.rate("telegram-error", logging.WARNING,
                                 "telegram alert failed: %s", exc)

    # ── Event logging ──────────────────────────────────────────

    def _log_events(self, obs: AcousticObservation) -> None:
        """Narrative events only — never one line per block."""
        if obs.detected:
            self.events.on_change(
                "acoustic-bearing", obs.bearing_deg,
                "acoustic bearing %.0f deg (source=%s conf=%.2f)",
                obs.bearing_deg or 0.0, obs.bearing_source,
                obs.bearing_confidence,
                min_delta=self.config.logging.bearing_change_deg)
            self.events.on_change(
                "acoustic-distance", obs.distance_m,
                "acoustic distance ~%.0f m (%s)", obs.distance_m or 0.0,
                obs.distance_text(),
                min_frac=self.config.logging.distance_change_frac)
        else:
            # Forget the last values so the next contact logs afresh
            # instead of being compared against a previous target's.
            self.events.forget("acoustic-bearing", "acoustic-distance")

        self.events.rate(
            "acoustic-heartbeat", logging.DEBUG,
            "acoustic: state=%s p=%.2f level=%.0f dBFS noise=%.0f dBFS",
            obs.engine_state, obs.p_smoothed, obs.level_dbfs, obs.noise_dbfs,
            interval=5.0)

    # ── Teardown ───────────────────────────────────────────────

    def _teardown(self) -> None:
        for name, closer in (
                ("uart", lambda: self._uart and self._uart.close()),
                ("engine", lambda: self._engine and self._engine.close())):
            try:
                closer()
            except Exception as exc:
                log.debug("error closing %s: %s", name, exc)

        if self._overflow_count:
            log.warning("audio buffer overflows this session: %d",
                        self._overflow_count)
        if self._block_times:
            avg = sum(self._block_times) / len(self._block_times)
            log.info("acoustic block processing: mean %.0f ms, max %.0f ms "
                     "(budget 500 ms)", avg * 1000.0,
                     max(self._block_times) * 1000.0)
        self._set_health(SubsystemState.OFFLINE, "stopped")
        log.info("acoustic worker stopped")

    # ── Diagnostics ────────────────────────────────────────────

    @property
    def mean_block_ms(self) -> Optional[float]:
        if not self._block_times:
            return None
        return 1000.0 * sum(self._block_times) / len(self._block_times)

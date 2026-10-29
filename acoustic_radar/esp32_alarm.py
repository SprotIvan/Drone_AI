#!/usr/bin/env python3
"""
esp32_alarm.py — a speaker/buzzer on an ESP32 plugged into the Pi by USB.

The Pi decides, the ESP32 only beeps. Public surface used by main.py, the
same shape as respeaker_led.RespeakerLed:

    start()      spawn the worker (never raises, never blocks on hardware)
    submit(on)   publish the wanted state — O(1), UI-thread safe
    status       AlarmStatus
    detail       one-line human explanation of `status`
    stop()       send "off", close the port

═══════════════════════════════════════════════════════════════════
WHEN IT BEEPS
═══════════════════════════════════════════════════════════════════

alarm_for_target() maps the ONE fused target (SensorFusion.update) onto
on/off. Where the microphone holds the drone, it beeps exactly while
respeaker_led.frame_for_target() shows the ring red, so the speaker and the
ring cannot disagree:

    ON   VISUAL_TRACKING                camera confirmed it (also visual-only)
         ACOUSTIC_TRACKING              } while the microphone's own alarm
         VISUAL_ACQUISITION             } is held (radar ALARM or
         VISUAL_LOST_ACOUSTIC_TRACKING  } ALARM_COASTING) = the ring is red
    OFF  SEARCHING, TARGET_LOST         no sensor has the drone
         ACOUSTIC_DETECTED              unconfirmed contact — ON only with
                                        alarm.beep_on_unconfirmed_acoustic,
                                        and only while the detector has it

⚠️ WHY THE MICROPHONE SIDE DOES NOT FOLLOW THE FUSED STATE ALONE. The
fused state outlives the microphone: after radar.Detector releases the
alarm at the end of its coasting window (acoustic.coast_seconds), the
fusion layer holds the target for fusion.target_lost_after_s more (5 s)
before TARGET_LOST. The speaker used to follow that, so it kept beeping
~5 s after the ring had gone blue — on top of the coasting it had already
waited through. The off-delay after the drone falls silent is now the 2 s
analysis window plus acoustic.coast_seconds, the same as the ring's.

"The drone disappeared" is still the station's own definition, with its own
timeouts (visual.lost_after_s, acoustic.coast_seconds) — not a second,
competing one invented here.

═══════════════════════════════════════════════════════════════════
PROTOCOL (text lines, 115200 8N1, '\\n'; see esp32_alarm/esp32_alarm.ino)
═══════════════════════════════════════════════════════════════════

    Pi -> ESP32  "PING"            ESP32 -> Pi  "PONG ALARM v1 pin=14 hz=2700"
    Pi -> ESP32  "A1" / "A0"       ESP32 -> Pi  "OK A1" / "OK A0"
    Pi -> ESP32  "F2000"           ESP32 -> Pi  "OK F2000"  (passive-buzzer tone)
    ESP32 at boot                  -> "READY ALARM v1 pin=14 hz=2700"

The Pi repeats the current A1/A0 every alarm.heartbeat_s. The firmware goes
silent by itself after 2 s without a valid command, so a crashed or stopped
station cannot leave the speaker beeping. A port is used only after the
firmware has answered PING: alarm commands are never written into a serial
device that did not identify itself.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from enum import Enum
from typing import List, Optional

log = logging.getLogger("station.alarm")

#: What the firmware must answer to PING (or announce at boot).
FIRMWARE_ID = "ALARM v1"


def alarm_for_target(target, cfg) -> bool:
    """THE single mapping from the fused state to the speaker. Pure."""
    from respeaker_led import frame_for_target
    from sensor_fusion import SystemState
    from target_state import Freshness

    state = getattr(target, "state", None)
    if state is SystemState.VISUAL_TRACKING:
        return True
    if state in (SystemState.ACOUSTIC_TRACKING,
                 SystemState.VISUAL_ACQUISITION,
                 SystemState.VISUAL_LOST_ACOUSTIC_TRACKING):
        # The microphone holds these, so the speaker lets go when the ring
        # does, not target_lost_after_s later (see WHEN IT BEEPS).
        return frame_for_target(target).mode.is_alarm
    if state is SystemState.ACOUSTIC_DETECTED:
        acoustic = target.acoustic
        return (bool(getattr(cfg, "beep_on_unconfirmed_acoustic", False))
                and acoustic is not None and acoustic.detected
                and target.acoustic_freshness is not Freshness.LOST
                and target.acoustic_health.ok)
    return False


class AlarmStatus(Enum):
    DISABLED = "DISABLED"            # switched off in the configuration
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"                # an ESP32 answered and is being driven
    UNAVAILABLE = "UNAVAILABLE"      # no ESP32 found / it stopped answering

    @property
    def ok(self) -> bool:
        return self is AlarmStatus.ACTIVE


def candidate_ports(cfg) -> List[str]:
    """The configured port, or every USB serial device (by-id names first,
    de-duplicated by the device they point to)."""
    if cfg.port:
        return [str(cfg.port)]
    seen, out = set(), []
    for p in (sorted(glob.glob("/dev/serial/by-id/*"))
              + sorted(glob.glob("/dev/ttyUSB*"))
              + sorted(glob.glob("/dev/ttyACM*"))):
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            out.append(p)
    return out


class Esp32Alarm:

    def __init__(self, cfg):
        self.cfg = cfg
        self.status = (AlarmStatus.DISABLED if not cfg.enabled
                       else AlarmStatus.STARTING)
        self.detail = "" if cfg.enabled else "disabled in configuration"
        self.port: Optional[str] = None
        self.commands_sent = 0
        self.on_count = 0                 # OFF->ON transitions sent

        self._want = False
        self._last_submit = 0.0
        self._freq_req: Optional[int] = cfg.tone_hz
        self._freq_sent: Optional[int] = None
        self._ser = None
        self._sent: Optional[bool] = None
        self._last_send = 0.0
        self._last_reply = 0.0
        self._rx = b""
        self._retry_after = 0.0

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        if not self.cfg.enabled:
            log.info("ESP32 alarm disabled in configuration")
            return False
        self._thread = threading.Thread(target=self._run, name="alarm",
                                        daemon=True)
        self._thread.start()
        return True

    def submit(self, on: bool) -> None:
        """Called from the UI thread on every fused update."""
        self._want = bool(on)
        self._last_submit = time.monotonic()
        self._wake.set()

    def set_frequency(self, hz: int) -> None:
        """Tone of the passive buzzer, 50-10000 Hz. Takes effect at once and
        is re-sent after every reconnect or board reboot."""
        hz = int(hz)
        if not 50 <= hz <= 10000:
            raise ValueError(f"tone must be 50-10000 Hz, got {hz}")
        self._freq_req = hz
        self._wake.set()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("alarm thread did not stop within %.1fs", timeout)

    def status_text(self) -> str:
        return f"ESP32 alarm: {self.status.value}" + (
            f" ({self.detail})" if self.detail else "")

    # ── Worker ─────────────────────────────────────────────────

    def _set_status(self, status: AlarmStatus, detail: str) -> None:
        if status is not self.status or detail != self.detail:
            (log.info if status.ok else log.warning)(
                "ESP32 alarm: %s — %s", status.value, detail)
        self.status, self.detail = status, detail

    def _effective_want(self) -> bool:
        # Safety net, as LedConfig.watchdog_s: a UI loop that stopped
        # submitting must not leave the speaker on.
        if time.monotonic() - self._last_submit > self.cfg.watchdog_s:
            return False
        return self._want

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    if self._ser is None:
                        if time.monotonic() >= self._retry_after:
                            self._connect()
                    else:
                        self._tick()
                except Exception as exc:          # never fatal
                    self._drop(f"serial error: {exc}")
                self._wake.wait(0.05 if self._ser is not None else 0.5)
                self._wake.clear()
        finally:
            self._shutdown_port()

    def _connect(self) -> None:
        ports = candidate_ports(self.cfg)
        if not ports:
            self._set_status(AlarmStatus.UNAVAILABLE,
                             "no USB serial device (/dev/ttyUSB*, /dev/ttyACM*)"
                             " — is the ESP32 plugged in?")
            self._retry_after = time.monotonic() + self.cfg.retry_interval_s
            return
        tried = []
        for port in ports:
            ser, reply = self._open_and_handshake(port)
            if ser is not None:
                self._ser, self.port = ser, port
                self._sent = None                 # resend the state at once
                self._freq_sent = None            # ... and the tone
                self._last_reply = time.monotonic()
                self._rx = b""
                self._set_status(AlarmStatus.ACTIVE, f"{port}: {reply}")
                return
            tried.append(f"{port}: {reply}")
        self._set_status(AlarmStatus.UNAVAILABLE,
                         "no alarm firmware answered — " + "; ".join(tried))
        self._retry_after = time.monotonic() + self.cfg.retry_interval_s

    def _open_and_handshake(self, port: str):
        try:
            import serial as pyserial
        except ImportError as exc:
            return None, f"pyserial not installed ({exc})"
        try:
            ser = pyserial.Serial()
            ser.port = port
            ser.baudrate = int(self.cfg.baud)
            ser.timeout = 0
            ser.write_timeout = 0.5
            # Not asserted: on the usual ESP32 USB-serial boards DTR/RTS drive
            # the EN/BOOT auto-reset circuit. The kernel may still pulse them
            # while opening, so the handshake below allows for a reboot.
            ser.dtr = False
            ser.rts = False
            ser.open()
        except Exception as exc:
            return None, f"cannot open ({exc})"
        deadline = time.monotonic() + self.cfg.handshake_timeout_s
        buf = b""
        next_ping = 0.0
        try:
            while time.monotonic() < deadline and not self._stop.is_set():
                if time.monotonic() >= next_ping:
                    ser.write(b"PING\n")
                    next_ping = time.monotonic() + 0.3
                buf += ser.read(256)
                for line in buf.split(b"\n")[:-1]:
                    text = line.decode("ascii", "replace").strip()
                    if FIRMWARE_ID in text and text.split()[0] in ("PONG",
                                                                   "READY"):
                        return ser, text
                buf = buf.split(b"\n")[-1][-256:]
                time.sleep(0.02)
        except Exception as exc:
            ser.close()
            return None, f"handshake failed ({exc})"
        ser.close()
        return None, "no answer to PING"

    def _tick(self) -> None:
        now_t = time.monotonic()
        if self._freq_req is not None and self._freq_req != self._freq_sent:
            self._ser.write(f"F{self._freq_req}\n".encode("ascii"))
            self._freq_sent = self._freq_req
        want = self._effective_want()
        if want != self._sent or now_t - self._last_send >= self.cfg.heartbeat_s:
            self._ser.write(b"A1\n" if want else b"A0\n")
            if want and self._sent is not True:
                self.on_count += 1
            if want != self._sent:
                log.info("speaker %s", "ON" if want else "OFF")
            self._sent, self._last_send = want, now_t
            self.commands_sent += 1

        self._rx += self._ser.read(256)
        lines = self._rx.split(b"\n")
        self._rx = lines[-1][-256:]
        for line in lines[:-1]:
            text = line.decode("ascii", "replace").strip()
            if not text:
                continue
            self._last_reply = now_t
            if text.startswith("READY"):
                # The ESP32 rebooted (brown-out, reset): it is silent now and
                # back on its default tone, so both must be sent again.
                log.warning("ESP32 alarm rebooted (%s) — resending state", text)
                self._sent = None
                self._freq_sent = None
            elif text.startswith("OK F"):
                log.info("buzzer tone %s Hz", text[4:])
            elif text.startswith("ERR"):
                log.warning("ESP32 rejected a command: %s", text)

        # Every command is acknowledged; a device that stopped answering is
        # dropped and searched for again (it may have been unplugged).
        if now_t - self._last_reply > max(3.0, 4 * self.cfg.heartbeat_s):
            self._drop("no reply from the ESP32")

    def _drop(self, why: str) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._sent = None
        self._set_status(AlarmStatus.UNAVAILABLE, why)
        self._retry_after = time.monotonic() + self.cfg.retry_interval_s

    def _shutdown_port(self) -> None:
        if self._ser is None:
            return
        try:
            self._ser.write(b"A0\n")
            self._ser.flush()
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass
        self._ser = None


if __name__ == "__main__":
    # Bench test without the station (stop the station first — the port can
    # be open in one program only):
    #     python esp32_alarm.py [PORT]
    # then type a frequency in Hz and the buzzer beeps at it; 0 = silence,
    # Enter on an empty line = exit. The loudest frequency can then go into
    # fusion_config.json as "alarm": {"tone_hz": ...}.
    import sys
    from fusion_config import AlarmConfig

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = Esp32Alarm(AlarmConfig(port=sys.argv[1] if len(sys.argv) > 1 else None,
                               retry_interval_s=2.0))
    a.start()
    t0 = time.monotonic()
    while a.status is not AlarmStatus.ACTIVE and time.monotonic() - t0 < 10:
        a.submit(False)
        time.sleep(0.2)
    print(a.status_text())
    if a.status is AlarmStatus.ACTIVE:
        state = {"want": False, "run": True}

        def keep_alive():
            # The firmware goes silent after 2 s without A1/A0.
            while state["run"]:
                a.submit(state["want"])
                time.sleep(0.1)

        threading.Thread(target=keep_alive, daemon=True).start()
        try:
            while True:
                s = input("частота, Гц (50-10000), 0 = тиша, Enter = вихід: ").strip()
                if not s:
                    break
                try:
                    hz = int(s)
                except ValueError:
                    print("  потрібне ціле число")
                    continue
                if hz == 0:
                    state["want"] = False
                    print("  тиша")
                    continue
                try:
                    a.set_frequency(hz)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue
                state["want"] = True
                print(f"  пищить на {hz} Гц (200 мс звук / 200 мс пауза)")
        except (KeyboardInterrupt, EOFError):
            pass
        state["run"] = False
    a.stop()

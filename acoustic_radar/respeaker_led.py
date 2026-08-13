#!/usr/bin/env python3
"""
respeaker_led.py — The ReSpeaker XVF3800 LED ring as a target indicator.

    CLEAR / SEARCHING   dim blue ring        "listening, nothing found"
    ALARM               red sector at the acoustic bearing
    ALARM_COASTING      the SAME red sector, frozen at the last valid
                        bearing, held for the detector's coasting window

═══════════════════════════════════════════════════════════════════
⚠️  WHAT THIS MODULE REFUSES TO GUESS
═══════════════════════════════════════════════════════════════════

There is no LED control code anywhere in this repository, and no USB
protocol implementation of any kind. The ONLY verified path to the array is
the one doa.py already uses:

    subprocess: python xvf_host.py <COMMAND_NAME> [args...]

`xvf_host.py` is Seeed's own utility and ships with the array; it is not
part of this project. Its command set is defined by the FIRMWARE on the
device, not by anything here. Therefore:

  • No command name is hard-coded as a fact.
  • Before any command is sent, the installed xvf_host.py is asked for its
    OWN list of supported commands (`discover_commands`).
  • A command is used only if that list contains it. If the list cannot be
    obtained, or contains nothing recognisable, the controller reports
    UNAVAILABLE, logs exactly why, and never writes to the device.
  • `fusion_config.led.cmd_*` lets an operator name the commands
    explicitly, which always wins over discovery.

Run `python respeaker_led.py --probe` on the Raspberry Pi to see what your
array actually supports.

═══════════════════════════════════════════════════════════════════
WHY THIS IS A THREAD WITH A ONE-SLOT MAILBOX
═══════════════════════════════════════════════════════════════════

One LED write is a Python interpreter start-up — doa.py measured that at
200-500 ms. Doing it on the UI thread would stall the display, and through
the GIL it would stall the camera and audio threads with it. So the UI
thread only calls `submit()`, which publishes a small immutable frame into
a LatestValue and returns; a dedicated worker performs the slow writes.

The worker is also the only place that decides whether a write is needed at
all. It keeps a shadow copy of the ring and writes ONLY the LEDs whose
colour actually changed, so a sector that has not moved costs nothing and a
sector that moved by one position costs two writes rather than a full
repaint.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from doa import find_xvf_host, unapply_orientation
from target_state import LatestValue

log = logging.getLogger("station.led")

Colour = Tuple[int, int, int]


# ═══════════════════════════════════════════════════════════════
#  What the ring is being asked to show
# ═══════════════════════════════════════════════════════════════

class LedMode(Enum):
    OFF = "OFF"
    SEARCHING = "SEARCHING"
    ALARM = "ALARM"
    COASTING = "ALARM_COASTING"

    @property
    def is_alarm(self) -> bool:
        return self in (LedMode.ALARM, LedMode.COASTING)


@dataclass(frozen=True)
class LedFrame:
    """
    One requested ring appearance.

    `bearing_deg` is in the INSTALLATION frame — the same 0-360 degrees the
    radar and the HUD show, i.e. after radar_calibration.json's
    doa_offset_deg has been applied. Converting it back into the array's own
    frame is this module's job (see `bearing_to_led_index`), because the
    ring is bolted to the array, not to the compass.
    """
    mode: LedMode = LedMode.SEARCHING
    bearing_deg: Optional[float] = None


class LedStatus(Enum):
    DISABLED = "DISABLED"            # switched off in the configuration
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"                # writing to real hardware
    UNAVAILABLE = "UNAVAILABLE"      # cannot drive the ring; see `detail`

    @property
    def ok(self) -> bool:
        return self is LedStatus.ACTIVE

    def as_subsystem_state(self):
        """
        Map onto the SubsystemState the HUD already knows how to colour.

        Kept here rather than in hud.py so the display never has to learn
        what an LED status means — it only ever draws lamps.
        """
        from target_state import SubsystemState
        return {
            LedStatus.ACTIVE: SubsystemState.ONLINE,
            LedStatus.STARTING: SubsystemState.STARTING,
            LedStatus.DISABLED: SubsystemState.DISABLED,
            LedStatus.UNAVAILABLE: SubsystemState.OFFLINE,
        }[self]


# ═══════════════════════════════════════════════════════════════
#  Mapping the fused target onto a ring appearance
# ═══════════════════════════════════════════════════════════════

def frame_for_target(target) -> LedFrame:
    """
    THE single mapping from the authoritative fused state to the ring.

    This function is the reason the LED, the radar widget and the camera cue
    can never disagree: all three are rendered from the same FusedTarget
    produced by one SensorFusion.update() call, and this is the only place
    that turns it into LED terms. Nothing else in the project is allowed to
    decide what the ring shows.

    Note what is NOT consulted here: no UI repaint counter, no separate
    confidence threshold, no independent timer. A frame is a pure function
    of the fused state, so a UI refresh cannot produce a different colour
    than the one before it while the state is unchanged — which is what
    makes the RED/BLUE/RED flicker in requirement 5 structurally
    impossible rather than merely unlikely.
    """
    from target_state import Freshness

    acoustic = getattr(target, "acoustic", None)
    if acoustic is None:
        return LedFrame(LedMode.SEARCHING, None)

    # A reading old enough to be LOST is not evidence of anything. STALE is
    # tolerated: at 2 Hz a reading is routinely up to 1.5 s old between
    # blocks, and dropping to blue in that gap is precisely the flicker
    # this design forbids.
    if getattr(target, "acoustic_freshness", None) is Freshness.LOST:
        return LedFrame(LedMode.SEARCHING, None)

    if not getattr(target, "acoustic_health", None) or \
            not target.acoustic_health.ok:
        return LedFrame(LedMode.SEARCHING, None)

    if acoustic.coasting:
        return LedFrame(LedMode.COASTING, acoustic.bearing_deg)
    if acoustic.confirmed:
        return LedFrame(LedMode.ALARM, acoustic.bearing_deg)
    return LedFrame(LedMode.SEARCHING, None)


# ═══════════════════════════════════════════════════════════════
#  Ring geometry
# ═══════════════════════════════════════════════════════════════

def bearing_to_led_index(bearing_deg: float, led_count: int,
                         doa_offset_deg: float = 0.0,
                         doa_invert: bool = False,
                         led_zero_offset_deg: float = 0.0,
                         clockwise: bool = True) -> int:
    """
    Which LED sits closest to an installation-frame bearing.

    Two rotations, in this order, and both of them matter:

      1. installation frame -> array frame, undoing radar_calibration.json's
         doa_offset_deg / doa_invert. Skipping this lights an LED that is
         wrong by exactly the mount offset — a mistake that looks perfectly
         plausible on screen and is only visible with the hardware in hand.

      2. array frame -> ring index, using where LED 0 physically sits and
         which way the indices run.
    """
    if led_count <= 0:
        raise ValueError("led_count must be positive")

    array_deg = unapply_orientation(float(bearing_deg), doa_offset_deg,
                                    doa_invert)
    ring_deg = (array_deg - float(led_zero_offset_deg)) % 360.0
    index = int(round(ring_deg / 360.0 * led_count)) % led_count
    if not clockwise:
        index = (led_count - index) % led_count
    return index


def sector_indices(centre: int, span: int, led_count: int) -> List[int]:
    """LED indices of a sector of `span` LEDs centred on `centre`."""
    span = max(1, min(int(span), int(led_count)))
    half = (span - 1) // 2
    start = centre - half
    return [(start + i) % led_count for i in range(span)]


# ═══════════════════════════════════════════════════════════════
#  Talking to xvf_host.py
# ═══════════════════════════════════════════════════════════════

#: Tokens that look like an xvf_host control command in its help/command-map
#: output: upper-case, underscore-separated, at least two segments.
_COMMAND_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

#: Flags to try when asking xvf_host.py to list its commands. Different
#: releases of the Seeed utility expose different ones, so each is tried and
#: the first that yields a plausible command list wins. Nothing is assumed
#: about which one exists.
_LIST_FLAGS: Sequence[Sequence[str]] = (
    ("--list",), ("-l",), ("--help",), ("-h",), (),
)

#: SEARCH PATTERNS, not command names. Each is matched against the command
#: list the FIRMWARE itself reported; a pattern that matches nothing is
#: simply not used, and a pattern that matches more than one command is
#: treated as ambiguous and refused. No command is ever sent because it
#: appears in this table — only because the device declared it.
#
# LED_EFFECT is the automatic-animation selector on the XVF3800 firmware
# actually observed in the field (which reports LED_BRIGHTNESS, LED_COLOR,
# LED_DOA_COLOR, LED_EFFECT, LED_GAMMIFY, LED_RING_COLOR, LED_SPEED and NO
# LED_AUTO_MODE at all). It is the register that has to be quietened for
# host control; `led.auto_mode_off_value` decides what is written to it and
# the exact call is logged, because an effect index is a cosmetic setting
# whose enumeration this project cannot read.
_AUTO_MODE_PATTERNS = ("LED_AUTO_MODE", "AUTO_LED_MODE", "LED_AUTO",
                       "LED_EFFECT")
_RING_COLOUR_PATTERNS = ("LED_RING_COLOUR", "LED_RING_COLOR", "LED_RING",
                         "LED_COLOUR_ALL", "LED_COLOR_ALL", "LED_ALL",
                         "LED_COLOUR", "LED_COLOR")
_LED_COLOUR_PATTERNS = ("LED_INDIVIDUAL", "LED_SET_ONE", "LED_PIXEL",
                        "LED_SINGLE", "LED_INDEX")

#: The firmware states its own arity when it rejects a call, e.g.
#: "Error: LED_RING_COLOR value count is 12, but 3 values provided".
#: Parsing it turns a dead indicator into a self-correcting one.
_ARITY_ERROR = re.compile(r"value count is\s+(\d+)", re.IGNORECASE)


@dataclass
class XvfCommands:
    """Command names this array is known — not assumed — to support."""
    available: Tuple[str, ...] = ()
    auto_mode: Optional[str] = None
    ring_colour: Optional[str] = None
    led_colour: Optional[str] = None
    #: How many values the ring command wants, MEASURED by reading it back
    #: from the device (or learned from the device's own error message).
    #: This is what distinguishes "one colour for the whole ring" (3) from
    #: "one packed colour per LED" (led_count) — a difference that cannot
    #: be inferred from the command's NAME, and which the observed
    #: LED_RING_COLOR gets exactly backwards from what the name suggests.
    ring_arity: Optional[int] = None
    detail: str = ""

    def ring_is_per_pixel(self, led_count: Optional[int]) -> bool:
        return (self.ring_colour is not None and led_count is not None
                and self.ring_arity == int(led_count))

    @property
    def can_drive_ring(self) -> bool:
        return self.ring_colour is not None or self.led_colour is not None

    def can_drive_sector(self, led_count: Optional[int] = None) -> bool:
        """
        A sector needs per-LED addressing.

        Two ways to get it: a command that sets one LED at a time, or a ring
        command that takes one value per LED — which is far better, since it
        paints the whole ring in ONE subprocess call instead of twelve.
        """
        return (self.led_colour is not None
                or self.ring_is_per_pixel(led_count))

    def led_related(self) -> Tuple[str, ...]:
        return tuple(c for c in self.available if "LED" in c.upper())


def discover_commands(script_path: Path,
                      timeout: float = 5.0) -> Tuple[str, ...]:
    """
    Ask the installed xvf_host.py which control commands it supports.

    Returns the command names found, or an empty tuple. Never raises: an
    array that cannot be interrogated must degrade to "LED unavailable",
    not take the detection pipeline down with it.
    """
    for flag in _LIST_FLAGS:
        try:
            out = subprocess.run(
                [sys.executable, str(script_path), *flag],
                capture_output=True, text=True, timeout=timeout,
                # xvf_host.py reads its command maps relative to its own
                # directory — same requirement doa.py documents.
                cwd=str(script_path.parent))
        except (subprocess.TimeoutExpired, OSError):
            continue
        text = (out.stdout or "") + "\n" + (out.stderr or "")
        found = sorted(set(_COMMAND_TOKEN.findall(text)))
        # A real command list has many entries; one or two tokens is much
        # more likely to be a usage string or a traceback.
        if len(found) >= 5:
            return tuple(found)
    return ()


def read_command_values(script_path: Path, command: str,
                        timeout: float = 5.0) -> List[float]:
    """
    Read a control command's CURRENT values back from the device.

    This is how the ring command's arity is established without guessing:
    `xvf_host.py LED_RING_COLOR` with no values is a read, and counting the
    numbers it prints says how many the write form expects. The observed
    firmware wants 12 — one packed colour per LED — even though the name
    reads like a single colour for the whole ring.
    """
    try:
        out = subprocess.run(
            [sys.executable, str(script_path), command],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(script_path.parent))
    except (subprocess.TimeoutExpired, OSError):
        return []
    text = (out.stdout or "") + "\n" + (out.stderr or "")
    # Drop the echoed command name so its digits are not read as values.
    text = re.sub(re.escape(command), " ", text, flags=re.IGNORECASE)
    return [float(m) for m in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def _match_unique(patterns: Sequence[str],
                  available: Sequence[str]) -> Tuple[Optional[str], str]:
    """
    Find the one command matching a pattern group.

    Exact matches are preferred. Failing that, a prefix match is accepted
    ONLY when it is unique — an ambiguous match is refused rather than
    resolved by picking the first, because sending the wrong control
    command to a DSP is not a recoverable mistake made visible by a log
    line.
    """
    upper = {c.upper(): c for c in available}
    for pattern in patterns:
        if pattern in upper:
            return upper[pattern], f"exact match {pattern}"
    for pattern in patterns:
        hits = [orig for up, orig in upper.items() if up.startswith(pattern)]
        if len(hits) == 1:
            return hits[0], f"unique prefix match {pattern} -> {hits[0]}"
        if len(hits) > 1:
            return None, (f"ambiguous: {pattern} matches {', '.join(hits)} "
                          f"— name it explicitly in fusion_config.json")
    return None, "no matching command reported by the device"


def resolve_commands(led_cfg, script_path: Optional[Path],
                     timeout: float = 5.0) -> XvfCommands:
    """
    Work out which commands to use: explicit configuration first, then
    discovery against the device's own command list.
    """
    configured = XvfCommands(
        auto_mode=led_cfg.cmd_auto_mode,
        ring_colour=led_cfg.cmd_ring_colour,
        led_colour=led_cfg.cmd_led_colour)

    if script_path is None:
        return XvfCommands(detail="xvf_host.py not found (set XVF_HOST_PATH)")

    available = discover_commands(script_path, timeout)
    if not available:
        if configured.can_drive_ring:
            # The operator named the commands. Trust them: they had the
            # hardware in front of them, which this code does not.
            configured.detail = ("command list unavailable; using the names "
                                 "from fusion_config.json")
            return configured
        return XvfCommands(
            detail=f"{script_path.name} did not report a command list, and "
                   f"no led.cmd_* names are configured")

    resolved = XvfCommands(available=available)
    notes: List[str] = []

    for attr, patterns, explicit in (
            ("auto_mode", _AUTO_MODE_PATTERNS, led_cfg.cmd_auto_mode),
            ("ring_colour", _RING_COLOUR_PATTERNS, led_cfg.cmd_ring_colour),
            ("led_colour", _LED_COLOUR_PATTERNS, led_cfg.cmd_led_colour)):
        if explicit:
            if explicit.upper() in {c.upper() for c in available}:
                setattr(resolved, attr, explicit)
                notes.append(f"{attr}={explicit} (configured)")
            else:
                notes.append(f"{attr}={explicit} CONFIGURED BUT NOT SUPPORTED "
                             f"by this firmware — ignored")
            continue
        name, why = _match_unique(patterns, available)
        if name:
            setattr(resolved, attr, name)
            notes.append(f"{attr}={name} ({why})")
        else:
            notes.append(f"{attr}: {why}")

    resolved.detail = "; ".join(notes)
    return resolved


# ═══════════════════════════════════════════════════════════════
#  The controller
# ═══════════════════════════════════════════════════════════════

class RespeakerLed:
    """
    Owns the LED ring. Public surface used by main.py:

        start()    spawn the worker (never raises, never blocks on hardware)
        submit(f)  publish the desired appearance — O(1), UI-thread safe
        status     LedStatus
        detail     one-line human explanation of `status`
        stop()     restore a safe state and release the device
    """

    #: Hardware writes performed per worker cycle. A full repaint of a
    #: 12-LED ring is 12 subprocess spawns; doing them in one burst would
    #: hold the ring in a half-painted state for seconds. Spreading them
    #: keeps every individual cycle short and the ring converging.
    MAX_WRITES_PER_CYCLE = 4

    #: Backoff after the failure limit is hit, in seconds. Long enough not
    #: to spam a detached device, short enough that replugging recovers
    #: without restarting the station.
    RETRY_INTERVAL_S = 30.0

    def __init__(self, led_cfg, calibration_cfg: Optional[dict] = None,
                 script_path: Optional[Path] = None):
        self.cfg = led_cfg
        calibration_cfg = calibration_cfg or {}
        self._doa_offset = float(calibration_cfg.get("doa_offset_deg", 0.0))
        self._doa_invert = bool(calibration_cfg.get("doa_invert", False))
        self._script_path = script_path

        self.status = (LedStatus.DISABLED if not led_cfg.enabled
                       else LedStatus.STARTING)
        self.detail = "" if led_cfg.enabled else "disabled in configuration"

        self._requested: LatestValue[LedFrame] = LatestValue(LedFrame())
        self._last_submit = 0.0
        self._commands = XvfCommands()

        # Shadow copy of what each LED is believed to be showing, so only
        # genuine differences are written. None = never written.
        self._shadow: Dict[int, Optional[Colour]] = {}
        self._ring_shadow: Optional[Colour] = None
        self._auto_mode_disabled = False

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failures = 0
        self._writes = 0
        self._last_write_time = 0.0
        self._retry_after = 0.0
        self._arity_just_learned = False

    # ── Introspection ──────────────────────────────────────────

    @property
    def led_count(self) -> Optional[int]:
        return self.cfg.led_count

    def status_text(self) -> str:
        return f"LED: {self.status.value}" + (f" ({self.detail})"
                                              if self.detail else "")

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start the worker. Returns True if the ring will actually be driven.

        Setup that touches the device happens ON the worker thread, not
        here: discovery runs xvf_host.py, which takes hundreds of
        milliseconds, and main.py must not spend that before the camera and
        microphone are running.
        """
        if not self.cfg.enabled:
            log.info("LED ring disabled in configuration")
            return False
        self._thread = threading.Thread(target=self._run, name="led",
                                        daemon=True)
        self._thread.start()
        return True

    def submit(self, frame: LedFrame) -> None:
        """
        Request an appearance. Called from the UI thread every fused update.

        Deliberately trivial: one publish into a one-slot mailbox. All
        comparison, quantisation and I/O happen on the worker.
        """
        self._requested.publish(frame)
        self._last_submit = time.monotonic()
        self._wake.set()

    def stop(self, timeout: float = 2.0) -> None:
        """Restore a safe ring state and release the device."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("LED thread did not stop within %.1fs — leaving "
                            "it to the daemon shutdown", timeout)

    # ── Worker ─────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as exc:                     # never fatal
            log.warning("LED subsystem could not start: %s", exc)
            self._set_status(LedStatus.UNAVAILABLE, str(exc))
            return

        try:
            while not self._stop.is_set():
                try:
                    self._tick()
                except Exception as exc:
                    # A failure inside one cycle must not end the thread:
                    # the array may come back, and even if it does not, the
                    # detection pipeline is unaffected either way.
                    self._note_failure(str(exc))
                self._wake.wait(self.cfg.min_write_interval_s)
                self._wake.clear()
        finally:
            self._shutdown_ring()

    def _setup(self) -> None:
        cfg = self.cfg
        # Failures that a replugged or freshly-powered array can fix are
        # retried; configuration mistakes are not, because retrying them
        # would only repeat the same log line forever.
        self._retry_after = 0.0

        if self._script_path is None:
            self._script_path = find_xvf_host()
        if self._script_path is None:
            self._set_status(
                LedStatus.UNAVAILABLE,
                "xvf_host.py not found — set XVF_HOST_PATH. This is the only "
                "control path to the array that this project has verified")
            self._retry_after = time.monotonic() + self.RETRY_INTERVAL_S
            return

        self._commands = resolve_commands(cfg, self._script_path,
                                          cfg.command_timeout_s)
        log.info("LED command resolution: %s", self._commands.detail or "none")
        if self._commands.available:
            log.info("LED-related commands reported by the array: %s",
                     ", ".join(self._commands.led_related()) or "(none)")

        if not self._commands.can_drive_ring:
            self._set_status(
                LedStatus.UNAVAILABLE,
                "no usable LED command — run `python respeaker_led.py "
                "--probe` and set led.cmd_* in fusion_config.json")
            # An array that was busy or asleep during discovery may answer
            # later; a firmware that genuinely has no LED command will not,
            # and will just re-log the same line every 30 s.
            self._retry_after = time.monotonic() + self.RETRY_INTERVAL_S
            return

        if cfg.led_count is None:
            self._set_status(
                LedStatus.UNAVAILABLE,
                "led.led_count not configured — the number of LEDs on the "
                "ring is a hardware fact this project will not guess")
            return

        self._measure_ring_arity()

        if not self._commands.can_drive_sector(cfg.led_count):
            log.warning("LED ring supports whole-ring colour only — the red "
                        "TARGET SECTOR cannot be drawn, so the ring will "
                        "show the target STATE without a bearing")

        self._shadow = {i: None for i in range(int(cfg.led_count))}
        self._disable_auto_mode()
        # ⚠️ Only after the auto-mode write, and only if that write did not
        # already push the controller past its failure limit. Declaring
        # ACTIVE unconditionally here would overwrite a legitimate
        # UNAVAILABLE with a green lamp on a device that is not answering.
        if self.status is not LedStatus.UNAVAILABLE:
            self._set_status(
                LedStatus.ACTIVE,
                f"{cfg.led_count} LEDs via {self._script_path.name}")

    def _measure_ring_arity(self) -> None:
        """
        Ask the device how many values its ring command wants.

        ⚠️ THIS IS WHY THE RING STAYED DARK ON REAL HARDWARE. The command
        is called LED_RING_COLOR, which reads like "one colour for the whole
        ring", so it was sent three values (R, G, B). The firmware answered

            Error: LED_RING_COLOR value count is 12, but 3 values provided

        — it is a PER-PIXEL command taking one packed colour per LED. A
        command's arity is a property of the firmware and cannot be inferred
        from its name, so it is measured here rather than assumed: reading
        the command back returns its current values, and counting them
        settles the question. `_note_arity_from_error` recovers the same
        number from a rejection if the read is not available.
        """
        cmd = self._commands.ring_colour
        if cmd is None or self._script_path is None:
            return
        values = read_command_values(self._script_path, cmd,
                                     self.cfg.command_timeout_s)
        if values:
            self._commands.ring_arity = len(values)
            log.info("LED ring command %s takes %d value(s) — %s", cmd,
                     len(values),
                     "one per LED, the whole ring paints in a single call"
                     if self._commands.ring_is_per_pixel(self.cfg.led_count)
                     else "a single colour for the whole ring")
            if (self._commands.ring_arity not in (1, 3)
                    and not self._commands.ring_is_per_pixel(
                        self.cfg.led_count)):
                log.warning("...but led.led_count is %s, which does not match "
                            "the %d values %s expects. Set led_count to %d.",
                            self.cfg.led_count, len(values), cmd, len(values))

    def _note_arity_from_error(self, message: str) -> bool:
        """
        Learn the ring command's arity from the device's own complaint.

        Returns True if something was learned, so the caller can retry
        instead of counting the call as a hard failure.
        """
        match = _ARITY_ERROR.search(message)
        if not match:
            return False
        wanted = int(match.group(1))
        if self._commands.ring_arity == wanted:
            return False
        self._commands.ring_arity = wanted
        log.info("LED ring command wants %d values (learned from the "
                 "device's own error) — adapting", wanted)
        if self.cfg.led_count and wanted != int(self.cfg.led_count) \
                and wanted not in (1, 3):
            log.warning("led.led_count is %d but the ring command wants %d "
                        "values — set led_count to %d",
                        int(self.cfg.led_count), wanted, wanted)
        return True

    def _disable_auto_mode(self) -> None:
        """
        Take the ring off the firmware's own animation (LED_AUTO_MODE = 0).

        Without this the DSP keeps repainting the ring on speech, claps and
        its own DOA estimate, and every frame written here is overwritten a
        moment later — which looks exactly like the RED/BLUE flicker
        requirement 5 forbids, while having nothing to do with this code.
        """
        name = self._commands.auto_mode
        if not name:
            log.warning("no automatic-LED-mode command found on this array — "
                        "the firmware may keep animating the ring on its own "
                        "and fight the target indication")
            return
        if self._invoke(name, self.cfg.auto_mode_off_value):
            self._auto_mode_disabled = True
            log.info("array LED automatic mode disabled (%s %s)", name,
                     self.cfg.auto_mode_off_value)

    # ── One cycle ──────────────────────────────────────────────

    def _tick(self) -> None:
        if self.status is not LedStatus.ACTIVE:
            if self._retry_after and time.monotonic() >= self._retry_after:
                log.info("retrying LED hardware after backoff")
                self._retry_after = 0.0
                self._failures = 0
                self._setup()
            return

        frame = self._effective_frame()
        target = self._render(frame)
        self._apply(target)

    def _effective_frame(self) -> LedFrame:
        """
        The frame to render, after the staleness watchdog.

        Requirement 14: if the audio subsystem wedges, nobody will ever
        submit a SEARCHING frame, and a naive controller would hold the red
        sector lit forever — telling an operator a drone is present when
        the system in fact stopped looking. Silence for longer than
        watchdog_s therefore falls back to searching on its own.
        """
        frame = self._requested.get() or LedFrame()
        if self._last_submit <= 0.0:
            return frame
        age = time.monotonic() - self._last_submit
        if age > self.cfg.watchdog_s and frame.mode.is_alarm:
            log.warning("no LED frame for %.1fs — falling back to SEARCHING "
                        "(the station may have stalled)", age)
            return LedFrame(LedMode.SEARCHING, None)
        return frame

    def _render(self, frame: LedFrame) -> Dict[int, Colour]:
        """Frame -> the colour every LED should be showing."""
        cfg = self.cfg
        n = int(cfg.led_count)

        if frame.mode is LedMode.OFF:
            return {i: (0, 0, 0) for i in range(n)}

        base = _rgb(cfg.colour_searching)
        if not frame.mode.is_alarm:
            return {i: base for i in range(n)}

        alarm = _rgb(cfg.colour_alarm if frame.mode is LedMode.ALARM
                     else cfg.colour_coasting)

        if frame.bearing_deg is None or not self._commands.can_drive_sector(n):
            # An alarm with no bearing must not point anywhere. Lighting the
            # whole ring red says "a drone is confirmed, direction unknown",
            # which is true; lighting one arbitrary LED would not be.
            return {i: alarm for i in range(n)}

        centre = bearing_to_led_index(
            frame.bearing_deg, n,
            doa_offset_deg=self._doa_offset, doa_invert=self._doa_invert,
            led_zero_offset_deg=cfg.led_zero_offset_deg,
            clockwise=cfg.led_index_clockwise)
        sector = set(sector_indices(centre, cfg.sector_leds, n))
        return {i: (alarm if i in sector else base) for i in range(n)}

    def _apply(self, target: Dict[int, Colour]) -> None:
        """Write only what changed, and at most MAX_WRITES_PER_CYCLE of it."""
        now_t = time.monotonic()
        if now_t - self._last_write_time < self.cfg.min_write_interval_s:
            return

        n = int(self.cfg.led_count or 0)

        # ── Best case: the ring command takes one value per LED ──
        # The whole ring, sector included, goes out in ONE subprocess call.
        # This is the path the observed XVF3800 firmware uses.
        if self._commands.ring_is_per_pixel(n):
            if self._shadow == target:
                return
            values = [str(self._pack(target[i])) for i in range(n)]
            if self._write_ring(values):
                self._shadow = dict(target)
                self._ring_shadow = None
                self._last_write_time = now_t
            return

        # ── Uniform colour and a whole-ring command: one write replaces N ──
        colours = set(target.values())
        if len(colours) == 1 and self._commands.ring_colour:
            colour = next(iter(colours))
            if self._ring_shadow == colour:
                return
            values = ([str(self._pack(colour))]
                      if self._commands.ring_arity == 1
                      else [str(v) for v in colour])
            if self._write_ring(values):
                self._ring_shadow = colour
                self._shadow = dict.fromkeys(self._shadow, colour)
                self._last_write_time = now_t
            return

        if not self._commands.led_colour:
            return                       # cannot address individual LEDs

        pending = [(i, c) for i, c in sorted(target.items())
                   if self._shadow.get(i) != c]
        if not pending:
            return
        self._ring_shadow = None
        for index, colour in pending[:self.MAX_WRITES_PER_CYCLE]:
            if not self._invoke(self._commands.led_colour, str(index),
                                *map(str, colour)):
                return
            self._shadow[index] = colour
        self._last_write_time = now_t
        # ⚠️ Do NOT set _wake here to "hurry up" the rest of a repaint.
        # The early return above rate-limits on _last_write_time, so a set
        # event would make wait() return instantly and spin the thread for
        # the whole min_write_interval_s without doing any work — CPU taken
        # from the camera and audio threads for nothing. The loop already
        # wakes every min_write_interval_s, which is exactly when the next
        # batch becomes eligible.

    def _pack(self, colour: Colour) -> int:
        """
        One LED colour as the single integer a per-pixel command takes.

        ⚠️ THE ONE THING THIS MODULE COULD NOT READ OFF THE DEVICE. The
        firmware reports how MANY values it wants (measured, see
        _measure_ring_arity) but not how each one is encoded, and reading
        the ring back on a dark array returns zeros, which say nothing.
        24-bit 0xRRGGBB is the near-universal convention and is therefore
        the default — but it IS an assumption, and it is the only one left
        in this file. If red and blue come out swapped on the ring, set
        `led.led_value_order` to "bgr"; the geometry and the state logic are
        unaffected either way.
        """
        r, g, b = colour
        if str(self.cfg.led_value_order).lower() == "bgr":
            r, b = b, r
        return (r << 16) | (g << 8) | b

    # ── Device I/O ─────────────────────────────────────────────

    def _write_ring(self, values: Sequence[str]) -> bool:
        """
        Write the ring command, retrying once if the device corrects us.

        The firmware states its own arity when it rejects a call. Learning
        from that and retrying immediately means a mismatch costs one wasted
        call at startup instead of leaving the ring dark until somebody
        reads the log.
        """
        assert self._commands.ring_colour is not None
        if self._invoke(self._commands.ring_colour, *values):
            return True
        if not self._arity_just_learned:
            return False
        self._arity_just_learned = False
        n = int(self.cfg.led_count or 0)
        wanted = self._commands.ring_arity
        if wanted == n and n:
            return False        # caller will rebuild per-pixel next cycle
        if wanted == 1 and len(values) == 3:
            r, g, b = (int(v) for v in values)
            return self._invoke(self._commands.ring_colour,
                                str(self._pack((r, g, b))))
        return False

    def _invoke(self, command: str, *args: str) -> bool:
        """
        Run one xvf_host.py control command. Returns success.

        ⚠️ VALUES GO BEHIND A FLAG, NOT POSITIONALLY. The utility's own
        usage line is

            xvf_host.py [-h] [-l] [--vid VID] [--pid PID]
                        [--values VALUES [VALUES ...]] [COMMAND]

        so a write is `xvf_host.py LED_RING_COLOR --values 0 0 40`.
        Appending the numbers straight after the command name — which is
        what doa.py's READ path looks like, because a read passes no values
        at all — makes argparse reject them as unrecognized arguments and
        every write silently fails. `led.values_flag` exists so a different
        release of the utility can be accommodated without editing code.

        Never raises. Every failure mode of the array — unplugged, busy,
        permission denied, utility missing — has to end up as a log line and
        a status change, because the microphone, the camera and the radar
        must keep working regardless of what the indicator is doing.
        """
        assert self._script_path is not None
        argv = [command]
        if args:
            flag = self.cfg.values_flag
            argv += ([flag, *args] if flag else list(args))
        try:
            out = subprocess.run(
                [sys.executable, str(self._script_path), *argv],
                capture_output=True, text=True,
                timeout=self.cfg.command_timeout_s,
                cwd=str(self._script_path.parent))
        except subprocess.TimeoutExpired:
            self._note_failure(f"timeout running {command}")
            return False
        except OSError as exc:
            self._note_failure(f"cannot run xvf_host.py: {exc}")
            return False

        if out.returncode != 0:
            snippet = " ".join(((out.stderr or out.stdout or "").split()))[:120]
            self._note_failure(f"{command} failed: {snippet}")
            return False

        self._writes += 1
        self._failures = 0
        return True

    def _note_failure(self, message: str) -> None:
        # A rejection that tells us the command's real arity is information,
        # not a fault: adapt and let the caller retry rather than counting
        # it toward the give-up limit.
        if self._note_arity_from_error(message):
            self._arity_just_learned = True
            return
        self._failures += 1
        if self._failures == 1 or self._failures == \
                self.cfg.max_consecutive_failures:
            log.warning("LED: %s (failure %d)", message, self._failures)
        if self._failures >= self.cfg.max_consecutive_failures:
            self._set_status(LedStatus.UNAVAILABLE,
                             f"{self._failures} consecutive failures: "
                             f"{message}")
            self._retry_after = time.monotonic() + self.RETRY_INTERVAL_S
            self._shadow = dict.fromkeys(self._shadow, None)
            self._ring_shadow = None

    def _set_status(self, status: LedStatus, detail: str = "") -> None:
        if status is not self.status or detail != self.detail:
            level = (logging.INFO if status.ok or status is LedStatus.DISABLED
                     else logging.WARNING)
            log.log(level, "LED status: %s%s", status.value,
                    f" — {detail}" if detail else "")
        self.status = status
        self.detail = detail

    def _shutdown_ring(self) -> None:
        """
        Leave the ring in a defensible state on the way out.

        Best effort only, and bounded: shutdown must not hang on a device
        that has stopped answering. A ring left showing a red target after
        the station exited would be actively misleading, so searching-blue
        is attempted first and darkness accepted as the fallback.
        """
        if self.status is not LedStatus.ACTIVE:
            return
        try:
            base = _rgb(self.cfg.colour_searching)
            n = int(self.cfg.led_count or 0)
            if self._commands.ring_is_per_pixel(n):
                self._write_ring([str(self._pack(base))] * n)
            elif self._commands.ring_colour:
                self._write_ring([str(self._pack(base))]
                                 if self._commands.ring_arity == 1
                                 else [str(v) for v in base])
            elif self._commands.led_colour and n:
                for i in range(n):
                    self._invoke(self._commands.led_colour, str(i),
                                 *map(str, base))
        except Exception as exc:
            log.debug("LED shutdown write failed: %s", exc)
        log.info("LED worker stopped after %d hardware writes", self._writes)

    # ── Diagnostics ────────────────────────────────────────────

    @property
    def writes(self) -> int:
        return self._writes


def _rgb(value) -> Colour:
    """Coerce a colour from config (tuple from Python, list from JSON)."""
    r, g, b = (int(v) for v in tuple(value)[:3])
    clamp = lambda v: max(0, min(255, v))          # noqa: E731
    return clamp(r), clamp(g), clamp(b)


# ═══════════════════════════════════════════════════════════════
#  Probe / self-test
# ═══════════════════════════════════════════════════════════════

def _probe() -> int:
    """Report what the attached array actually supports. Needs hardware."""
    import calibration
    from fusion_config import load as load_config

    print("=" * 66)
    print("respeaker_led.py — hardware probe")
    print("=" * 66)

    path = find_xvf_host()
    print(f"\nxvf_host.py: {path or 'NOT FOUND (set XVF_HOST_PATH)'}")
    if path is None:
        print("\nThis is the only control path to the array that this "
              "project has verified.\nIt ships with the ReSpeaker; it is not "
              "part of this repository.")
        return 1

    cfg = load_config()
    commands = resolve_commands(cfg.led, path, cfg.led.command_timeout_s)
    print(f"\ncommands reported: {len(commands.available)}")
    led_cmds = commands.led_related()
    if led_cmds:
        print("LED-related commands:")
        for c in led_cmds:
            print(f"   {c}")
    else:
        print("no LED-related commands were reported by this firmware")

    print(f"\nresolution: {commands.detail}")

    # Arity, read back from the device. This is the number that decides
    # whether the target SECTOR can be drawn at all, and it cannot be
    # guessed from the command's name — the observed LED_RING_COLOR takes
    # one value per LED despite sounding like one colour for the ring.
    suggested_count = cfg.led.led_count
    if commands.ring_colour:
        values = read_command_values(path, commands.ring_colour,
                                     cfg.led.command_timeout_s)
        if values:
            print(f"\n{commands.ring_colour} currently returns {len(values)} "
                  f"value(s): {' '.join(str(int(v)) for v in values[:16])}")
            if len(values) not in (1, 3):
                suggested_count = len(values)
                print(f"   -> one value per LED: this ring has "
                      f"{len(values)} LEDs, and the whole target sector "
                      f"paints in a single call.")
            else:
                print("   -> a single colour for the whole ring: the target "
                      "SECTOR cannot be drawn, only the target STATE.")
        else:
            print(f"\n{commands.ring_colour}: could not read its current "
                  f"values — arity will be learned from the device's first "
                  f"rejection instead.")

    print("\nput the confirmed names into fusion_config.json:")
    print('   { "led": {')
    print(f'       "led_count": {suggested_count or "<count them>"},')
    print(f'       "cmd_auto_mode":   {_q(commands.auto_mode)},')
    print(f'       "cmd_ring_colour": {_q(commands.ring_colour)},')
    print(f'       "cmd_led_colour":  {_q(commands.led_colour)}')
    print("   } }")

    calib = calibration.load()
    print(f"\ndoa_offset_deg = {calib.get('doa_offset_deg')} "
          f"(undone before choosing an LED — see bearing_to_led_index)")
    return 0


def _q(value: Optional[str]) -> str:
    return f'"{value}"' if value else "null  /* not found */"


def _selftest() -> int:
    """Geometry and mapping only — runs anywhere, no hardware."""
    print("=" * 66)
    print("respeaker_led.py — self-test (no hardware required)")
    print("=" * 66)

    print("\n1. Bearing -> LED index on a 12-LED ring, no mount offset:")
    for bearing in (0.0, 30.0, 90.0, 142.0, 180.0, 270.0, 359.0):
        idx = bearing_to_led_index(bearing, 12)
        print(f"   {bearing:6.1f}° -> LED {idx:2d}  "
              f"sector {sector_indices(idx, 3, 12)}")

    print("\n2. The mount offset is UNDONE, not applied twice:")
    print("   installation bearing 142°, doa_offset_deg = +40°")
    print(f"   naive (wrong):    LED {bearing_to_led_index(142.0, 12)}")
    print(f"   correct:          LED "
          f"{bearing_to_led_index(142.0, 12, doa_offset_deg=40.0)}")
    print("   ^ the array itself hears the drone at 102°, not 142°")

    print("\n3. Sector wraps across LED 0:")
    print(f"   centre 0, span 3 -> {sector_indices(0, 3, 12)}")
    print(f"   centre 11, span 5 -> {sector_indices(11, 5, 12)}")

    print("\n4. A frame is a pure function of state — same state in, "
          "identical frame out:")
    a = LedFrame(LedMode.ALARM, 142.0)
    b = LedFrame(LedMode.ALARM, 142.0)
    print(f"   {a} == {b} -> {a == b}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    # A mistyped flag must not silently run the wrong thing. `--prob`
    # quietly running the offline self-test looks exactly like a probe that
    # found no hardware, which is the most misleading failure this script
    # could have.
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    unknown = [a for a in flags if a not in ("--probe", "--self-test", "-h",
                                             "--help")]
    if unknown:
        print(f"unknown option(s): {' '.join(unknown)}\n"
              f"usage: respeaker_led.py [--probe | --self-test]\n"
              f"   --probe      ask the attached array what it supports "
              f"(needs the hardware)\n"
              f"   --self-test  geometry checks only, runs anywhere "
              f"(default)")
        sys.exit(2)
    if "-h" in flags or "--help" in flags:
        print("usage: respeaker_led.py [--probe | --self-test]")
        sys.exit(0)
    if "--probe" in flags:
        sys.exit(_probe())
    sys.exit(_selftest())

#!/usr/bin/env python3
"""
bearing_frame.py — THE canonical target direction, and every transform out of it.

═══════════════════════════════════════════════════════════════════
THE CANONICAL BEARING
═══════════════════════════════════════════════════════════════════

    canonical_bearing_deg
        0    = the direction the INSTALLATION faces ("front" / north)
        +    = CLOCKWISE seen from above (90 = right/east, 270 = left/west)
        range [0, 360)

This convention was not invented here. It is the one radar_overlay.py
already implements, and that can be proved by evaluating its own code:

    _polar_to_xy(cx, cy, bearing, r) = (cx + r·cos(bearing−90°),
                                        cy + r·sin(bearing−90°))

    bearing   0 -> (cx,   cy−r)  UP     = north
    bearing  90 -> (cx+r, cy  )  RIGHT  = east
    bearing 180 -> (cx,   cy+r)  DOWN   = south
    bearing 270 -> (cx−r, cy  )  LEFT   = west

camera_cue.py is consistent with it too: it puts a target at
x = W/2 + f·tan(bearing − boresight), so a bearing GREATER than the
boresight lands to the RIGHT of the image centre — which is only correct
if bearing increases clockwise.

═══════════════════════════════════════════════════════════════════
⚠️  WHY A REFLECTION IS NOT AN OFFSET  (the root cause of the mirroring)
═══════════════════════════════════════════════════════════════════

A direction sensor can disagree with the canonical frame in two
independent ways:

    ROTATION    its zero points somewhere else          (one parameter)
    HANDEDNESS  its angles run the other way round      (one BIT)

Both must be calibrated. An installation that only ever adjusts the
rotation cannot represent a handedness error at all:

    correct:   canonical = (zero − raw) mod 360      [mirrored source]
    offset:    canonical = (raw + off)  mod 360

    equal  =>  2·raw = zero − off  =>  ONE value of raw

So an offset-only correction is exact at exactly ONE bearing and wrong
everywhere else — and the error grows at TWICE the rate the target moves.
A target tuned to look right ahead reads correctly, and the same target
90° away reads 180° out, i.e. on the opposite side of the display.

That is precisely the reported symptom: "the arrow correctly detects
approximately WEST but can occasionally rotate toward EAST".

Handedness therefore has its own parameter here, per source, and the
calibration procedure that determines it needs TWO known directions —
one point can only ever fit the rotation.

═══════════════════════════════════════════════════════════════════
⚠️  EVERY SOURCE HAS ITS OWN CONVENTION
═══════════════════════════════════════════════════════════════════

    SRP-PHAT      PROVEN from the code. doa.ArrayDOA builds its steering
                  vectors as unit = (cos θ, sin θ) over `mic_positions_m`,
                  i.e. θ = atan2(y, x): zero on the +x axis, increasing
                  COUNTER-CLOCKWISE. Its handedness is therefore KNOWN to
                  be anti-clockwise; only where +x physically points is
                  unknown.

    XVF3800 USB   UNKNOWN — REQUIRES PHYSICAL CALIBRATION. The azimuth
                  comes out of Seeed's firmware through xvf_host.py and
                  nothing in this repository documents its zero or its
                  handedness.

These two are NOT interchangeable, and applying one calibration to both —
which is what a single shared doa_offset_deg/doa_invert does — guarantees
that at most one of them is right. Each source gets its own convention
below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


def wrap360(deg: float) -> float:
    """Wrap to [0, 360)."""
    return float(deg) % 360.0


def wrap180(deg: float) -> float:
    """
    Wrap to [−180, +180). The only correct way to subtract two bearings.

    Exactly 180 comes out as −180. Same direction either way, and it is
    always outside every lens on this station, so no consumer can tell.
    Matching camera_cue.wrap_signed_deg exactly matters more than the sign
    of a degenerate case.
    """
    return (float(deg) + 180.0) % 360.0 - 180.0


def angular_distance(a: float, b: float) -> float:
    """Shortest angle between two bearings, 0..180. 359 to 1 is 2, not 358."""
    return abs(wrap180(a - b))


def circular_mean_deg(values, weights=None) -> float:
    """
    Mean of angles. The arithmetic mean is wrong here: mean(350, 10) = 180.

    Kept in this module so that every layer that averages a direction uses
    the same implementation as every other one.
    """
    values = list(values)
    if not values:
        return 0.0
    if weights is None:
        weights = [1.0] * len(values)
    sx = sum(w * math.sin(math.radians(v)) for v, w in zip(values, weights))
    cx = sum(w * math.cos(math.radians(v)) for v, w in zip(values, weights))
    if abs(sx) < 1e-12 and abs(cx) < 1e-12:
        return 0.0
    return wrap360(round(math.degrees(math.atan2(sx, cx)), 6))


# ═══════════════════════════════════════════════════════════════
#  Source conventions
# ═══════════════════════════════════════════════════════════════

class Handedness(Enum):
    """Which way a sensor's angles run, seen from above."""
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CCW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SourceConvention:
    """
    How to turn one sensor's raw angle into the canonical bearing.

        canonical = (zero_deg ± raw) mod 360

    `zero_deg` is the canonical bearing the sensor calls 0. `handedness`
    says whether its angles already run clockwise (add) or run the other
    way (subtract).

    UNKNOWN handedness is NOT silently treated as clockwise. A source whose
    handedness has never been measured produces an UNCALIBRATED bearing,
    which the display marks and the LED and camera cue refuse to point
    with — they assert a physical direction, and a coin-flip on handedness
    is a 50% chance of pointing an operator at the opposite horizon.
    """

    zero_deg: float = 0.0
    handedness: Handedness = Handedness.UNKNOWN
    label: str = ""

    @property
    def calibrated(self) -> bool:
        return self.handedness is not Handedness.UNKNOWN

    def to_canonical(self, raw_deg: float) -> float:
        signed = (raw_deg if self.handedness is Handedness.COUNTER_CLOCKWISE
                  else raw_deg)
        if self.handedness is Handedness.COUNTER_CLOCKWISE:
            signed = -raw_deg
        return wrap360(self.zero_deg + signed)

    def from_canonical(self, canonical_deg: float) -> float:
        """Inverse, for driving hardware that speaks the sensor's frame."""
        delta = wrap360(canonical_deg - self.zero_deg)
        if self.handedness is Handedness.COUNTER_CLOCKWISE:
            delta = wrap360(-delta)
        return delta

    def describe(self) -> str:
        if not self.calibrated:
            return (f"{self.label or 'source'}: UNKNOWN — REQUIRES PHYSICAL "
                    f"CALIBRATION (run: python calibrate.py doa)")
        return (f"{self.label or 'source'}: zero at {self.zero_deg:+.0f}deg, "
                f"{self.handedness.value}")


#: SRP-PHAT's handedness is PROVEN by doa.ArrayDOA's own steering vectors
#: (unit = cos/sin over mic_positions_m => atan2 => counter-clockwise).
#: Where the array's +x axis physically points is NOT proven, so the zero
#: still has to be measured; only the handedness is free.
SRP_HANDEDNESS = Handedness.COUNTER_CLOCKWISE


def source_convention(source: str, cfg: dict) -> SourceConvention:
    """
    Build the convention for one DOA source from radar_calibration.json.

    ⚠️ Per source, deliberately. `doa_offset_deg` / `doa_invert` describe
    the USB DSP, because that is what they were measured against. Applying
    them to SRP-PHAT — whose handedness is known to be the opposite kind of
    thing — is the shared-calibration bug this function exists to end.
    """
    if source == "srp":
        zero = cfg.get("srp_zero_deg")
        if zero is None:
            # Not measured. The handedness is known, the rotation is not,
            # so the result is still uncalibrated.
            return SourceConvention(0.0, Handedness.UNKNOWN, "SRP-PHAT")
        return SourceConvention(float(zero), SRP_HANDEDNESS, "SRP-PHAT")

    if source == "usb":
        if cfg.get("doa_handedness") is None and "doa_invert" not in cfg:
            return SourceConvention(0.0, Handedness.UNKNOWN, "XVF3800 USB")
        explicit = cfg.get("doa_handedness")
        if explicit is not None:
            hand = (Handedness.COUNTER_CLOCKWISE
                    if str(explicit).upper() in ("CCW", "COUNTER_CLOCKWISE")
                    else Handedness.CLOCKWISE)
        else:
            # `doa_invert` is the legacy spelling of the same bit.
            hand = (Handedness.COUNTER_CLOCKWISE if cfg.get("doa_invert")
                    else Handedness.CLOCKWISE)
        return SourceConvention(float(cfg.get("doa_offset_deg", 0.0)), hand,
                                "XVF3800 USB")

    return SourceConvention(0.0, Handedness.UNKNOWN, source or "unknown")


# ═══════════════════════════════════════════════════════════════
#  The canonical bearing itself
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CanonicalBearing:
    """
    One target direction, in the canonical frame, with its provenance.

    Every output layer — radar, LED ring, camera cue — derives from THIS
    and performs only its own output transform. None of them is allowed to
    look at the raw sensor angle or to re-apply a calibration, which is
    what made them disagree.
    """

    deg: Optional[float] = None
    confidence: float = 0.0
    source: str = "none"
    #: The source's convention was fully measured. False = the direction
    #: may be mirrored or rotated; do not point hardware with it.
    calibrated: bool = False
    #: Two or more equally-supported directions were seen this update.
    #: A tie between opposite beams is the classic 180-degree flip.
    ambiguous: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.deg is not None

    @property
    def usable_for_pointing(self) -> bool:
        """Safe to aim physical hardware (LED ring) or a camera overlay."""
        return self.ok and self.calibrated

    def text(self) -> str:
        if self.deg is None:
            return "N/A"
        text = f"{self.deg:.0f}deg"
        if not self.calibrated:
            text += " (UNCAL)"
        if self.ambiguous:
            text += " (+/-)"
        return text


# ═══════════════════════════════════════════════════════════════
#  Transforms OUT of canonical — one per output layer
# ═══════════════════════════════════════════════════════════════

def to_radar_screen_deg(canonical_deg: float) -> float:
    """
    Canonical bearing -> the angle radar_overlay._polar_to_xy consumes.

    Identity, and that is the point: the radar defines the canonical frame,
    so it applies no correction of its own. Kept as a named function so
    that a future change to the radar cannot quietly become a second,
    competing convention.
    """
    return wrap360(canonical_deg)


def to_camera_relative_deg(canonical_deg: float,
                           camera_boresight_deg: float) -> float:
    """
    Canonical bearing -> signed angle off the camera's optical axis.

    Positive = to the RIGHT in the image, which follows from the canonical
    frame being clockwise and the camera looking along its boresight.
    """
    return wrap180(canonical_deg - camera_boresight_deg)


def to_led_index(canonical_deg: float, led_count: int,
                 led_zero_deg: float = 0.0,
                 clockwise: bool = True) -> int:
    """
    Canonical bearing -> index of the LED closest to it.

    ⚠️ THIS TAKES THE CANONICAL BEARING, NOT A RAW SENSOR ANGLE.

    An earlier version un-applied `doa_offset_deg` first, on the reasoning
    that "the ring is bolted to the array, not to the compass". That is a
    SECOND transform on an already-transformed angle, and it makes the ring
    depend on the microphone's calibration: re-calibrate the DOA and the
    ring silently rotates, even though neither the ring nor the drone
    moved. It also assumed the DSP's azimuth zero coincides with the ring's
    LED 0, which nothing establishes.

    The ring needs exactly two numbers of its own, both physical facts
    about the board and independent of every other subsystem:

        led_zero_deg  the canonical bearing LED 0 sits at
        clockwise     whether indices run clockwise seen from above

    Both are measured with `python respeaker_led.py --bearing sweep`.
    """
    if led_count <= 0:
        raise ValueError("led_count must be positive")
    offset = wrap360(canonical_deg - led_zero_deg)
    if not clockwise:
        offset = wrap360(-offset)
    return int(round(offset / 360.0 * led_count)) % led_count


def led_sector(centre_index: int, span: int, led_count: int) -> Tuple[int, ...]:
    """The LED indices of a sector of `span` LEDs centred on `centre_index`."""
    span = max(1, min(int(span), int(led_count)))
    half = (span - 1) // 2
    start = centre_index - half
    return tuple((start + i) % led_count for i in range(span))


# ═══════════════════════════════════════════════════════════════
#  The audit table
# ═══════════════════════════════════════════════════════════════

#: Every layer's convention, and how it was established. "PROVEN" means it
#: was derived by evaluating the code, not read from a comment.
COORDINATE_AUDIT = (
    # layer,          zero,                     positive,   units,  wrap, basis
    ("Physical mic",  "UNKNOWN - REQUIRES PHYSICAL CALIBRATION",
     "UNKNOWN", "deg", "-", "mic_positions_m is a DEFAULT in calibration.py, "
     "never measured against the real board"),
    ("SRP-PHAT out",  "+x axis of mic_positions_m",
     "counter-clockwise", "deg", "0..360", "PROVEN: ArrayDOA steering "
     "vectors are (cos, sin) => atan2"),
    ("XVF3800 USB",   "UNKNOWN - REQUIRES PHYSICAL CALIBRATION",
     "UNKNOWN", "deg", "0..360", "Seeed firmware; nothing in this repo "
     "documents it"),
    ("Canonical",     "installation front / north",
     "clockwise", "deg", "0..360", "DEFINED HERE; matches the radar"),
    ("Radar screen",  "up",
     "clockwise", "deg", "0..360", "PROVEN: _polar_to_xy(b) = "
     "(cos(b-90), sin(b-90)) with y down"),
    ("LED ring",      "led_zero_deg (canonical)",
     "led_index_clockwise", "index", "mod N", "CALIBRATE: respeaker_led.py "
     "--bearing sweep"),
    ("Camera cue",    "camera_boresight_deg (canonical)",
     "right in image", "px", "+/-HFOV/2", "PROVEN: x = W/2 + f*tan(rel)"),
    ("Screen",        "top-left",
     "x right, y down", "px", "-", "OpenCV"),
)


@dataclass(frozen=True)
class LayerAim:
    """Where one output layer points, reduced to a comparable direction."""
    radar_screen: Tuple[float, float]   # unit vector, +x right, +y DOWN
    led_index: int
    camera_rel_deg: float

    @property
    def radar_side(self) -> str:
        x, y = self.radar_screen
        if abs(x) < 0.35:
            return "FRONT" if y < 0 else "BEHIND"
        return "RIGHT" if x > 0 else "LEFT"

    def led_side(self, led_count: int, physical_zero_deg: float = 0.0,
                 physical_clockwise: bool = True) -> str:
        """
        Where the chosen LED PHYSICALLY points.

        ⚠️ Takes the ring's PHYSICAL layout, which is not the same thing as
        the configured layout. Feeding the configured values back in here
        would just undo `to_led_index` and make the check tautological — it
        would pass no matter what the configuration said, which is exactly
        the sort of test that lets a mirrored ring reach hardware.

        In production the two agree, and the check confirms the software
        aims correctly. In the tests they are deliberately made to differ,
        which is how the check is proved to have teeth.
        """
        deg = self.led_index / led_count * 360.0
        if not physical_clockwise:
            deg = wrap360(-deg)
        deg = wrap360(deg + physical_zero_deg)
        if deg < 45 or deg >= 315:
            return "FRONT"
        if deg < 135:
            return "RIGHT"
        if deg < 225:
            return "BEHIND"
        return "LEFT"

    @property
    def camera_side(self) -> str:
        if abs(self.camera_rel_deg) >= 135.0:
            return "BEHIND"
        if abs(self.camera_rel_deg) < 45.0:
            return "FRONT"
        return "RIGHT" if self.camera_rel_deg > 0 else "LEFT"


def aim_of(canonical_deg: float, led_count: int = 12,
           led_zero_deg: float = 0.0, led_clockwise: bool = True,
           camera_boresight_deg: float = 0.0) -> LayerAim:
    """
    Drive all three output transforms from ONE canonical bearing.

    This is the deterministic test harness for the whole directional chain:
    inject a known bearing, ask every layer where it points, and require
    that they agree. It needs no microphone, no camera and no LED ring, so
    the agreement can be asserted in CI instead of squinted at on hardware.
    """
    screen = math.radians(to_radar_screen_deg(canonical_deg) - 90.0)
    return LayerAim(
        radar_screen=(math.cos(screen), math.sin(screen)),
        led_index=to_led_index(canonical_deg, led_count, led_zero_deg,
                               led_clockwise),
        camera_rel_deg=to_camera_relative_deg(canonical_deg,
                                              camera_boresight_deg))


def verify_chain(led_count: int = 12, led_zero_deg: float = 0.0,
                 led_clockwise: bool = True,
                 camera_boresight_deg: float = 0.0,
                 physical_zero_deg: Optional[float] = None,
                 physical_clockwise: Optional[bool] = None) -> List[tuple]:
    """
    Check every cardinal direction through radar, LED and camera.

    `led_*` are what the CONFIGURATION says; `physical_*` are what the ring
    actually is. They default to being the same, which is the production
    case and must agree. Passing a different physical layout simulates a
    miscalibrated ring, and every direction must then be reported as
    disagreeing — that is what proves this check can fail.

    Returns (bearing, expected, radar, led, camera, agree) per direction.
    """
    phys_zero = (led_zero_deg if physical_zero_deg is None
                 else physical_zero_deg)
    phys_cw = (led_clockwise if physical_clockwise is None
               else physical_clockwise)
    out: List[tuple] = []
    for bearing, expected in ((0.0, "FRONT"), (90.0, "RIGHT"),
                              (180.0, "BEHIND"), (270.0, "LEFT")):
        aim = aim_of(bearing, led_count, led_zero_deg, led_clockwise,
                     camera_boresight_deg)
        radar = aim.radar_side
        led = aim.led_side(led_count, phys_zero, phys_cw)
        camera = aim.camera_side
        out.append((bearing, expected, radar, led, camera,
                    radar == led == camera == expected))
    return out


def audit_table() -> str:
    """The coordinate audit as a printable table."""
    head = ("Layer", "Zero", "Positive", "Units", "Wrap")
    widths = [16, 42, 20, 6, 8]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for layer, zero, pos, units, wrap, _basis in COORDINATE_AUDIT:
        cells = (layer, zero, pos, units, wrap)
        lines.append("| " + " | ".join(c.ljust(w)
                                       for c, w in zip(cells, widths)) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 78)
    print("bearing_frame.py — coordinate audit")
    print("=" * 78)
    print(audit_table())
    print("\nBasis for each row:")
    for layer, _z, _p, _u, _w, basis in COORDINATE_AUDIT:
        print(f"   {layer:<16} {basis}")

    print("\n" + "=" * 78)
    print("Why an offset cannot fix a mirrored source")
    print("=" * 78)
    mirrored = SourceConvention(0.0, Handedness.COUNTER_CLOCKWISE, "mirrored")
    print(f"{'raw':>6} {'correct':>9} {'offset -180':>12} {'error':>8}")
    for raw in range(0, 360, 45):
        correct = mirrored.to_canonical(raw)
        naive = wrap360(raw - 180.0)
        print(f"{raw:6d} {correct:9.0f} {naive:12.0f} "
              f"{angular_distance(correct, naive):8.0f}")
    print("\nThe offset is exact at ONE bearing and 180deg out a quarter turn")
    print("away. This is the 'sometimes WEST, sometimes EAST' symptom.")

    print("\n" + "=" * 78)
    print("Round-trip: canonical -> every layer -> back")
    print("=" * 78)
    N = 12
    print(f"{'canonical':>10} {'radar':>7} {'LED':>5} {'cam rel (bore 0)':>18}")
    for b in (0.0, 45.0, 90.0, 180.0, 270.0, 359.0):
        print(f"{b:10.0f} {to_radar_screen_deg(b):7.0f} "
              f"{to_led_index(b, N):5d} "
              f"{to_camera_relative_deg(b, 0.0):18.0f}")

    print("\n" + "=" * 78)
    print("DETERMINISTIC CHAIN CHECK — one bearing in, three layers out")
    print("=" * 78)
    print(f"{'bearing':>8} {'expected':>9} {'radar':>7} {'LED':>7} "
          f"{'camera':>7}  agree")
    for bearing, expected, radar, led, camera, agree in verify_chain():
        print(f"{bearing:8.0f} {expected:>9} {radar:>7} {led:>7} "
              f"{camera:>7}  {'YES' if agree else 'NO'}")

    print("\nAnd with a MIRRORED ring, the check catches it:")
    for bearing, expected, radar, led, camera, agree in verify_chain(
            physical_clockwise=False):
        print(f"{bearing:8.0f} {expected:>9} {radar:>7} {led:>7} "
              f"{camera:>7}  {'YES' if agree else 'NO'}")

    print("\nWrap sanity — 359 to 1 is a 2deg move, not 358:")
    print(f"   angular_distance(359, 1) = {angular_distance(359.0, 1.0):.0f}")
    print(f"   circular_mean(350, 10)   = {circular_mean_deg([350.0, 10.0]):.0f}")

#!/usr/bin/env python3
"""
camera_cue.py — Acoustic bearing → camera image coordinates.

═══════════════════════════════════════════════════════════════════
⚠️  THE CENTRAL HONESTY PROBLEM OF THIS INTEGRATION
═══════════════════════════════════════════════════════════════════

    acoustic bearing  ≠  camera pixel column

They are measurements in two different coordinate frames:

    ACOUSTIC FRAME    azimuth 0–360°, origin = the microphone array's own
                      zero direction, rotated by `doa_offset_deg` from
                      radar_calibration.json. No elevation is measured at
                      all — the ReSpeaker array reports azimuth only.

    CAMERA FRAME      pixels, origin = top-left of a 640x480 image, with
                      an angular scale set by the lens (focal length in
                      pixels) and a pointing direction set by however the
                      camera happens to be bolted down.

Converting one into the other requires knowing the angle between the
array's zero direction and the camera's optical axis. That measurement does
not exist anywhere in this repository, and it cannot be derived from the
code — it is a property of the physical mount.

So this module does NOT invent it. `camera_boresight_deg = None` (the
default) makes every method return "not calibrated", the UI states that
plainly, and no cue is drawn. The full software interface is implemented and
ready; only the one measured number is missing.

═══════════════════════════════════════════════════════════════════
WHAT THE GEOMETRY LOOKS LIKE ONCE CALIBRATED
═══════════════════════════════════════════════════════════════════

For a pinhole camera with horizontal focal length f (pixels) and image
width W, a ray at horizontal angle θ from the optical axis lands at

    x = W/2 + f · tan(θ)

and the half-field of view is

    HFOV/2 = atan( (W/2) / f )

θ here is the SIGNED angle between the target's azimuth and the camera's
boresight azimuth, wrapped to (−180°, +180°]. Outside ±HFOV/2 the target is
simply not in the picture, and the correct output is an off-screen arrow,
not a pixel clamped to the frame edge.

Because there is no acoustic elevation, the cue can only ever be a VERTICAL
BAND spanning the full image height — never a point, never a box. Drawing a
point would assert an elevation measurement the hardware cannot make.

The band's width is the bearing uncertainty mapped through the same
projection, so a low-confidence bearing produces a visibly wider band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from fusion_config import GeometryConfig


# ═══════════════════════════════════════════════════════════════
#  Result
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BearingCue:
    """
    Where in the image the acoustic bearing points.

    `available` False means "cannot be computed" and carries the reason;
    callers must render that reason rather than fall back to a default
    position.
    """
    available: bool = False
    reason: str = "not calibrated"

    x_px: Optional[float] = None          # centre column of the cue band
    half_width_px: Optional[float] = None # uncertainty half-width, pixels
    in_view: bool = False                 # target inside the field of view
    off_screen_side: int = 0              # -1 left, +1 right, 0 in view
    rel_bearing_deg: Optional[float] = None   # signed angle off boresight
    hfov_deg: Optional[float] = None
    frame_width_px: int = 0
    uncertainty_deg: Optional[float] = None

    @property
    def uncertainty_exceeds_fov(self) -> bool:
        """
        True when the bearing is too coarse to localise anything within this
        lens — the cue band is as wide as the picture.

        This is the normal case for the FAR camera: DOA accuracy is roughly
        ±15° at best, while the IMX477's field of view is only 28° wide. The
        UI must say so instead of drawing a confident-looking marker that
        implies precision the sensor does not have.
        """
        if self.half_width_px is None or self.frame_width_px <= 0:
            return False
        # Half the frame is the practical cut-off: a band wider than that
        # no longer tells an operator anything they could not see by
        # glancing at the whole image.
        return self.half_width_px * 2.0 >= 0.5 * self.frame_width_px


# ═══════════════════════════════════════════════════════════════
#  Projection
# ═══════════════════════════════════════════════════════════════

def wrap_signed_deg(angle: float) -> float:
    """Wrap an angle to (−180, +180]."""
    return (angle + 180.0) % 360.0 - 180.0


def horizontal_fov_deg(focal_px: float, width_px: int) -> float:
    """Full horizontal field of view of a pinhole camera, in degrees."""
    return 2.0 * math.degrees(math.atan((width_px / 2.0) / focal_px))


class BearingProjector:
    """
    Projects acoustic azimuths into image columns for one camera.

    Usage:
        proj = BearingProjector(config.geometry, width=640)
        cue  = proj.project(camera_id=0, bearing_deg=142.0, confidence=0.8)
        if cue.available and cue.in_view:
            draw_band(cue.x_px, cue.half_width_px)
    """

    # Bearing uncertainty, in degrees, at zero and at full confidence.
    # These bound the width of the drawn band. They are display parameters,
    # not measurements: the underlying DOA accuracy is roughly ±15° for
    # SRP-PHAT on a 43 mm array (doa.py's own stated figure), and unknown
    # for the XVF3800's internal DSP, so the band is never drawn narrower
    # than MIN_UNCERTAINTY_DEG no matter how confident the reading claims
    # to be.
    MIN_UNCERTAINTY_DEG = 15.0
    MAX_UNCERTAINTY_DEG = 45.0

    def __init__(self, geometry: GeometryConfig, width_px: int):
        self.geometry = geometry
        self.width_px = int(width_px)

    # ── Introspection ──────────────────────────────────────────

    def is_calibrated(self, camera_id: int) -> bool:
        return (self.geometry.camera_boresight_deg.get(camera_id) is not None
                and bool(self.geometry.camera_focal_px.get(camera_id)))

    def status_text(self, camera_id: int) -> str:
        """One-line explanation for the UI."""
        if self.geometry.camera_focal_px.get(camera_id) in (None, 0):
            return "BEARING->VIEW: NO FOCAL CALIBRATION"
        if self.geometry.camera_boresight_deg.get(camera_id) is None:
            return "BEARING->VIEW: NOT CALIBRATED (mount offset unmeasured)"
        hfov = horizontal_fov_deg(
            float(self.geometry.camera_focal_px[camera_id]), self.width_px)
        return f"BEARING->VIEW: ON (HFOV {hfov:.0f}°)"

    # ── Projection ─────────────────────────────────────────────

    def project(self, camera_id: int, bearing_deg: Optional[float],
                confidence: float = 0.0) -> BearingCue:
        if bearing_deg is None:
            return BearingCue(False, "no acoustic bearing")

        focal = self.geometry.camera_focal_px.get(camera_id)
        if not focal:
            return BearingCue(False, "camera focal length not calibrated")

        boresight = self.geometry.camera_boresight_deg.get(camera_id)
        if boresight is None:
            # The deliberate refusal. Everything else is ready; this one
            # number must be measured on the physical mount.
            return BearingCue(False, "camera boresight azimuth not calibrated")

        focal = float(focal)
        hfov = horizontal_fov_deg(focal, self.width_px)
        rel = wrap_signed_deg(float(bearing_deg) - float(boresight))

        # Outside the field of view: report which side, no pixel.
        if abs(rel) >= hfov / 2.0:
            return BearingCue(
                available=True, reason="target outside camera field of view",
                x_px=None, half_width_px=None, in_view=False,
                off_screen_side=1 if rel > 0 else -1,
                rel_bearing_deg=rel, hfov_deg=hfov,
                frame_width_px=self.width_px)

        # Inside: project through the pinhole model.
        x = self.width_px / 2.0 + focal * math.tan(math.radians(rel))

        # Uncertainty band. Confidence 1.0 -> MIN, 0.0 -> MAX.
        c = min(max(float(confidence), 0.0), 1.0)
        unc_deg = (self.MAX_UNCERTAINTY_DEG
                   - c * (self.MAX_UNCERTAINTY_DEG - self.MIN_UNCERTAINTY_DEG))
        # Project both edges and halve the span, so the band widens
        # correctly toward the frame edges where tan() is non-linear.
        lo = max(-hfov / 2.0 + 1e-3, rel - unc_deg)
        hi = min(hfov / 2.0 - 1e-3, rel + unc_deg)
        x_lo = self.width_px / 2.0 + focal * math.tan(math.radians(lo))
        x_hi = self.width_px / 2.0 + focal * math.tan(math.radians(hi))
        half_width = max(4.0, abs(x_hi - x_lo) / 2.0)

        return BearingCue(available=True, reason="", x_px=x,
                          half_width_px=half_width, in_view=True,
                          off_screen_side=0, rel_bearing_deg=rel,
                          hfov_deg=hfov, frame_width_px=self.width_px,
                          uncertainty_deg=unc_deg)

    # ── Reverse direction: pixel → bearing ─────────────────────

    def bearing_of_pixel(self, camera_id: int,
                         x_px: float) -> Optional[float]:
        """
        Azimuth of an image column, in the acoustic frame.

        Used to sanity-check a visual track against the acoustic bearing
        (do the two sensors agree they are looking at the same object?).
        Returns None when uncalibrated — the check is then simply skipped
        rather than performed against a made-up value.
        """
        focal = self.geometry.camera_focal_px.get(camera_id)
        boresight = self.geometry.camera_boresight_deg.get(camera_id)
        if not focal or boresight is None:
            return None
        rel = math.degrees(math.atan((float(x_px) - self.width_px / 2.0)
                                     / float(focal)))
        return (float(boresight) + rel) % 360.0

    def agreement_deg(self, camera_id: int, x_px: float,
                      bearing_deg: Optional[float]) -> Optional[float]:
        """
        Absolute angular disagreement between a visual track and the
        acoustic bearing, or None if it cannot be computed.
        """
        if bearing_deg is None:
            return None
        visual_bearing = self.bearing_of_pixel(camera_id, x_px)
        if visual_bearing is None:
            return None
        return abs(wrap_signed_deg(visual_bearing - float(bearing_deg)))


if __name__ == "__main__":
    from fusion_config import load

    print("=" * 66)
    print("camera_cue.py — self-test")
    print("=" * 66)

    cfg = load()
    proj = BearingProjector(cfg.geometry, width_px=640)

    print("\n1. Default state — mount offset was never measured:")
    for cam in (0, 1):
        print(f"   camera {cam}: {proj.status_text(cam)}")
        cue = proj.project(cam, 142.0, 0.9)
        print(f"      project(142°) -> available={cue.available} "
              f"reason='{cue.reason}'")

    print("\n2. With a hypothetical boresight of 130° on camera 0:")
    cfg.geometry.camera_boresight_deg[0] = 130.0
    proj = BearingProjector(cfg.geometry, width_px=640)
    print(f"   {proj.status_text(0)}")
    hfov = horizontal_fov_deg(cfg.geometry.camera_focal_px[0], 640)
    print(f"   HFOV = {hfov:.1f}°  (half = {hfov/2:.1f}°)")
    for bearing in (130.0, 136.0, 142.0, 145.0, 200.0, 40.0):
        cue = proj.project(0, bearing, confidence=0.8)
        rel = f"{cue.rel_bearing_deg:+.1f}°" if cue.rel_bearing_deg is not None else "?"
        if cue.in_view:
            print(f"   bearing {bearing:5.0f}° (rel {rel:>7}) -> "
                  f"x={cue.x_px:6.1f} px  ±{cue.half_width_px:.0f} px")
        else:
            side = "LEFT" if cue.off_screen_side < 0 else "RIGHT"
            print(f"   bearing {bearing:5.0f}° (rel {rel:>7}) -> "
                  f"OFF-SCREEN {side}: {cue.reason}")

    print("\n3. Round-trip pixel <-> bearing:")
    for x in (0.0, 160.0, 320.0, 480.0, 639.0):
        b = proj.bearing_of_pixel(0, x)
        back = proj.project(0, b, 1.0)
        print(f"   x={x:6.1f} -> bearing {b:6.2f}° -> x={back.x_px:6.1f}")

    print("\n4. Wide lens (camera 1, focal 501.7) sees much more:")
    cfg.geometry.camera_boresight_deg[1] = 130.0
    proj1 = BearingProjector(cfg.geometry, width_px=640)
    print(f"   {proj1.status_text(1)}")
    print()

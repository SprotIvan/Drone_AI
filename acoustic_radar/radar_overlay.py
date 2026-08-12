#!/usr/bin/env python3
"""
radar_overlay.py — Acoustic radar drawn INSIDE the camera window.

This replaces the separate Pygame window of radar_gui.py. The station has
exactly one window; the radar is a widget composited into a corner of the
camera image.

═══════════════════════════════════════════════════════════════════
PERFORMANCE DESIGN
═══════════════════════════════════════════════════════════════════

The radar's artwork splits into two parts:

    STATIC   range rings, ring labels, compass spokes, N/E/S/W marks,
             the panel background and its border. These change only when
             the range scale changes (a handful of times per flight).

    DYNAMIC  the sweep line, the target blip, the trail, the bearing
             wedge, the text readouts. These change every frame.

The static part is rendered ONCE into a cached BGR tile and re-blitted with
a single `numpy` copy per frame; only the dynamic part is actually drawn.
Without this, ~40 cv2 primitive calls would run per frame purely to redraw
an unchanged grid, which is exactly the kind of cost that eats the camera's
frame budget on a Pi.

═══════════════════════════════════════════════════════════════════
WHAT EACH ELEMENT MEANS
═══════════════════════════════════════════════════════════════════

    N marker            top of the dial. Bearings are shown in the
                        acoustic frame AFTER radar_calibration.json's
                        doa_offset_deg has been applied. If that offset has
                        not been calibrated, "N" is the array's own zero
                        direction, not magnetic north — the header says so.
    range rings         concentric circles, labelled in metres. The scale
                        auto-selects from UIConfig.radar_scales_m.
    green sweep         a liveness indicator only. It does NOT represent a
                        physical scanning beam: the microphone array
                        listens in all directions at once. It is drawn dim
                        precisely so it is not mistaken for a measurement.
    solid wedge         the bearing, with an angular width equal to the
                        bearing uncertainty. A wide wedge = a low-
                        confidence direction.
    blip                the target, placed at (bearing, distance). Drawn
                        ONLY when BOTH are known.
    dashed radial       drawn INSTEAD of a blip when the bearing is known
                        but the distance is not. The target is somewhere
                        along that line — the radar refuses to invent a
                        radius for it.
    pulsing outline     drawn when the distance is known but the bearing is
                        not: a ring at the correct radius, all around,
                        because the direction genuinely is unknown.
    trail               previous positions, fading with age.
    mirror ghost        for a 2-microphone array the direction is only
                        known up to a mirror reflection; the alternative
                        solution is drawn as a hollow marker.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from sensor_fusion import FusedTarget, SystemState
from target_state import Approach, Freshness

# ── Palette (BGR) ──
COL_BG = (18, 14, 10)
COL_PANEL_EDGE = (90, 70, 40)
COL_RING = (60, 90, 60)
COL_RING_TEXT = (120, 150, 120)
COL_SPOKE = (45, 65, 45)
COL_SWEEP = (60, 140, 60)
COL_TEXT = (215, 225, 220)
COL_DIM = (130, 140, 135)
COL_ACOUSTIC = (70, 200, 255)      # amber — acoustic contact
COL_ALARM = (60, 60, 255)          # red — confirmed target
COL_VISUAL = (90, 235, 120)        # green — visually confirmed
COL_STALE = (120, 120, 120)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: OpenCV's Hershey fonts are ASCII-only: any non-ASCII codepoint is drawn
#: as "?". Degree and plus-minus signs are unavoidable in this domain, so
#: every string handed to cv2.putText goes through ascii_safe() first.
#: (The same strings keep their proper Unicode form in logs and on the
#: console, which do support UTF-8.)
_ASCII_MAP = {
    "°": "deg",    # °
    "±": "+/-",    # ±
    "–": "-",      # –
    "—": "-",      # —
    "→": "->",     # →
    "≈": "~",      # ≈
}


def ascii_safe(text: str) -> str:
    """Make a string renderable by cv2.putText."""
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _polar_to_xy(cx: float, cy: float, bearing_deg: float,
                 radius_px: float) -> Tuple[int, int]:
    """
    Compass bearing → pixel.

    0° is up (north) and angles increase CLOCKWISE, matching how bearings
    are read. Screen y grows downward, so the standard maths convention
    must be rotated by −90° and the sign of the sine kept as-is:
        screen_angle = bearing − 90°
    """
    rad = math.radians(bearing_deg - 90.0)
    return (int(round(cx + radius_px * math.cos(rad))),
            int(round(cy + radius_px * math.sin(rad))))


class RadarOverlay:
    """
    Renders the acoustic radar into a square tile, then composites it.

    One instance per application; it owns its caches.
    """

    def __init__(self, size_px: int, scales_m: Sequence[float],
                 trail_s: float = 8.0, sweep_deg_per_s: float = 90.0):
        self.size = int(size_px)
        self.scales = tuple(sorted(float(s) for s in scales_m))
        self.trail_s = trail_s
        self.sweep_deg_per_s = sweep_deg_per_s

        self.cx = self.size / 2.0
        self.cy = self.size / 2.0 + self.size * 0.03   # leave room for header
        self.max_radius = self.size * 0.36

        self._static: Optional[np.ndarray] = None
        self._static_scale: Optional[float] = None
        self._sweep_deg = 0.0

    # ── Scale selection ────────────────────────────────────────

    def choose_scale(self, distance_m: Optional[float]) -> float:
        """
        Smallest configured scale that contains the target.

        Hysteresis is unnecessary here because the scale only affects
        drawing; a scale change costs one cached re-render and is visually
        obvious (the ring labels change), so it is not a source of the
        oscillation that matters (camera switching).
        """
        if distance_m is None:
            return self.scales[len(self.scales) // 2]
        for s in self.scales:
            if distance_m <= s:
                return s
        return self.scales[-1]

    # ── Static artwork ─────────────────────────────────────────

    def _build_static(self, scale_m: float) -> np.ndarray:
        tile = np.full((self.size, self.size, 3), COL_BG, dtype=np.uint8)

        # Panel border
        cv2.rectangle(tile, (0, 0), (self.size - 1, self.size - 1),
                      COL_PANEL_EDGE, 1, cv2.LINE_AA)

        cx, cy, R = self.cx, self.cy, self.max_radius

        # Range rings: 4 evenly spaced circles up to the selected scale
        for i in range(1, 5):
            frac = i / 4.0
            r = R * frac
            cv2.circle(tile, (int(cx), int(cy)), int(r), COL_RING, 1,
                       cv2.LINE_AA)
            if i in (2, 4):
                label = f"{scale_m * frac:.0f}m"
                cv2.putText(tile, ascii_safe(label), (int(cx + 4), int(cy - r + 12)),
                            FONT, 0.32, COL_RING_TEXT, 1, cv2.LINE_AA)

        # Compass spokes every 30°
        for ang in range(0, 360, 30):
            inner = R * 0.12
            x1, y1 = _polar_to_xy(cx, cy, ang, inner)
            x2, y2 = _polar_to_xy(cx, cy, ang, R)
            cv2.line(tile, (x1, y1), (x2, y2), COL_SPOKE, 1, cv2.LINE_AA)

        # Cardinal marks
        for ang, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            x, y = _polar_to_xy(cx, cy, ang, R + self.size * 0.045)
            (tw, th), _ = cv2.getTextSize(ascii_safe(name), FONT, 0.4, 1)
            colour = COL_TEXT if name == "N" else COL_DIM
            cv2.putText(tile, ascii_safe(name), (x - tw // 2, y + th // 2), FONT, 0.4,
                        colour, 1, cv2.LINE_AA)

        # Centre = the sensor itself
        cv2.circle(tile, (int(cx), int(cy)), 3, COL_RING_TEXT, -1, cv2.LINE_AA)

        cv2.putText(tile, ascii_safe("ACOUSTIC RADAR"), (8, 14), FONT, 0.38, COL_TEXT, 1,
                    cv2.LINE_AA)
        return tile

    def _static_for(self, scale_m: float) -> np.ndarray:
        if self._static is None or self._static_scale != scale_m:
            self._static = self._build_static(scale_m)
            self._static_scale = scale_m
        return self._static

    # ── Frame render ───────────────────────────────────────────

    def render(self, target: FusedTarget, dt_s: float) -> np.ndarray:
        """Draw the radar tile for this frame. Returns a BGR image."""
        acoustic = target.acoustic
        distance = acoustic.distance_m if acoustic else None
        scale_m = self.choose_scale(distance)

        tile = self._static_for(scale_m).copy()
        cx, cy, R = self.cx, self.cy, self.max_radius

        # ── Liveness sweep (decorative, explicitly not a measurement) ──
        self._sweep_deg = (self._sweep_deg
                           + self.sweep_deg_per_s * max(dt_s, 0.0)) % 360.0
        if target.acoustic_online:
            for i in range(0, 22, 3):
                shade = max(20, 90 - i * 4)
                x, y = _polar_to_xy(cx, cy, self._sweep_deg - i, R)
                cv2.line(tile, (int(cx), int(cy)), (x, y), (0, shade, 0), 1,
                         cv2.LINE_AA)

        px_per_m = R / scale_m

        stale = target.acoustic_freshness is not Freshness.FRESH
        if target.state is SystemState.VISUAL_TRACKING:
            colour = COL_VISUAL
        elif acoustic is not None and acoustic.confirmed:
            colour = COL_ALARM
        else:
            colour = COL_ACOUSTIC
        if stale:
            colour = COL_STALE

        has_contact = (acoustic is not None and acoustic.detected
                       and target.state.has_target)

        if has_contact:
            self._draw_trail(tile, target, px_per_m)
            self._draw_target(tile, target, colour, px_per_m)

        self._draw_header(tile, target, scale_m, stale)
        return tile

    # ── Pieces ─────────────────────────────────────────────────

    def _draw_trail(self, tile: np.ndarray, target: FusedTarget,
                    px_per_m: float) -> None:
        """Past positions, fading with age. Only points with a distance."""
        for age_frac, bearing, dist in target.trail:
            if dist is None:
                continue
            r = min(dist * px_per_m, self.max_radius - 2)
            x, y = _polar_to_xy(self.cx, self.cy, bearing, r)
            fade = max(0.0, 1.0 - age_frac)
            shade = int(40 + 120 * fade)
            cv2.circle(tile, (x, y), 1, (shade // 2, shade, shade // 2), -1,
                       cv2.LINE_AA)

    def _draw_target(self, tile: np.ndarray, target: FusedTarget,
                     colour: Tuple[int, int, int], px_per_m: float) -> None:
        acoustic = target.acoustic
        if acoustic is None:
            return
        cx, cy, R = self.cx, self.cy, self.max_radius

        bearing = acoustic.bearing_deg
        distance = acoustic.distance_m

        # ── Case 1: no bearing ──
        if bearing is None:
            if distance is not None:
                # Distance known, direction not: a full ring at the right
                # radius. Placing a blip anywhere would fabricate a bearing.
                r = int(min(distance * px_per_m, R - 2))
                cv2.circle(tile, (int(cx), int(cy)), r, colour, 1, cv2.LINE_AA)
                _label(tile, "BEARING N/A", (int(cx - 34), int(cy - r - 6)),
                       colour)
            else:
                cv2.circle(tile, (int(cx), int(cy)), int(R * 0.5), colour, 1,
                           cv2.LINE_AA)
                _label(tile, "BEARING N/A", (int(cx - 34), int(cy - 4)), colour)
            return

        # ── Bearing wedge, width = bearing uncertainty ──
        conf = max(0.0, min(1.0, acoustic.bearing_confidence))
        half_width = 8.0 + (1.0 - conf) * 22.0
        for offset in (-half_width, half_width):
            x, y = _polar_to_xy(cx, cy, bearing + offset, R)
            cv2.line(tile, (int(cx), int(cy)), (x, y), COL_SPOKE, 1,
                     cv2.LINE_AA)

        # ── Case 2: bearing but no distance — dashed radial ──
        if distance is None:
            self._dashed_radial(tile, bearing, colour)
            x, y = _polar_to_xy(cx, cy, bearing, R * 0.55)
            cv2.circle(tile, (x, y), 5, colour, 1, cv2.LINE_AA)
            _label(tile, f"{bearing:.0f}° RANGE N/A", (8, self.size - 8),
                   colour)
        else:
            # ── Case 3: both known — blip plus range-uncertainty bar ──
            r = min(distance * px_per_m, R - 2)
            x, y = _polar_to_xy(cx, cy, bearing, r)

            if acoustic.distance_lo_m is not None \
                    and acoustic.distance_hi_m is not None:
                r_lo = min(acoustic.distance_lo_m * px_per_m, R - 2)
                r_hi = min(acoustic.distance_hi_m * px_per_m, R - 2)
                p1 = _polar_to_xy(cx, cy, bearing, r_lo)
                p2 = _polar_to_xy(cx, cy, bearing, r_hi)
                cv2.line(tile, p1, p2, colour, 1, cv2.LINE_AA)
                for p in (p1, p2):
                    cv2.circle(tile, p, 2, colour, -1, cv2.LINE_AA)

            cv2.circle(tile, (x, y), 5, colour, -1, cv2.LINE_AA)
            cv2.circle(tile, (x, y), 8, colour, 1, cv2.LINE_AA)

        # ── Mirror ghost for a 2-mic ambiguity ──
        if acoustic.bearing_ambiguous:
            mirror = (180.0 - bearing) % 360.0
            mr = (min(distance * px_per_m, R - 2) if distance is not None
                  else R * 0.55)
            mx, my = _polar_to_xy(cx, cy, mirror, mr)
            cv2.circle(tile, (mx, my), 5, colour, 1, cv2.LINE_AA)
            cv2.line(tile, (mx - 3, my - 3), (mx + 3, my + 3), colour, 1,
                     cv2.LINE_AA)

    def _dashed_radial(self, tile: np.ndarray, bearing: float,
                       colour: Tuple[int, int, int]) -> None:
        step = 8
        r = self.max_radius * 0.15
        while r < self.max_radius - 2:
            p1 = _polar_to_xy(self.cx, self.cy, bearing, r)
            p2 = _polar_to_xy(self.cx, self.cy, bearing,
                              min(r + step * 0.6, self.max_radius - 2))
            cv2.line(tile, p1, p2, colour, 1, cv2.LINE_AA)
            r += step

    def _draw_header(self, tile: np.ndarray, target: FusedTarget,
                     scale_m: float, stale: bool) -> None:
        """Scale, freshness and approach state, top-right of the tile."""
        right = self.size - 6

        scale_txt = f"R {scale_m:.0f}m"
        (tw, _), _ = cv2.getTextSize(ascii_safe(scale_txt), FONT, 0.34, 1)
        cv2.putText(tile, ascii_safe(scale_txt), (right - tw, 14), FONT, 0.34,
                    COL_RING_TEXT, 1, cv2.LINE_AA)

        if not target.acoustic_online:
            status, col = "OFFLINE", COL_ALARM
        elif stale and target.acoustic is not None:
            # ⚠️ Not gated on target.state.has_target. That gate meant a
            # 30-second-old reading was still headed "CONTACT" once the
            # target had been declared lost — the state having moved on is
            # precisely when the operator needs to be told the reading is
            # old, not a reason to stop telling them.
            status, col = "STALE", COL_STALE
        elif target.acoustic is not None and target.acoustic.detected:
            status, col = "CONTACT", COL_ACOUSTIC
        else:
            status, col = "LISTENING", COL_DIM
        (tw, _), _ = cv2.getTextSize(ascii_safe(status), FONT, 0.34, 1)
        cv2.putText(tile, ascii_safe(status), (right - tw, 28), FONT, 0.34, col, 1,
                    cv2.LINE_AA)

        # Approach state — derived, and labelled UNKNOWN when it is.
        if target.state.has_target:
            k = target.kinematics
            if k.approach is Approach.APPROACHING:
                txt, col = f"CLOSING {abs(k.speed_mps or 0):.0f} m/s", COL_ALARM
            elif k.approach is Approach.RECEDING:
                txt, col = f"OPENING {abs(k.speed_mps or 0):.0f} m/s", COL_DIM
            elif k.approach is Approach.STEADY:
                txt, col = "STEADY", COL_DIM
            else:
                txt, col = "CLOSURE UNKNOWN", COL_DIM
            (tw, _), _ = cv2.getTextSize(ascii_safe(txt), FONT, 0.32, 1)
            cv2.putText(tile, ascii_safe(txt), (right - tw, 42), FONT, 0.32, col, 1,
                        cv2.LINE_AA)


def _label(img: np.ndarray, text: str, org: Tuple[int, int],
           colour: Tuple[int, int, int]) -> None:
    cv2.putText(img, ascii_safe(text), org, FONT, 0.32, colour, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  Compositing
# ═══════════════════════════════════════════════════════════════

def composite(frame: np.ndarray, tile: np.ndarray, x: int, y: int,
              alpha: float = 0.88) -> None:
    """
    Blend the radar tile into the frame IN PLACE.

    Clipped to the frame bounds so a small window or an unexpected frame
    size can never raise. alpha < 1 keeps a hint of the underlying image
    visible so the widget reads as an overlay rather than a hole.
    """
    fh, fw = frame.shape[:2]
    th, tw = tile.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + tw), min(fh, y + th)
    if x1 <= x0 or y1 <= y0:
        return
    sub_t = tile[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    if alpha >= 1.0:
        roi[:] = sub_t
    else:
        cv2.addWeighted(sub_t, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


if __name__ == "__main__":
    # Renders the radar in every meaningful state to PNG files so the
    # drawing can be inspected without any hardware.
    from pathlib import Path

    from camera_cue import BearingProjector
    from fusion_config import load as load_config
    from sensor_fusion import SensorFusion
    from target_state import (AcousticObservation, SubsystemHealth,
                              SubsystemState, now)

    cfg = load_config()
    radar = RadarOverlay(320, cfg.ui.radar_scales_m, cfg.ui.radar_trail_s)
    proj = BearingProjector(cfg.geometry, 640)
    healthy = SubsystemHealth(SubsystemState.ONLINE)

    cases = {
        "01_searching": AcousticObservation(engine_state="LISTEN"),
        "02_bearing_only": AcousticObservation(
            engine_state="ALARM", p_smoothed=0.9, bearing_deg=142.0,
            bearing_confidence=0.75, distance_reason="not calibrated"),
        "03_bearing_and_range": AcousticObservation(
            engine_state="ALARM", p_smoothed=0.93, bearing_deg=142.0,
            bearing_confidence=0.85, distance_m=87.0, distance_lo_m=52.0,
            distance_hi_m=145.0),
        "04_range_only": AcousticObservation(
            engine_state="ALARM", p_smoothed=0.9, bearing_deg=None,
            distance_m=60.0, distance_lo_m=40.0, distance_hi_m=95.0),
        "05_ambiguous": AcousticObservation(
            engine_state="ALARM", p_smoothed=0.88, bearing_deg=35.0,
            bearing_confidence=0.4, bearing_ambiguous=True, distance_m=120.0,
            distance_lo_m=70.0, distance_hi_m=200.0),
    }

    out_dir = Path(__file__).with_name("_radar_preview")
    out_dir.mkdir(exist_ok=True)

    print("=" * 66)
    print("radar_overlay.py — rendering every display case")
    print("=" * 66)

    for name, obs in cases.items():
        fusion = SensorFusion(cfg, proj)
        for i in range(12):
            snap = fusion.update(
                AcousticObservation(**{**obs.__dict__, "seq": i,
                                       "timestamp": now()}),
                None, healthy, healthy)
        tile = radar.render(snap, 0.033)
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), tile)
        print(f"   {name:<22} state={snap.state.value:<20} -> {path.name}")

    # Composite onto a fake camera frame to check clipping
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    composite(frame, tile, 640 - 320 - 12, 12)
    composite(frame, tile, 600, 400)          # deliberately overflowing
    cv2.imwrite(str(out_dir / "06_composite.png"), frame)
    print(f"   composite + clipping   -> 06_composite.png")
    print(f"\n   Preview images in: {out_dir}")

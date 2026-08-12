#!/usr/bin/env python3
"""
hud.py — The single unified window.

Layout (one window, camera image is the primary element):

    ┌────────────────────────────────────────────────────────────┐
    │ ● SYSTEM STATE      TARGET #n        PRIORITY: CAMERA      │  status bar
    ├────────────────────────────────────────────────────────────┤
    │                                                            │
    │                      CAMERA IMAGE                          │
    │                   with YOLO bounding boxes                 │
    │                   and the acoustic bearing cue             │
    │                                          ┌──────────────┐  │
    │                                          │   ACOUSTIC   │  │
    │                                          │    RADAR     │  │
    │                                          └──────────────┘  │
    ├────────────────────────────────────────────────────────────┤
    │ ACOUSTIC ● CONF 91%  BRG 142°  RNG ~87 m (52-145)          │  sensor bar
    │ VISUAL   ● CONF 93%  BOX 62px  RNG 5.3 m   CAM: FAR/IMX477 │
    └────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
RULES THIS RENDERER FOLLOWS
═══════════════════════════════════════════════════════════════════

1. Any value the sensors cannot supply is printed as N/A or UNKNOWN. There
   is no placeholder number anywhere in this file.
2. Stale readings are shown greyed and tagged STALE — displayed, but never
   dressed up as current.
3. Acoustic distances are prefixed "~" and carry their interval, because
   they are level-based estimates with ±40–60% real accuracy. Visual
   distances (pinhole, from a tight box) are shown without a tilde but only
   when the focal length for that camera is calibrated.
4. The camera image is never obscured in the middle: all chrome is on the
   edges, and the radar sits in a corner.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from camera_cue import BearingProjector
from fusion_config import StationConfig
from radar_overlay import RadarOverlay, ascii_safe, composite
from sensor_fusion import FusedTarget, Priority, SystemState
from target_state import Approach, Freshness, SubsystemState

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Palette (BGR) ──
C_BAR = (26, 22, 18)
C_EDGE = (70, 60, 45)
C_TEXT = (225, 230, 228)
C_DIM = (135, 145, 140)
C_OK = (100, 220, 130)
C_WARN = (60, 190, 250)
C_ALARM = (60, 60, 255)
C_OFF = (90, 90, 95)
C_ACOUSTIC = (70, 200, 255)
C_VISUAL = (90, 235, 120)

STATUS_H = 30
SENSOR_H = 52


def _state_colour(state: SystemState) -> Tuple[int, int, int]:
    return {
        SystemState.SEARCHING: C_DIM,
        SystemState.ACOUSTIC_DETECTED: C_WARN,
        SystemState.ACOUSTIC_TRACKING: C_ACOUSTIC,
        SystemState.VISUAL_ACQUISITION: C_WARN,
        SystemState.VISUAL_TRACKING: C_VISUAL,
        SystemState.VISUAL_LOST_ACOUSTIC_TRACKING: C_WARN,
        SystemState.TARGET_LOST: C_ALARM,
    }[state]


def _health_colour(state: SubsystemState) -> Tuple[int, int, int]:
    return {
        SubsystemState.ONLINE: C_OK,
        SubsystemState.DEGRADED: C_WARN,
        SubsystemState.STARTING: C_DIM,
        SubsystemState.OFFLINE: C_ALARM,
        SubsystemState.DISABLED: C_OFF,
    }[state]


def _text(img, s, org, scale=0.42, colour=C_TEXT, thick=1) -> int:
    """
    Draw text and return its pixel width so callers can lay out a row.

    Everything goes through ascii_safe(): OpenCV's Hershey fonts have no
    glyph for '°' or '±' and silently render them as '?', which turned
    "BRG 142°" into "BRG 142??" on screen.
    """
    s = ascii_safe(s)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    return w


def _dot(img, centre, colour, r=4) -> None:
    cv2.circle(img, centre, r, colour, -1, cv2.LINE_AA)


class HUD:
    """
    Composes one displayable frame from the fused target state.

    Owns the radar widget and all drawing. Holds no sensor state of its own,
    so it can render whatever the fusion layer last produced — including
    "no camera at all", which it renders as a placeholder canvas rather than
    failing.
    """

    def __init__(self, config: StationConfig, projector: BearingProjector):
        self.config = config
        self.projector = projector
        self.radar: Optional[RadarOverlay] = None
        self._radar_size = 0
        self._last_render = 0.0
        self.show_help = False

    # ── Radar sizing ───────────────────────────────────────────

    def _ensure_radar(self, frame_w: int, frame_h: int) -> RadarOverlay:
        size = int(min(frame_w, frame_h) * self.config.ui.radar_size_frac)
        size = max(140, min(size, 420))
        if self.radar is None or size != self._radar_size:
            self.radar = RadarOverlay(size, self.config.ui.radar_scales_m,
                                      self.config.ui.radar_trail_s,
                                      self.config.ui.radar_sweep_deg_per_s)
            self._radar_size = size
        return self.radar

    # ── Main entry point ───────────────────────────────────────

    def render(self, target: FusedTarget, dt_s: float,
               fps: float = 0.0) -> np.ndarray:
        base = self._camera_canvas(target)
        h, w = base.shape[:2]

        canvas = np.zeros((h + STATUS_H + SENSOR_H, w, 3), dtype=np.uint8)
        canvas[STATUS_H:STATUS_H + h] = base

        self._draw_boxes(canvas, target, y_offset=STATUS_H)
        self._draw_bearing_cue(canvas, target, y_offset=STATUS_H,
                               height=h, width=w)

        radar = self._ensure_radar(w, h)
        tile = radar.render(target, dt_s)
        margin = self.config.ui.radar_margin_px
        composite(canvas, tile, w - tile.shape[1] - margin,
                  STATUS_H + margin)

        self._draw_status_bar(canvas, target, w, fps)
        self._draw_sensor_bar(canvas, target, w, h + STATUS_H)

        if self.show_help:
            self._draw_help(canvas, w)
        return canvas

    # ── Camera area ────────────────────────────────────────────

    def _camera_canvas(self, target: FusedTarget) -> np.ndarray:
        """
        The camera image, or an explicit placeholder when there is none.

        A missing camera must not stop the station: the acoustic radar and
        every readout keep working on this placeholder.
        """
        visual = target.visual
        if visual is not None and visual.frame is not None:
            frame = visual.frame
            scale = self.config.ui.display_scale
            if scale and abs(scale - 1.0) > 1e-3:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LINEAR)
            else:
                # The UI draws chrome on top, so it must never write into a
                # buffer the camera worker may still be holding.
                frame = frame.copy()
            return frame

        w = int(self.config.visual.frame_width * self.config.ui.display_scale)
        h = int(self.config.visual.frame_height * self.config.ui.display_scale)
        canvas = np.full((h, w, 3), 22, dtype=np.uint8)

        detail = target.visual_health.detail or "no frames"
        msg = "CAMERA OFFLINE"
        (tw, _), _ = cv2.getTextSize(ascii_safe(msg), FONT, 0.9, 2)
        cv2.putText(canvas, ascii_safe(msg), ((w - tw) // 2, h // 2 - 10), FONT, 0.9,
                    C_ALARM, 2, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(ascii_safe(detail), FONT, 0.45, 1)
        cv2.putText(canvas, ascii_safe(detail), ((w - tw2) // 2, h // 2 + 20), FONT, 0.45,
                    C_DIM, 1, cv2.LINE_AA)
        note = "acoustic subsystem continues to operate"
        (tw3, _), _ = cv2.getTextSize(ascii_safe(note), FONT, 0.42, 1)
        cv2.putText(canvas, ascii_safe(note), ((w - tw3) // 2, h // 2 + 46), FONT, 0.42,
                    C_DIM, 1, cv2.LINE_AA)
        return canvas

    def _draw_boxes(self, canvas: np.ndarray, target: FusedTarget,
                    y_offset: int) -> None:
        """Bounding boxes for every live track, scaled to the display."""
        visual = target.visual
        if visual is None or visual.frame is None:
            return
        scale = self.config.ui.display_scale
        primary = target.visual_track
        primary_id = primary.track_id if primary is not None else None

        # A track shown during a visual dropout is drawn dashed and labelled
        # LAST KNOWN — never as a live detection.
        coasting = (target.state is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING
                    or target.visual_freshness is not Freshness.FRESH)

        # The fusion layer keeps the last confirmed track alive across a
        # short visual dropout (STATE 7: "use last known visual position").
        # That track is NOT in visual.tracks any more — the camera no longer
        # reports it — so it has to be appended explicitly, or the operator
        # would see the box simply vanish at the moment it matters most.
        draw_list = list(visual.tracks)
        if primary is not None and all(t.track_id != primary_id
                                       for t in draw_list):
            draw_list.append(primary)

        for track in draw_list:
            x, y, w, h = track.bbox
            x1 = int(round(x * scale))
            y1 = int(round(y * scale)) + y_offset
            x2 = int(round((x + w) * scale))
            y2 = int(round((y + h) * scale)) + y_offset

            ch, cw = canvas.shape[:2]
            x1 = max(0, min(cw - 1, x1)); x2 = max(0, min(cw - 1, x2))
            y1 = max(y_offset, min(ch - 1, y1))
            y2 = max(y_offset, min(ch - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            is_primary = track.track_id == primary_id
            if coasting and is_primary:
                colour = C_WARN
            elif track.state == "Confirmed":
                colour = C_VISUAL if is_primary else (120, 190, 120)
            elif track.state == "Tentative":
                colour = (0, 200, 255)
            else:
                colour = (0, 0, 220)

            thickness = 2 if is_primary else 1
            if coasting and is_primary:
                self._dashed_rect(canvas, (x1, y1), (x2, y2), colour)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness,
                              cv2.LINE_AA)

            parts = [f"#{track.track_id}"]
            if coasting and is_primary:
                parts.append("LAST KNOWN")
            elif track.detection_score is not None:
                parts.append(f"{track.detection_score:.0%}")
            else:
                parts.append("coasting")
            if track.distance_m is not None:
                parts.append(f"{track.distance_m:.1f}m")
            label = " ".join(parts)

            ly = max(y_offset + 12, y1 - 6)
            (lw, lh), _ = cv2.getTextSize(ascii_safe(label), FONT, 0.4, 1)
            cv2.rectangle(canvas, (x1, ly - lh - 3), (x1 + lw + 4, ly + 2),
                          (20, 20, 20), -1)
            _text(canvas, label, (x1 + 2, ly), 0.4, colour)

    @staticmethod
    def _dashed_rect(img, p1, p2, colour, dash=6) -> None:
        x1, y1 = p1
        x2, y2 = p2
        for x in range(x1, x2, dash * 2):
            cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, 1)
            cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, 1)
        for y in range(y1, y2, dash * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, 1)
            cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, 1)

    def _draw_bearing_cue(self, canvas: np.ndarray, target: FusedTarget,
                          y_offset: int, height: int, width: int) -> None:
        """
        Where the acoustic bearing says to look, IF that is calibrated.

        Drawn as a vertical band spanning the full image height, never as a
        point: the array measures azimuth only and has no elevation, so any
        vertical position would be fabricated.
        """
        cue = target.bearing_cue
        if not cue.available:
            return
        scale = self.config.ui.display_scale

        if not cue.in_view:
            # Off-screen: an arrow at the correct edge, nothing more.
            side = cue.off_screen_side
            x = 18 if side < 0 else width - 18
            y = y_offset + height // 2
            tip = (x - 14, y) if side < 0 else (x + 14, y)
            cv2.arrowedLine(canvas, (x, y), tip, C_ACOUSTIC, 2,
                            cv2.LINE_AA, tipLength=0.5)
            txt = f"{abs(cue.rel_bearing_deg or 0):.0f}° off"
            _text(canvas, txt, (x - 22 if side < 0 else x - 50, y - 18),
                  0.38, C_ACOUSTIC)
            return

        cx = int(round((cue.x_px or 0.0) * scale))
        half = int(round((cue.half_width_px or 0.0) * scale))
        top, bot = y_offset, y_offset + height

        # Uncertainty band, drawn as a translucent fill.
        if half > 0:
            x0 = max(0, cx - half)
            x1 = min(width, cx + half)
            if x1 > x0:
                roi = canvas[top:bot, x0:x1]
                tint = np.full_like(roi, (40, 90, 110))
                cv2.addWeighted(tint, 0.18, roi, 0.82, 0.0, dst=roi)

        cv2.line(canvas, (cx, top), (cx, bot), C_ACOUSTIC, 1, cv2.LINE_AA)

        label = "ACOUSTIC BEARING"
        if cue.uncertainty_exceeds_fov:
            # Honest caveat: a ±15-45° DOA cannot localise inside a 28° lens.
            label += " (COARSE)"
        _text(canvas, label, (max(2, cx - 60), top + 16), 0.36, C_ACOUSTIC)

    # ── Status bar ─────────────────────────────────────────────

    def _draw_status_bar(self, canvas: np.ndarray, target: FusedTarget,
                         width: int, fps: float) -> None:
        cv2.rectangle(canvas, (0, 0), (width, STATUS_H), C_BAR, -1)
        cv2.line(canvas, (0, STATUS_H), (width, STATUS_H), C_EDGE, 1)

        colour = _state_colour(target.state)
        _dot(canvas, (14, STATUS_H // 2), colour, 5)
        x = 26
        x += _text(canvas, target.state.label, (x, 20), 0.5, colour, 1) + 18

        if target.state.has_target:
            x += _text(canvas, f"TARGET #{target.target_id}", (x, 20), 0.44,
                       C_TEXT) + 18

        prio = target.priority
        prio_colour = {Priority.CAMERA: C_VISUAL,
                       Priority.ACOUSTIC: C_ACOUSTIC,
                       Priority.NONE: C_DIM}[prio]
        _text(canvas, f"PRIORITY: {prio.value}", (x, 20), 0.44, prio_colour)

        right = width - 8
        # Timing lives here, next to the FPS counter, rather than in the
        # sensor bar: it is diagnostic information about the pipeline, not
        # a measurement of the target.
        fps_txt = f"{fps:.0f} FPS"
        if target.visual is not None and target.visual.inference_ms is not None:
            fps_txt = f"YOLO {target.visual.inference_ms:.0f}ms   " + fps_txt
        (tw, _), _ = cv2.getTextSize(ascii_safe(fps_txt), FONT, 0.4, 1)
        _text(canvas, fps_txt, (right - tw, 20), 0.4, C_DIM)

        # Sensor availability lamps
        lamp_x = right - tw - 100
        for label, health in (("MIC", target.acoustic_health),
                              ("CAM", target.visual_health)):
            c = _health_colour(health.state)
            _dot(canvas, (lamp_x, STATUS_H // 2), c, 4)
            _text(canvas, label, (lamp_x + 8, 19), 0.36, c)
            lamp_x += 46

    # ── Sensor bar ─────────────────────────────────────────────

    def _draw_sensor_bar(self, canvas: np.ndarray, target: FusedTarget,
                         width: int, top: int) -> None:
        cv2.rectangle(canvas, (0, top), (width, top + SENSOR_H), C_BAR, -1)
        cv2.line(canvas, (0, top), (width, top), C_EDGE, 1)

        acoustic = target.acoustic
        row1 = top + 20
        row2 = top + 40

        # ── Acoustic row ──
        a_health = target.acoustic_health
        if not a_health.ok:
            _dot(canvas, (14, row1 - 4), C_ALARM, 4)
            _text(canvas, "ACOUSTIC", (26, row1), 0.42, C_DIM)
            _text(canvas, f"OFFLINE - {a_health.detail or a_health.error or ''}",
                  (110, row1), 0.42, C_ALARM)
        else:
            fresh = target.acoustic_freshness
            if acoustic is None:
                dot_c = C_DIM
            elif fresh is not Freshness.FRESH and target.state.has_target:
                dot_c = C_OFF
            elif acoustic.confirmed:
                dot_c = C_ALARM
            elif acoustic.detected:
                dot_c = C_ACOUSTIC
            else:
                dot_c = C_OK
            _dot(canvas, (14, row1 - 4), dot_c, 4)

            x = 26
            x += _text(canvas, "ACOUSTIC", (x, row1), 0.42, C_DIM) + 10

            if acoustic is None:
                _text(canvas, "starting...", (x, row1), 0.42, C_DIM)
            else:
                label = {"ALARM": "DRONE CONFIRMED", "TRACK": "POSSIBLE DRONE",
                         "LISTEN": "LISTENING", "SLEEP": "SILENT"}.get(
                             acoustic.engine_state, acoustic.engine_state)
                x += _text(canvas, label, (x, row1), 0.42, dot_c) + 14
                x += _text(canvas, f"CONF {acoustic.p_smoothed:.0%}",
                           (x, row1), 0.42, C_TEXT) + 14
                x += _text(canvas, f"BRG {acoustic.bearing_text()}",
                           (x, row1), 0.42, C_TEXT) + 14
                x += _text(canvas, f"RNG {acoustic.distance_text()}",
                           (x, row1), 0.42, C_TEXT) + 14

                k = target.kinematics
                if target.state.has_target:
                    if k.approach is Approach.UNKNOWN:
                        x += _text(canvas, "CLOSURE UNKNOWN", (x, row1), 0.42,
                                   C_DIM) + 14
                    else:
                        kc = (C_ALARM if k.approach is Approach.APPROACHING
                              else C_DIM)
                        x += _text(canvas, k.text(), (x, row1), 0.42, kc) + 14

                if fresh is not Freshness.FRESH and target.state.has_target:
                    age = target.acoustic_age_s or 0.0
                    _text(canvas, f"[{fresh.value} {age:.1f}s]", (x, row1),
                          0.42, C_OFF)

        # ── Visual row ──
        v_health = target.visual_health
        visual = target.visual
        track = target.visual_track

        if not v_health.ok:
            _dot(canvas, (14, row2 - 4), C_ALARM, 4)
            _text(canvas, "VISUAL", (26, row2), 0.42, C_DIM)
            _text(canvas, f"OFFLINE - {v_health.detail or v_health.error or ''}",
                  (110, row2), 0.42, C_ALARM)
            return

        dot_c = C_VISUAL if track is not None else C_OK
        if target.visual_freshness is not Freshness.FRESH and visual is not None:
            dot_c = C_OFF
        _dot(canvas, (14, row2 - 4), dot_c, 4)

        x = 26
        x += _text(canvas, "VISUAL", (x, row2), 0.42, C_DIM) + 10

        if track is None:
            x += _text(canvas, "NO TARGET", (x, row2), 0.42, C_DIM) + 14
        else:
            status = ("LAST KNOWN"
                      if target.state is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING
                      else "TRACKING")
            x += _text(canvas, status, (x, row2), 0.42, dot_c) + 14
            x += _text(canvas, f"YOLO {target.visual_confidence_text()}",
                       (x, row2), 0.42, C_TEXT) + 14
            x += _text(canvas, f"BOX {track.bbox[2]:.0f}px", (x, row2), 0.42,
                       C_TEXT) + 14
            dist = (f"{track.distance_m:.1f} m" if track.distance_m is not None
                    else "N/A")
            x += _text(canvas, f"RNG {dist}", (x, row2), 0.42, C_TEXT) + 14

        if visual is not None:
            cam_txt = f"CAM: {visual.camera_name}"
            (tw, _), _ = cv2.getTextSize(ascii_safe(cam_txt), FONT, 0.42, 1)
            _text(canvas, cam_txt, (width - tw - 8, row2), 0.42, C_WARN)

            # The bearing→pixel mapping is deliberately disabled until the
            # mount geometry is measured. That refusal must be visible, or
            # an operator would assume the system simply never has a cue.
            if not self.projector.is_calibrated(visual.active_camera):
                note = "BEARING->VIEW: NOT CALIBRATED"
                (tw3, _), _ = cv2.getTextSize(ascii_safe(note), FONT, 0.36, 1)
                _text(canvas, note, (width - tw3 - 8, row1), 0.36, C_OFF)

    # ── Help ───────────────────────────────────────────────────

    def _draw_help(self, canvas: np.ndarray, width: int) -> None:
        lines = [
            "q / ESC   quit",
            "h         toggle this help",
            "f         freeze camera switching (for calibration)",
            "c         print focal-length calibration for the active camera",
            "r         reset the fusion state machine",
            "",
            "Colour key:",
            "  green   visually confirmed target (camera primary)",
            "  amber   acoustic contact (microphone primary)",
            "  red     acoustic alarm confirmed",
            "  grey    stale reading - shown, not trusted",
        ]
        pad = 12
        w = 340
        h = len(lines) * 18 + pad * 2
        x0 = (width - w) // 2
        y0 = STATUS_H + 40
        overlay = canvas[y0:y0 + h, x0:x0 + w]
        if overlay.size == 0:
            return
        dark = np.full_like(overlay, (18, 16, 14))
        cv2.addWeighted(dark, 0.9, overlay, 0.1, 0.0, dst=overlay)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), C_EDGE, 1)
        y = y0 + pad + 8
        for line in lines:
            _text(canvas, line, (x0 + pad, y), 0.4,
                  C_TEXT if not line.startswith(" ") else C_DIM)
            y += 18


if __name__ == "__main__":
    # Renders the HUD for each system state to PNG, with no hardware.
    from pathlib import Path

    from fusion_config import load as load_config
    from sensor_fusion import SensorFusion
    from target_state import (AcousticObservation, SubsystemHealth,
                              SubsystemState, VisualObservation, VisualTrack,
                              now)

    cfg = load_config()
    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    hud = HUD(cfg, proj)
    healthy = SubsystemHealth(SubsystemState.ONLINE)
    offline = SubsystemHealth(SubsystemState.OFFLINE, detail="no device")

    rng = np.random.default_rng(3)

    def fake_frame() -> np.ndarray:
        f = (rng.normal(48, 9, (480, 640, 3))).clip(0, 255).astype(np.uint8)
        cv2.rectangle(f, (0, 300), (640, 480), (60, 70, 60), -1)  # ground
        return f

    def vis(tracks=(), **kw):
        return VisualObservation(
            frame=fake_frame(), frame_width=640, frame_height=480,
            tracks=tracks, active_camera=0, camera_name="FAR/IMX477",
            inference_ms=24.0, timestamp=now(), **kw)

    track = VisualTrack(track_id=1, bbox=(300.0, 190.0, 62.0, 44.0),
                        state="Confirmed", quality_score=0.4,
                        detection_score=0.93, detection_age_s=0.0,
                        distance_m=5.1)

    out = Path(__file__).with_name("_hud_preview")
    out.mkdir(exist_ok=True)

    print("=" * 66)
    print("hud.py — rendering the unified window in each state")
    print("=" * 66)

    scenarios = []

    # 1. Searching
    f = SensorFusion(cfg, proj)
    s = f.update(AcousticObservation(engine_state="LISTEN", seq=1), vis(),
                 healthy, healthy)
    scenarios.append(("01_searching", s))

    # 2. Acoustic tracking at 200 m, no range calibration
    f = SensorFusion(cfg, proj)
    for i in range(10):
        s = f.update(AcousticObservation(
            engine_state="ALARM", p_smoothed=0.91, threshold=0.775,
            bearing_deg=142.0, bearing_confidence=0.8,
            distance_reason="not calibrated", seq=i, timestamp=now()),
            vis(), healthy, healthy)
    scenarios.append(("02_acoustic_no_range", s))

    # 3. Acoustic tracking with range
    f = SensorFusion(cfg, proj)
    for i in range(10):
        s = f.update(AcousticObservation(
            engine_state="ALARM", p_smoothed=0.87, threshold=0.775,
            bearing_deg=142.0, bearing_confidence=0.85, distance_m=87.0,
            distance_lo_m=52.0, distance_hi_m=145.0, seq=i, timestamp=now()),
            vis(), healthy, healthy)
    scenarios.append(("03_acoustic_tracking", s))

    # 4. Visual tracking, camera primary
    f = SensorFusion(cfg, proj)
    for i in range(12):
        s = f.update(AcousticObservation(
            engine_state="ALARM", p_smoothed=0.9, threshold=0.775,
            bearing_deg=138.0, bearing_confidence=0.8, distance_m=12.0,
            distance_lo_m=8.0, distance_hi_m=19.0, seq=i, timestamp=now()),
            vis((track,)), healthy, healthy)
    scenarios.append(("04_visual_tracking", s))

    # 5. Camera offline, acoustic continues
    f = SensorFusion(cfg, proj)
    for i in range(10):
        s = f.update(AcousticObservation(
            engine_state="ALARM", p_smoothed=0.88, threshold=0.775,
            bearing_deg=210.0, bearing_confidence=0.6, distance_m=140.0,
            distance_lo_m=85.0, distance_hi_m=230.0, seq=i, timestamp=now()),
            None, healthy, offline)
    scenarios.append(("05_camera_offline", s))

    # 6. Microphone offline, camera continues
    f = SensorFusion(cfg, proj)
    for i in range(12):
        s = f.update(None, vis((track,)), offline, healthy)
    scenarios.append(("06_mic_offline", s))

    for name, snap in scenarios:
        img = hud.render(snap, 0.033, fps=28.4)
        cv2.imwrite(str(out / f"{name}.png"), img)
        print(f"   {name:<24} state={snap.state.value:<26} "
              f"priority={snap.priority.value}")

    # Help overlay
    hud.show_help = True
    img = hud.render(scenarios[3][1], 0.033, fps=28.4)
    cv2.imwrite(str(out / "07_help.png"), img)
    print(f"   07_help                  (keyboard overlay)")
    print(f"\n   Preview images in: {out}")

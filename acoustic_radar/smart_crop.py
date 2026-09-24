#!/usr/bin/env python3
"""
smart_crop.py — Native-resolution windows for small (far) targets.

═══════════════════════════════════════════════════════════════════
WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════

The detector's input is 640x640. The published camera frame is 640x480, and
the sensor mode produces it by DOWNSCALING a larger image (1332x990 or
2028x1520 on this station's IMX477). Downscaling throws away exactly the
pixels a distant drone is made of: a target 20 px wide in the sensor mode is
about 10 px wide in the published frame.

A 640x640 window cut 1:1 from the native frame shows that target at its full
native size. A window covers only part of the field of view, so:

  • the full published frame is still inferred on every detector run —
    near (large) targets are found exactly as before;
  • one or more native windows are inferred on top of it, placed on small
    confirmed tracks first, the remainder sweeping the frame;
  • every window's boxes are mapped into the published frame and merged with
    the full-frame boxes, so the tracker sees ONE detection list in ONE pixel
    space — the same space, and the same focal length, as before.

═══════════════════════════════════════════════════════════════════
WHAT IS DELIBERATELY NOT HERE
═══════════════════════════════════════════════════════════════════

No detection range and no frame rate is claimed anywhere in this module. The
geometric gain is native_width / published_width (about 2.06x on the 1332x990
mode). Whether the lens actually resolves that detail, and what the extra
Hailo calls cost, can only be measured on the station.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Size = Tuple[int, int]                     # (width, height)
Origin = Tuple[int, int]                   # top-left of a window, native px
TrackBox = Tuple[float, float, float, float]   # (cx, cy, w, h), published px


# ═══════════════════════════════════════════════════════════════
#  Stream geometry
# ═══════════════════════════════════════════════════════════════

def native_stream_size(sensor_size: Size, published_size: Size) -> Size:
    """
    The largest even size inside the sensor mode that has the PUBLISHED
    stream's aspect ratio.

    ⚠️ WHY THE ASPECT RATIO MUST MATCH. The native and the published stream
    are two outputs of ONE exposure, and a box is moved from one to the
    other by a plain per-axis scale. That is only true if both streams show
    exactly the same region of the sensor. Where the two aspect ratios
    differ, the ISP has to reconcile them — by cropping one stream or by
    stretching both — and which it does depends on the libcamera version,
    which this code cannot see. Giving both streams the same aspect ratio
    removes the question: 1332x990 (1.345) becomes 1320x990 (4:3, like
    640x480), at a cost of 0.9 % of the native width.
    """
    sw, sh = int(sensor_size[0]), int(sensor_size[1])
    pw, ph = int(published_size[0]), int(published_size[1])
    if sw * ph > sh * pw:            # sensor mode is wider than published
        h = float(sh)
        w = h * pw / ph
    else:
        w = float(sw)
        h = w * ph / pw
    return (int(w) // 2 * 2, int(h) // 2 * 2)


def _spread(length: int, crop: int) -> List[int]:
    """Evenly spread window starts covering [0, length) edge to edge."""
    n = max(1, math.ceil(length / crop))
    if n == 1:
        return [(length - crop) // 2]
    step = (length - crop) / (n - 1)
    return [int(round(i * step)) for i in range(n)]


def tile_origins(native_size: Size, crop: int) -> List[Origin]:
    """
    Top-left corners of windows that together cover the whole native frame.

    Columns (and rows) are spread evenly from edge to edge, so neighbouring
    windows OVERLAP. With n = ceil(W / crop) columns the step is
    (W - crop) / (n - 1) <= crop, which guarantees full coverage. The
    overlap is what allows a window to discard a box its own edge cut in
    half: the neighbouring window sees that target whole
    (see boxes_from_crop).

        1320 x 990,  crop 640  ->  3 x 2 =  6 windows, overlap 300 / 290 px
        2026 x 1520, crop 640  ->  4 x 3 = 12 windows, overlap 178 / 200 px

    Empty when the frame is smaller than one window in either direction.
    """
    w, h = int(native_size[0]), int(native_size[1])
    if crop <= 0 or w < crop or h < crop:
        return []
    xs = _spread(w, crop)
    ys = _spread(h, crop)
    return [(x, y) for y in ys for x in xs]


def centred_origin(centre: Tuple[float, float], native_size: Size,
                   crop: int) -> Origin:
    """A window of side `crop` centred on `centre` (native px), kept inside
    the frame."""
    w, h = int(native_size[0]), int(native_size[1])
    x = int(round(float(centre[0]) - crop / 2.0))
    y = int(round(float(centre[1]) - crop / 2.0))
    return (min(max(x, 0), w - crop), min(max(y, 0), h - crop))


# ═══════════════════════════════════════════════════════════════
#  Window placement
# ═══════════════════════════════════════════════════════════════

class CropPlanner:
    """
    Decides where the native windows go on one detector run.

    Priority:
      1. CONFIRMED tracks narrower than `track_below_px` (published px),
         smallest first — the farthest targets need the pixels most. The
         window is centred on the track's current estimate. A small target
         moves slowly across the image (its angular rate is v/R, like its
         size), so it stays well inside a window placed one detector run
         earlier: half a window is 320 native px.
      2. CANDIDATES narrower than `track_below_px` — tracks the tracker has
         not confirmed (yet, or any more), smallest first.
      3. whatever budget remains sweeps the tile grid round-robin, so every
         part of the frame is looked at natively within
         ceil(tiles / sweeping crops per run) detector runs.

    ⚠️ WHY CANDIDATES GET A WINDOW (found on the station, fixed here). The
    first version gave windows to confirmed tracks only. A far target that
    only a native window can see then had to be CONFIRMED by the sweep
    alone — three detections before four consecutive misses. The sweep
    returns to a given window once every `tiles / crops` runs (6 with one
    crop on 1320x990), so in most of the frame the new track was deleted
    between two visits and the target could never be confirmed: in a
    simulation with the real tracker, 67 % of positions in the frame never
    confirmed with one crop. Following the candidate lets the window that
    found it look again on the very next run.
    """

    def __init__(self) -> None:
        self._sweep_index = 0
        self._tiles_key: Optional[Tuple[int, int, int]] = None
        self._tiles: List[Origin] = []

    @property
    def tiles(self) -> List[Origin]:
        return list(self._tiles)

    def plan(self, native_size: Size, scale: Tuple[float, float], crop: int,
             max_crops: int, track_boxes: Sequence[TrackBox],
             track_below_px: float,
             candidate_boxes: Sequence[TrackBox] = ()) -> List[Origin]:
        """
        Window origins (native px) for this run, at most `max_crops`.

        `track_boxes` are CONFIRMED tracks and `candidate_boxes` unconfirmed
        ones, both as (cx, cy, w, h) in published px; `scale` is
        (native_w / published_w, native_h / published_h).
        """
        if max_crops <= 0:
            return []
        key = (int(native_size[0]), int(native_size[1]), int(crop))
        if key != self._tiles_key:
            self._tiles_key = key
            self._tiles = tile_origins(native_size, crop)
            self._sweep_index = 0
        if not self._tiles:
            return []

        sx, sy = float(scale[0]), float(scale[1])
        origins: List[Origin] = []

        def small(boxes):
            return sorted((b for b in boxes
                           if 0.0 < float(b[2]) < float(track_below_px)),
                          key=lambda b: float(b[2]))

        for cx, cy, _w, _h in small(track_boxes) + small(candidate_boxes):
            if len(origins) >= max_crops:
                break
            origin = centred_origin((float(cx) * sx, float(cy) * sy),
                                    native_size, crop)
            if origin not in origins:
                origins.append(origin)

        for _ in range(len(self._tiles)):
            if len(origins) >= max_crops:
                break
            origin = self._tiles[self._sweep_index % len(self._tiles)]
            self._sweep_index += 1
            if origin not in origins:
                origins.append(origin)
        return origins


# ═══════════════════════════════════════════════════════════════
#  Detections: window -> published frame, and merging
# ═══════════════════════════════════════════════════════════════

def boxes_from_crop(boxes: Sequence[np.ndarray], scores: Sequence[float],
                    origin: Origin, crop: int, native_size: Size,
                    scale: Tuple[float, float], edge_px: float = 1.0
                    ) -> Tuple[List[np.ndarray], List[float]]:
    """
    Map one window's detections into the PUBLISHED frame.

    `boxes` are (x, y, w, h) in the window's own pixels — what
    HailoInference.predict_with_scores() returns for a 640x640 input, which
    it clips to the window. `scale` is (native_w / published_w,
    native_h / published_h).

    ⚠️ A BOX TOUCHING AN INNER EDGE OF THE WINDOW IS DISCARDED. The decoder
    clips boxes to the window, so a target the window cuts in half comes back
    as a confident box around half a target: wrong centre and wrong width,
    so a wrong distance and a wrong bearing downstream. The windows overlap
    by hundreds of pixels (tile_origins), so a target small enough to need a
    window is whole in a neighbouring one, and a large one is seen whole by
    the full-frame inference. An edge that is also the edge of the FRAME is
    not "inner": a target there is cut by the sensor, not by this code, and
    is kept exactly as the full-frame path would keep it.
    """
    x0, y0 = int(origin[0]), int(origin[1])
    nw, nh = int(native_size[0]), int(native_size[1])
    sx, sy = float(scale[0]), float(scale[1])
    out_b: List[np.ndarray] = []
    out_s: List[float] = []
    for box, score in zip(boxes, scores):
        bx, by, bw, bh = (float(v) for v in box[:4])
        if x0 > 0 and bx <= edge_px:
            continue
        if y0 > 0 and by <= edge_px:
            continue
        if x0 + crop < nw and bx + bw >= crop - edge_px:
            continue
        if y0 + crop < nh and by + bh >= crop - edge_px:
            continue
        out_b.append(np.array([(x0 + bx) / sx, (y0 + by) / sy,
                               bw / sx, bh / sy], dtype=np.float32))
        out_s.append(float(score))
    return out_b, out_s


def _overlaps(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(IoU, intersection over the SMALLER box) of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0, 0.0
    inter = iw * ih
    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0,
            inter / smaller if smaller > 0 else 0.0)


def merge_detections(groups: Sequence[Tuple[Sequence[np.ndarray],
                                            Sequence[float]]],
                     iou_thresh: float
                     ) -> Tuple[List[np.ndarray], List[float]]:
    """
    One detection list from several inferences on the SAME exposure.

    A target seen both in the full frame and in a window comes back twice,
    at two resolutions. Two boxes for one drone would open two tracks — the
    "one drone, two targets" defect (N2) by a new route.

    Greedy suppression, highest score first. A box is dropped when it
    overlaps an already-kept box by more than the detector's own
    `iou_thresh`, measured as:
      • IoU, for two boxes from the SAME inference (already NMS'd there, so
        in practice this never fires — kept only so the rule is total);
      • intersection over the SMALLER box, for boxes from DIFFERENT
        inferences. The same drone at two resolutions does not give two
        boxes of the same extent: the full-frame box of a small target is
        typically inflated around the tight native one, and plain IoU of a
        box nested inside a larger one can fall below the threshold even
        though both describe one object.

    ⚠️ A SINGLE NON-EMPTY GROUP IS RETURNED UNTOUCHED. Its boxes already went
    through the detector's own NMS, and nothing from another inference can
    duplicate them. In particular, with crop_max_per_frame = 0 the only
    group is the full-frame result, returned exactly as the original
    pipeline would have used it.
    """
    groups = [(list(b), list(s)) for b, s in groups]
    non_empty = [g for g in groups if g[0]]
    if not non_empty:
        return [], []
    if len(non_empty) == 1:
        return non_empty[0][0], non_empty[0][1]

    entries = [(float(score), gi, box)
               for gi, (boxes, scores) in enumerate(non_empty)
               for box, score in zip(boxes, scores)]
    # Stable sort: on equal scores the earlier group (the full frame) wins.
    entries.sort(key=lambda e: -e[0])

    kept: List[Tuple[float, int, np.ndarray]] = []
    for score, gi, box in entries:
        duplicate = False
        for _ks, kgi, kbox in kept:
            iou, ios = _overlaps(box, kbox)
            measure = iou if gi == kgi else ios
            if measure > float(iou_thresh):
                duplicate = True
                break
        if not duplicate:
            kept.append((score, gi, box))
    return [k[2] for k in kept], [k[0] for k in kept]


# ═══════════════════════════════════════════════════════════════
#  Run-time check: do the two streams really line up?
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StreamAlignment:
    """
    Where the published frame's content sits in the downscaled native frame.

    `left` / `right` are (dx, dy) in published px, measured on the left and
    right thirds of the frame: the published patch at (x, y) is found at
    (x + dx, y + dy) in the native frame scaled to the published size.
    (0, 0) = both streams show the same scene at the same place. Equal
    non-zero shifts = an offset; shifts that differ between left and right =
    a scale mismatch.
    """
    verdict: str                               # ALIGNED|MISALIGNED|INCONCLUSIVE
    left: Optional[Tuple[float, float]] = None
    right: Optional[Tuple[float, float]] = None
    detail: str = ""


def _gray(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        # Channel order does not matter for a structural correlation.
        arr = arr.mean(axis=2)
    return arr.astype(np.float32)


def measure_stream_alignment(native, published, tolerance_px: float = 1.0,
                             min_std: float = 2.0, min_response: float = 0.5
                             ) -> StreamAlignment:
    """
    MEASURE, from one exposure, whether the published stream is a pure
    downscale of the native one — the assumption every crop box relies on.

    The native frame is downscaled to the published size, and the shift
    between the two is measured by PHASE CORRELATION, to a fraction of a
    pixel, separately on the left and the right third of the frame (equal
    shifts = an offset, opposite shifts = a scale mismatch).

    ⚠️ WHY PHASE CORRELATION (measured on the station). The first version
    located each patch by normalised cross-correlation and required the peak
    to beat every value more than one pixel away. On the station's real
    outdoor frames the peak sat exactly at (0, 0), but the correlation one
    and two pixels away was still 0.995-0.999 — natural scenes are dominated
    by large smooth structures, which barely decorrelate over a pixel — so
    every measurement came out INCONCLUSIVE although the streams were
    aligned. Phase correlation whitens the spectrum, so the peak is sharp on
    real scenes too. On those same frames it measured shifts of at most
    0.05 px with responses of 0.92-0.99, and a deliberate 2 px shift was
    recovered as 1.94-1.97 px.

    INCONCLUSIVE — never a guess — when:
      • a patch is flat (`min_std`): blank sky, a lens cap. Phase
        correlation reports a HIGH response for a flat patch, so this check
        is not optional;
      • the response is below `min_response`: the two patches do not share
        content. Unrelated noise measured 0.12-0.18, real matches 0.92-0.99.
    These thresholds only choose the verdict word; nothing here changes a
    detection.
    """
    import cv2

    pub = _gray(published)
    ph, pw = pub.shape[:2]
    small = cv2.resize(_gray(native), (pw, ph), interpolation=cv2.INTER_AREA)
    third = pw // 3
    if third < 16 or ph < 16:
        return StreamAlignment("INCONCLUSIVE", detail="frame too small")
    window = cv2.createHanningWindow((third, ph), cv2.CV_32F)

    shifts = {}
    reasons = []
    for name, x0 in (("left", 0), ("right", pw - third)):
        a = np.ascontiguousarray(pub[:, x0:x0 + third])
        b = np.ascontiguousarray(small[:, x0:x0 + third])
        std = float(a.std())
        if std < min_std:
            shifts[name] = None
            reasons.append(f"{name} patch flat (std {std:.1f})")
            continue
        (dx, dy), response = cv2.phaseCorrelate(a, b, window)
        if response < min_response:
            shifts[name] = None
            reasons.append(f"{name} response {response:.2f} too weak")
            continue
        shifts[name] = (round(float(dx), 2), round(float(dy), 2))

    left, right = shifts.get("left"), shifts.get("right")
    if left is None or right is None:
        return StreamAlignment("INCONCLUSIVE", left, right, "; ".join(reasons))

    worst = max(abs(v) for v in (*left, *right))
    detail = f"left {left}, right {right} px"
    if worst <= float(tolerance_px):
        return StreamAlignment("ALIGNED", left, right, detail)
    # Distance between the two patch centres, published px.
    spread = float(pw - third)
    scale_err = (right[0] - left[0]) / spread if spread > 0 else 0.0
    return StreamAlignment(
        "MISALIGNED", left, right,
        f"{detail}; implied horizontal scale error {scale_err:+.2%}")

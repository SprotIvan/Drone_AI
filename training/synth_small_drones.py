#!/usr/bin/env python3
"""
synth_small_drones.py — small drones (12-40 px) pasted onto real background
frames from the station, for the far-target case the public datasets lack.

    python synth_small_drones.py <work_dir> [--count 2000] [--seed 0]

Reads   <work_dir>/raw/*/        close-up drone images (cut-out sources)
        <work_dir>/station/      station frames WITHOUT a .txt (backgrounds)
Writes  <work_dir>/synthetic/    synth_*.jpg + YOLO .txt
        <work_dir>/synthetic_sheet.jpg   12 examples with their boxes

HOW, AND WHY THIS WAY
  • Cut-outs use a MASK, never the box rectangle. A rectangle carries a
    patch of foreign background with it, and a model learns to find patches.
    Only close-ups with ONE box, of a class the SOURCES map calls "drone",
    on a near-uniform background (catalogue white, clear sky) are used: the
    border ring around the box gives the background colour, pixels far from
    it are the drone. Masks covering too little or too much of their box,
    or touching its edge, are rejected.
  • Colour is premultiplied by opacity through resize, rotation and blur,
    so the drone's edge mixes with the frame it lands on — not with the
    catalogue white around the cut-out (a light rim) or black padding.
  • A paste is refused if it shrank below 3/4 of the wanted size (thin
    arms fading out) or is nearly the colour of where it landed (a
    labelled invisible drone teaches the model to fire on nothing).
  • Size 12-40 px on the long side, mostly 16-32: what the station's cameras
    see at tens of metres. Random flip, +/-15 deg rotation, and half of the
    drones darkened toward a silhouette (a drone against a bright sky).
  • Mask edges are feathered, then noise and a JPEG round trip are applied to
    the WHOLE frame, so the pasted region carries the same noise and
    compression as its surroundings.
  • The box is taken from the mask, so it is tight.

⚠️ Synthetic drones still differ from real ones. These frames go to TRAIN
only (build_dataset.py), and whether they helped is decided on REAL station
frames — never on synthetic ones.
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (BAD_LABELS, IMG_EXT, class_roles,  # noqa: E402
                           parse_label, read_names)
from fetch_datasets import SOURCES  # noqa: E402

MIN_SRC_AREA = 0.15        # only close-ups are cut-out sources


def cutout(img, box, ring=4, max_ring_std=22.0, dist=40.0):
    """(rgb, alpha) of the drone inside `box`, or None if not separable."""
    h, w = img.shape[:2]
    _, cx, cy, bw, bh = box
    m = 0.04
    x0, y0 = int(max(0, (cx - bw / 2 - m) * w)), int(max(0, (cy - bh / 2 - m) * h))
    x1, y1 = int(min(w, (cx + bw / 2 + m) * w)), int(min(h, (cy + bh / 2 + m) * h))
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] < 40 or crop.shape[1] < 40:
        return None
    border = np.concatenate([crop[:ring].reshape(-1, 3), crop[-ring:].reshape(-1, 3),
                             crop[:, :ring].reshape(-1, 3), crop[:, -ring:].reshape(-1, 3)])
    if border.std(axis=0).max() > max_ring_std:
        return None                                   # busy background
    bg = np.median(border, axis=0)
    fg = (np.linalg.norm(crop.astype(np.float32) - bg, axis=2) > dist).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    # ⚠️ A mask that reaches the edge of the crop means the drone was cut
    # by its box or the background is not separable — pasting it leaves a
    # straight edge or a patch of foreign background, and a model learns to
    # find those instead of drones (seen in the first trial run).
    b = 2
    if mask[:b].any() or mask[-b:].any() or mask[:, :b].any() or mask[:, -b:].any():
        return None
    # Pixels on the drone's outline are blended with the source background
    # (usually white): erode them away, or they paste as a light halo.
    k = max(3, (min(mask.shape) // 60) | 1)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    frac = mask.mean()
    if not 0.08 <= frac <= 0.85:
        return None
    # ⚠️ A mask that fills its tightest (rotated) rectangle is a card, a box
    # or a label — a product shot with the drone printed on a white card
    # over a coloured background cuts out as the whole card (seen in the
    # second trial: ~1 paste in 5). A drone's silhouette — arms, props,
    # legs — leaves much of that rectangle empty.
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    if mask.sum() / max(rw * rh, 1.0) > 0.85:
        return None
    ys, xs = np.nonzero(mask)
    crop, mask = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], \
        mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop, mask


def collect_cutouts(raw_dir, limit=600, max_reads=6000, seed=0,
                    include_nc=False):
    """Up to `limit` usable cut-outs; at most `max_reads` images are read —
    most close-ups have a busy background and are rejected, and a few
    hundred distinct drones are plenty for variety.

    ⚠️ Only boxes whose class the SOURCES map calls "drone": the raw
    folders also hold bird close-ups (drone-vs-bird) and an airliner
    ("Fixed-wing" in anti-drone) — cut out, they would be pasted as
    LABELLED drones."""
    rng = random.Random(seed)
    cands = []
    for slug, version, licence, cmap, _note in SOURCES:
        if ("NC" in licence and not include_nc) or slug in BAD_LABELS:
            continue
        name = f"{slug.replace('/', '__')}__v{version}"
        src = raw_dir / name
        if not (src / "data.yaml").exists():
            continue
        roles = class_roles(read_names(src / "data.yaml"), cmap, name)
        for split in ("train", "valid", "test"):
            for p in (src / split / "images").glob("*"):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                boxes = parse_label(p.parent.parent / "labels" / (p.stem + ".txt"))
                if len(boxes) == 1 and boxes[0][0] < len(roles) \
                        and roles[boxes[0][0]] == "drone" \
                        and boxes[0][3] * boxes[0][4] > MIN_SRC_AREA:
                    cands.append((p, boxes[0]))
    rng.shuffle(cands)
    cuts = []
    for p, box in cands[:max_reads]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        c = cutout(img, box)
        if c is not None:
            cuts.append(c)
        if len(cuts) >= limit:
            break
    return cuts, len(cands)


def paste(bg, rgb, mask, size, rng, upper_bias=0.7, margin=3,
          min_contrast=12.0):
    """Paste one drone; returns (x0, y0, x1, y1) of the tight box, or None.

    Colour is carried PREMULTIPLIED by opacity through every step (resize,
    rotation, blur). Resizing or blurring plain colour mixes the drone's
    edge with whatever surrounds it in the ARRAY — the white catalogue
    background of the cut-out, or black padding — and that pastes as a
    light or dark rim. Premultiplied, the edge mixes only with the
    background it is pasted onto, as optical blur does."""
    h, w = bg.shape[:2]
    sh, sw = mask.shape
    s = size / max(sh, sw)
    nw, nh = max(2, int(round(sw * s))), max(2, int(round(sh * s)))
    m = mask.astype(np.float32)
    p = cv2.resize(rgb.astype(np.float32) * m[..., None], (nw, nh),
                   interpolation=cv2.INTER_AREA)
    a = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        p, a = p[:, ::-1], a[:, ::-1]
    ang = rng.uniform(-15, 15)
    pad = max(nw, nh) + margin
    M = cv2.getRotationMatrix2D((pad, pad), ang, 1.0)
    canvas_p = np.zeros((2 * pad, 2 * pad, 3), np.float32)
    canvas_a = np.zeros((2 * pad, 2 * pad), np.float32)
    oy, ox = pad - nh // 2, pad - nw // 2
    canvas_p[oy:oy + nh, ox:ox + nw] = p
    canvas_a[oy:oy + nh, ox:ox + nw] = a
    p = cv2.warpAffine(canvas_p, M, (2 * pad, 2 * pad), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(canvas_a, M, (2 * pad, 2 * pad), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) < 4:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # A thin drone (arms, props) fades below half opacity when shrunk and
    # only its body is left: a 4 px box for a "16 px" drone. Refuse it —
    # the caller picks another cut-out.
    if max(y1 - y0, x1 - x0) < 0.75 * size:
        return None
    # The patch keeps a margin around the box so the blur fades out inside
    # it instead of being cut at the box edge.
    Y0, X0 = max(0, y0 - margin), max(0, x0 - margin)
    p = p[Y0:y1 + margin, X0:x1 + margin]
    a = a[Y0:y1 + margin, X0:x1 + margin]
    if rng.random() < 0.5:                       # silhouette against the sky
        p = p * rng.uniform(0.25, 0.7)
    sig = rng.uniform(0.5, 1.0)                  # optics + feathered edge
    p = cv2.GaussianBlur(p, (0, 0), sig)
    a = np.clip(cv2.GaussianBlur(a, (0, 0), sig), 0, 1)
    # The label is what is at least half opaque AFTER the blur; a drone
    # the blur fades below that everywhere is a ghost, not a target.
    core = a > 0.5
    ys, xs = np.nonzero(core)
    if len(xs) < 4:
        return None
    ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if max(ty1 - ty0, tx1 - tx0) < 0.75 * size:
        return None
    ph, pw = a.shape
    if pw >= w or ph >= h:
        return None
    ymax = h - ph
    y = int(rng.uniform(0, ymax * 0.66)) if rng.random() < upper_bias \
        else int(rng.uniform(0, ymax))
    x = int(rng.uniform(0, w - pw))
    roi = bg[y:y + ph, x:x + pw].astype(np.float32)
    out = roi * (1 - a[..., None]) + p
    # A drone the colour of what it lands on is invisible — and a LABELLED
    # invisible drone teaches the model to fire on nothing.
    if np.abs(out - roi)[core].mean() < min_contrast:
        return None
    bg[y:y + ph, x:x + pw] = out.clip(0, 255).astype(np.uint8)
    return x + tx0, y + ty0, x + tx1, y + ty1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clutter", action="store_true",
                    help="backgrounds from <work_dir>/negatives/train/ (sky, "
                         "trees, buildings, weather, birds) instead of the "
                         "station frames; written to synthetic_clutter/")
    args = ap.parse_args()
    work = Path(args.work_dir)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.clutter:
        # TRAIN backgrounds only: negatives/val must stay drone-free, it is
        # what measures false positives on scenes the model never saw.
        # The ft_v2 misses were mostly 7-24 px drones against trees and
        # buildings (val error analysis, 29.09.2026): the drones go
        # anywhere in the frame here, not mostly into the upper sky.
        bgs = [p for p in sorted((work / "negatives" / "train").rglob("*"))
               if p.suffix.lower() in IMG_EXT]
        out, prefix, upper_bias = work / "synthetic_clutter", "synthc_", 0.35
        # Busy and dark backgrounds let a drone pass the station-frame
        # contrast floor (12 levels) and still be invisible to the eye — a
        # dark drone on a night sky, a pale one on snow (seen on a sample
        # sheet). A labelled invisible drone teaches firing on nothing.
        min_contrast = 20.0
    else:
        bgs = [p for p in sorted((work / "station").rglob("*"))
               if p.suffix.lower() in IMG_EXT
               and not p.with_suffix(".txt").exists()]
        out, prefix, upper_bias = work / "synthetic", "synth_", 0.7
        min_contrast = 12.0
    if not bgs:
        sys.exit("no backgrounds")
    cuts, n_cand = collect_cutouts(work / "raw", seed=args.seed)
    print(f"backgrounds: {len(bgs)} | cut-outs: {len(cuts)} usable of {n_cand} "
          f"single-drone close-ups")
    if not cuts:
        sys.exit("no usable cut-outs")

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(prefix + "*"):
        old.unlink()
    sizes = []
    sheet = []
    for n in range(args.count):
        bg = cv2.imread(str(rng.choice(bgs)))
        h, w = bg.shape[:2]
        # Sizes are what the drone measures once the frame is letterboxed
        # to 640 — station frames are 640 wide, the other backgrounds are
        # not, so the pasted size scales with the frame.
        scale = max(h, w) / 640.0
        boxes = []
        for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
            size = rng.uniform(16, 32) if rng.random() < 0.7 else rng.uniform(12, 40)
            size *= scale
            for _try in range(10):
                rgb, mask = rng.choice(cuts)
                snapshot = bg.copy()
                b = paste(bg, rgb, mask, size, rng, upper_bias=upper_bias,
                          min_contrast=min_contrast)
                if b is None:
                    continue                  # too thin at this size
                if all(b[2] <= o[0] or b[0] >= o[2] or b[3] <= o[1] or b[1] >= o[3]
                       for o in boxes):
                    boxes.append(b)
                    sizes.append(max(b[2] - b[0], b[3] - b[1]) / scale)
                    break
                bg = snapshot                              # overlapped: undo
        if not boxes:
            continue
        noisy = bg.astype(np.float32) + np.random.normal(0, rng.uniform(0.5, 3.0), bg.shape)
        ok, enc = cv2.imencode(".jpg", noisy.clip(0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(80, 95)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        stem = out / f"{prefix}{n:05d}"
        cv2.imwrite(str(stem) + ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        with open(str(stem) + ".txt", "w") as f:
            for x0, y0, x1, y1 in boxes:
                f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                        f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
        if len(sheet) < 12:
            vis = img.copy()
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(vis, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 0, 255), 1)
            sheet.append(cv2.resize(vis, (480, 360)))
    while len(sheet) < 12:
        sheet.append(np.zeros((360, 480, 3), np.uint8))
    # Outside the output folder: build_dataset.py takes every image in it.
    cv2.imwrite(str(work / f"{out.name}_sheet.jpg"),
                np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, 12, 4)]))
    s = np.array(sizes)
    print(f"wrote {len(list(out.glob(prefix + '*.jpg')))} images, {len(s)} drones; "
          f"long side at 640 px: min {s.min():.0f} median {np.median(s):.0f} "
          f"max {s.max():.0f}")


if __name__ == "__main__":
    main()

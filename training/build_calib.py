#!/usr/bin/env python3
"""
build_calib.py — the calibration set for the Hailo compile, taken from the
TRAIN split of the merged dataset and preprocessed exactly the way the
station hands a frame to the HEF.

    python build_calib.py <dataset_dir> <out_dir> [--n 1024] [--seed 0]
    python build_calib.py <dataset_dir> <out_dir> --pick-only   # list only
    python build_calib.py <dataset_dir> <out_dir> --pick calib_pick.txt

<dataset_dir>   the unzipped drone_dataset (images/train, labels/train)
<out_dir>/000000.npy ...   one image per file: float32 (640, 640, 3), RGB, 0..255
<out_dir>/calib_list.txt   which dataset image each file came from, and why
<out_dir>/calib_sheet.jpg  16 of them, for a look

WHY THESE CHOICES
  * TRAIN split only. The optimizer's quantization-aware fine-tune adjusts
    the quantized weights on these images; taking them from val would leak
    val into the model that the comparison is then run on.
  * Letterboxed to 640x640 with grey 114, RGB, values 0..255 — the same
    arithmetic as HailoInference._letterbox in TWO_CAMERAS_FIXED.py (the
    station's 640x480 frame lands between two 80-px grey bars). The chip
    divides by 255 itself (normalization in the model script), so values
    stay 0..255. The current HEF was calibrated on images STRETCHED to
    640x640 (dron_buster/hailo_recompile/build_calib.py) — a geometry the
    station never sends.
  * Stratified by drone size, so each of the three heads sees positives.
    Stride 8 decides the far, small targets; a set of mostly large drones
    would fit its class output's range to background alone. All train
    images without a drone (the station's own frames) are included, up to
    10 % of the set: they are the scene the chip actually looks at.
  * One file per image, not one 5 GB array: compile_hef.py stacks them into
    the in-memory array the current HEF was built from when RAM allows,
    and otherwise hands the Dataflow Compiler the directory itself (its
    "npy_dir" input: one image per .npy file).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 640
PAD = 114
# Long side of the LARGEST drone in the frame, px after letterboxing to 640.
BUCKETS = [("<16", 0, 16, 0.25), ("16-32", 16, 32, 0.30),
           ("32-64", 32, 64, 0.25), (">=64", 64, 1e9, 0.20)]
MAX_BACKGROUND_SHARE = 0.10


def letterbox(img):
    """Copy of HailoInference._letterbox (TWO_CAMERAS_FIXED.py) — keep the
    two identical: calibration must see what the chip is fed."""
    sh, sw = img.shape[:2]
    r = min(SIZE / sh, SIZE / sw)
    nw, nh = int(round(sw * r)), int(round(sh * r))
    left = int(round((SIZE - nw) / 2.0 - 0.1))
    top = int(round((SIZE - nh) / 2.0 - 0.1))
    out = np.full((SIZE, SIZE, 3), PAD, dtype=np.uint8)
    if nw == sw and nh == sh:
        out[top:top + nh, left:left + nw] = img
    else:
        out[top:top + nh, left:left + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def largest_drone_px(label_path, w, h):
    """Long side (px at 640) of the largest box, 0.0 for a background."""
    best = 0.0
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            best = max(best, max(float(p[3]) * w, float(p[4]) * h))
    return best * SIZE / max(w, h)


def select(ds, n, seed):
    """[(image path, why)] — the calibration images, in the order they go
    into the set (the optimizer's statistics use the first 64)."""
    backgrounds, pools = [], {name: [] for name, *_ in BUCKETS}
    for img in sorted((ds / "images" / "train").glob("*.jpg")):
        lab = ds / "labels" / "train" / (img.stem + ".txt")
        if not lab.exists():
            continue
        with Image.open(img) as im:         # header only, no pixel decode
            w, h = im.size
        px = largest_drone_px(lab, w, h)
        if px == 0.0:
            backgrounds.append(img)
            continue
        for name, lo, hi, _share in BUCKETS:
            if lo <= px < hi:
                pools[name].append(img)
                break

    rng = random.Random(seed)
    n_bg = min(len(backgrounds), int(n * MAX_BACKGROUND_SHARE))
    picked = [(p, "background") for p in rng.sample(backgrounds, n_bg)]
    n_drone = n - n_bg
    for name, _lo, _hi, share in BUCKETS:
        take = min(int(round(n_drone * share)), len(pools[name]))
        rng.shuffle(pools[name])
        picked += [(p, name) for p in pools[name][:take]]
        pools[name] = pools[name][take:]
    # Rounding, or a bucket with too few images: fill from what is left.
    rest = [(p, name) for name, v in pools.items() for p in v]
    rng.shuffle(rest)
    picked += rest[:max(0, n - len(picked))]
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick-only", action="store_true",
                    help="write <out_dir>/calib_pick.txt (the selection, in "
                         "order) and stop — to ship only those images")
    ap.add_argument("--pick", help="a calib_pick.txt to follow instead of "
                                   "selecting: exactly those images, in order")
    args = ap.parse_args()
    ds, out = Path(args.dataset_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.pick:
        picked = []
        for line in Path(args.pick).read_text().splitlines():
            why, name = line.split("\t")
            picked.append((ds / "images" / "train" / name, why))
    else:
        picked = select(ds, args.n, args.seed)
    if args.pick_only:
        (out / "calib_pick.txt").write_text(
            "".join(f"{why}\t{p.name}\n" for p, why in picked))
        print(f"{len(picked)} picked -> {out / 'calib_pick.txt'}")
        return
    for old in out.glob("*.npy"):
        old.unlink()

    counts = {}
    with open(out / "calib_list.txt", "w") as f:
        for i, (p, why) in enumerate(picked):
            bgr = cv2.imread(str(p))
            rgb = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            np.save(out / f"{i:06d}.npy", rgb.astype(np.float32))
            f.write(f"{i:06d}.npy\t{why}\t{p.name}\n")
            counts[why] = counts.get(why, 0) + 1
    print(f"calibration set: {len(picked)} images -> {out}")
    print("  by largest drone (px at 640):",
          ", ".join(f"{k} {counts.get(k, 0)}"
                    for k in [b[0] for b in BUCKETS] + ["background"]))

    tiles = [cv2.resize(cv2.cvtColor(np.load(out / f"{i:06d}.npy")
                                     .astype(np.uint8), cv2.COLOR_RGB2BGR),
                        (320, 320)) for i in range(min(16, len(picked)))]
    while len(tiles) < 16:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r * 4:r * 4 + 4]) for r in range(4)])
    cv2.imwrite(str(out / "calib_sheet.jpg"), sheet)


if __name__ == "__main__":
    main()

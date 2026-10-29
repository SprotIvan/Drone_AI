#!/usr/bin/env python3
"""
select_closeups.py — choose the close-up drone photos for ft_v8 from the
candidates of collect_closeups.py, and split them into train / val by
independent group.

    python select_closeups.py CANDIDATES_DIR OUT_DIR

Rules (each one counted in OUT_DIR/selection.json):
  * zhejiang-university (drones-detection-with-yolov8) dropped whole: its
    images are zoomed crops of the uav-detection photos with boxes that run
    off the image (seen on review/closeups/zhejiang_000.jpg).
  * largest box > 70 % of the image dropped (the drone IS the image — product
    shots; the station never sees that), and boxes cut by >= 3 image borders.
  * two labels on one object (IoU > 0.6) -> image dropped.
  * sequence groups (drone-vs-bird videoNN, paresh octocopter_K, anti-drone
    per-person series) that have a frame in the independent test
    eval_v5/drone_close are dropped whole — other frames of the same video
    must not be trained on.
  * near-duplicates inside the candidates (dHash <= 6 and 64x64 thumbnails
    within 6 grey levels) -> one kept per cluster.
  * per-source caps for train; ~12 % of the GROUPS of every source held out
    as close-up validation (a group never on both sides).
"""
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TEST_SRC = Path(r"C:\Github\drone_training\eval_v5\drone_close\source.txt")
CAPS = {"drone-vs-bird-detection": 800, "drone-detect-suvzw": 700, "drone-yolov5-b4787": 450}
MAX_AREA = 0.70
VAL_FRAC = 0.12


def stem_of(raw):
    return Path(raw).name.split("_jpg.rf.")[0].split("_png.rf.")[0].split(".rf.")[0]


def group(src, stem):
    if "drone-vs-bird" in src:
        m = re.match(r"(video\d+)_", stem)
        return m.group(1) if m else "photo:" + stem
    if "paresh" in src:
        m = re.match(r"([A-Za-z]+)_(\d+)_\d+$", stem)
        return f"{m.group(1)}_{m.group(2)}" if m else "photo:" + stem
    if "anti-drone" in src:
        m = re.match(r"([A-Za-z]+)", stem)
        return m.group(1) if m else stem
    return "photo:" + stem


def short(src):
    return src.split("__")[1]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def bits(a):
    d = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in d), 2)


def main():
    cand, out = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads((cand / "manifest.json").read_text())
    test_groups = set()
    for line in TEST_SRC.read_text(encoding="utf-8").splitlines():
        _, raw = line.split(" ", 1)
        src = raw.split("\\raw\\")[1].split("\\")[0]
        g = group(src, stem_of(raw))
        if not g.startswith("photo:"):
            test_groups.add((short(src), g))
    stats = Counter()
    keep = []
    for e in man:
        s = short(e["src"])
        e["group"] = group(e["src"], stem_of(e["path"]))
        boxes = [[float(v) for v in l.split()[1:]] for l in
                 (cand / "labels" / (Path(e["file"]).stem + ".txt")).read_text().splitlines() if l.strip()]
        e["box_list"] = boxes
        if "zhejiang" in e["src"]:
            stats["drop: zhejiang source"] += 1
            continue
        if (s, e["group"]) in test_groups:
            stats["drop: same video/series as a drone_close test image"] += 1
            continue
        if e["max_area"] > MAX_AREA:
            stats[f"drop: largest box > {MAX_AREA:.0%} of image"] += 1
            continue
        cut = max(sum((cx - w / 2 <= 0.002, cy - h / 2 <= 0.002, cx + w / 2 >= 0.998, cy + h / 2 >= 0.998))
                  for cx, cy, w, h in boxes)
        if cut >= 3:
            stats["drop: box cut by >= 3 image borders"] += 1
            continue
        if any(iou(boxes[i], boxes[j]) > 0.6 for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            stats["drop: two labels on one object"] += 1
            continue
        keep.append(e)
    # near-duplicates inside the candidates
    H, TH = [], []
    for e in keep:
        with Image.open(cand / "images" / e["file"]) as im:
            g = im.convert("L")
            H.append(bits(np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)))
            TH.append(np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.int16))
    H = np.array(H, dtype=np.uint64)
    TH = np.stack(TH)
    alive = np.ones(len(keep), bool)
    for i in range(len(keep)):
        if not alive[i]:
            continue
        d = np.bitwise_count(H ^ H[i])
        cand_j = np.flatnonzero((d <= 6) & alive)
        cand_j = cand_j[cand_j > i]
        if len(cand_j):
            mad = np.abs(TH[cand_j] - TH[i]).mean(axis=(1, 2))
            for j in cand_j[mad <= 6]:
                alive[j] = False
                stats["drop: near-duplicate of another candidate"] += 1
    keep = [e for e, a in zip(keep, alive) if a]
    # group split + caps
    rng = random.Random(8)
    by_src = defaultdict(lambda: defaultdict(list))
    for e in keep:
        by_src[short(e["src"])][e["group"]].append(e)
    train, val = [], []
    for s, groups in sorted(by_src.items()):
        gl = sorted(groups)
        rng.shuffle(gl)
        n_img = sum(len(groups[g]) for g in gl)
        v, t = [], []
        for g in gl:
            (v if sum(len(x) for x in v) < VAL_FRAC * n_img else t).append(groups[g])
        vi = [e for x in v for e in x]
        ti = [e for x in t for e in x]
        cap = CAPS.get(s)
        if cap and len(ti) > cap:
            rng.shuffle(ti)
            stats[f"drop: over the train cap of {s} ({cap})"] += len(ti) - cap
            ti = ti[:cap]
        if len(vi) > 120:
            rng.shuffle(vi)
            stats[f"drop: over the val cap of {s} (120)"] += len(vi) - 120
            vi = vi[:120]
        train += ti
        val += vi
    for part, items in (("train", train), ("val", val)):
        (out / part / "images").mkdir(parents=True, exist_ok=True)
        (out / part / "labels").mkdir(parents=True, exist_ok=True)
        for e in items:
            shutil.copy2(cand / "images" / e["file"], out / part / "images" / e["file"])
            shutil.copy2(cand / "labels" / (Path(e["file"]).stem + ".txt"),
                         out / part / "labels" / (Path(e["file"]).stem + ".txt"))
    rep = {"rules": dict(stats), "test_sequence_groups": sorted(map(list, test_groups)),
           "train": dict(Counter(short(e["src"]) for e in train)), "val": dict(Counter(short(e["src"]) for e in val)),
           "train_boxes": sum(len(e["box_list"]) for e in train), "val_boxes": sum(len(e["box_list"]) for e in val),
           "train_groups": len({(short(e["src"]), e["group"]) for e in train}),
           "val_groups": len({(short(e["src"]), e["group"]) for e in val}),
           "items": {"train": [{k: e[k] for k in ("file", "src", "group", "path", "max_area")} for e in train],
                     "val": [{k: e[k] for k in ("file", "src", "group", "path", "max_area")} for e in val]}}
    assert not ({(short(e["src"]), e["group"]) for e in train} & {(short(e["src"]), e["group"]) for e in val})
    (out / "selection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
label_check.py — candidate labelling errors in a YOLO dataset, found by
disagreement between the labels and TWO models' predictions (predict_pool.py
output). A model is never taken as ground truth: every category below is a
list of images to LOOK AT, drawn on sheets (labels green, model A blue,
model B magenta) for review by eye.

    python label_check.py DATASET.zip PRED_A.json PRED_B.json OUT_DIR [--sample 48]

  bg_confident   background image (no label) where BOTH models put a box >= 0.50
                 at the same place (IoU >= 0.5): an unlabelled drone, or a hard
                 negative worth knowing about
  missing_label  drone image with a box of BOTH models >= 0.60 (IoU >= 0.5)
                 touching no label (IoU < 0.10): a drone without a label?
  label_unfound  label box that NEITHER model matches at all (no box >= 0.05 with
                 IoU >= 0.10): a wrong label, or a drone both models cannot see
  loose_box      label box where both models agree with each other (IoU >= 0.7,
                 both >= 0.50) but not with the label (IoU 0.10-0.50): the label
                 may be off
"""
import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def source_of(name):
    stem = Path(name).stem
    return stem.split("__")[0] if "__" in stem else stem.rsplit("_", 1)[0]


def gt_boxes(z, img_name, w, h):
    lab = img_name.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
    out = []
    try:
        text = z.read(lab).decode()
    except KeyError:
        return out
    for line in text.splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def agreed(pa, pb, thr):
    """Boxes both models put at the same place, each >= thr."""
    out = []
    for a in pa:
        if a[4] < thr:
            continue
        for b in pb:
            if b[4] >= thr and iou(a, b) >= 0.5:
                out.append((a, b))
                break
    return out


def sheet(z, items, out_path, cols=6, size=320):
    tiles = []
    for name, info in items:
        im = cv2.cvtColor(np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = im.shape[:2]
        k = size / max(w, h)
        im = cv2.resize(im, (int(w * k), int(h * k)))
        can = np.full((size, size, 3), 30, np.uint8)
        can[:im.shape[0], :im.shape[1]] = im
        for (x1, y1, x2, y2) in info["gt"]:
            cv2.rectangle(can, (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k)), (0, 255, 0), 1)
        for col, key in (((255, 160, 0), "a"), ((255, 0, 255), "b")):
            for x1, y1, x2, y2, s in info[key]:
                if s >= 0.3:
                    cv2.rectangle(can, (int(x1 * k) - 1, int(y1 * k) - 1), (int(x2 * k) + 1, int(y2 * k) + 1), col, 1)
        cv2.rectangle(can, (0, 0), (size - 1, 16), (0, 0, 0), -1)
        cv2.putText(can, f"{info['i']} {Path(name).stem[:34]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (255, 255, 255), 1)
        tiles.append(can)
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    if tiles:
        cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("pred_a")
    ap.add_argument("pred_b")
    ap.add_argument("out")
    ap.add_argument("--sample", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    PA = json.loads(Path(a.pred_a).read_text())["preds"]
    PB = json.loads(Path(a.pred_b).read_text())["preds"]
    z = zipfile.ZipFile(a.dataset)
    cats = defaultdict(list)
    for name in sorted(PA):
        pa, pb = PA[name] or [], PB.get(name) or []
        with Image.open(io.BytesIO(z.read(name))) as im:
            w, h = im.size
        gt = gt_boxes(z, name, w, h)
        info = {"gt": gt, "a": pa, "b": pb, "split": name.split("/")[1], "src": source_of(name)}
        if not gt:
            ag = agreed(pa, pb, 0.50)
            if ag:
                cats["bg_confident"].append((name, dict(info, score=max(min(x[4], y[4]) for x, y in ag))))
            continue
        ag = agreed(pa, pb, 0.60)
        if any(max(iou(x, g) for g in gt) < 0.10 for x, _ in ag):
            cats["missing_label"].append((name, info))
        for g in gt:
            if not any(iou(p, g) >= 0.10 for p in pa + pb):
                cats["label_unfound"].append((name, info))
                break
        for g in gt:
            pair = [(x, y) for x, y in agreed(pa, pb, 0.50) if iou(x, y) >= 0.7]
            if pair and all(0.10 <= iou(x, g) < 0.50 for x, _ in pair) and \
                    max(iou(x, g) for x, _ in pair) < 0.5:
                cats["loose_box"].append((name, info))
                break
    rng = random.Random(a.seed)
    summary = {}
    for c, items in cats.items():
        by_split = Counter(i["split"] for _, i in items)
        by_src = Counter(i["src"] for _, i in items)
        pick = items if len(items) <= a.sample else rng.sample(items, a.sample)
        pick = sorted(pick)
        for k, (n, i) in enumerate(pick):
            i["i"] = k
        for s in range(0, len(pick), 24):
            sheet(z, pick[s:s + 24], out / f"{c}_{s // 24:02d}.jpg")
        summary[c] = {"count": len(items), "by_split": dict(by_split), "by_source": dict(by_src.most_common()),
                      "sheet_items": [n for n, _ in pick]}
        print(f"{c}: {len(items)}  {dict(by_split)}  top sources {dict(by_src.most_common(6))}")
    (out / "label_check.json").write_text(json.dumps({"summary": summary,
                                                       "all": {c: [n for n, _ in v] for c, v in cats.items()}},
                                                      indent=1))


if __name__ == "__main__":
    main()

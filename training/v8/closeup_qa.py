#!/usr/bin/env python3
"""
closeup_qa.py — label check of the selected close-ups by 2-fold cross
prediction: a model fine-tuned (from ft_v5) on fold A predicts fold B and vice
versa, so every image is judged by a model that never saw it. Folds are split
by the selection's group (video / series / single photo). A model is never
ground truth: the flags are a list of images to LOOK AT.

    python closeup_qa.py SELECTED_DIR OUT_DIR [--epochs 15]

flags per image
  label_unfound  a label box no held-out box (>= 0.10) overlaps with IoU >= 0.30
  loose          the held-out model's box (>= 0.50) overlaps the label only
                 0.10-0.55 and nothing overlaps it better
  extra          a held-out box >= 0.50 overlapping no label (IoU < 0.10)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def labels(img_path, w, h):
    lp = Path(str(img_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    out = []
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selected")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--base", default=r"C:\Github\drone_training\v8\models\ft_v5.pt")
    a = ap.parse_args()
    from ultralytics import YOLO
    sel, out = Path(a.selected), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((sel / "selection.json").read_text())
    items = [dict(e, part=p) for p in ("train", "val") for e in rep["items"][p]]
    groups = sorted({(e["src"], e["group"]) for e in items})
    random.Random(5).shuffle(groups)
    fold_of = {g: i % 2 for i, g in enumerate(groups)}
    for e in items:
        e["fold"] = fold_of[(e["src"], e["group"])]
        e["img"] = str(sel / e["part"] / "images" / e["file"])
    preds = {}
    for k in (0, 1):
        tr = [e["img"] for e in items if e["fold"] == k]
        te = [e["img"] for e in items if e["fold"] != k]
        (out / f"fold{k}_train.txt").write_text("\n".join(tr) + "\n")
        (out / f"fold{k}_test.txt").write_text("\n".join(te) + "\n")
        (out / f"fold{k}.yaml").write_text(
            f"train: {out / f'fold{k}_train.txt'}\nval: {out / f'fold{k}_test.txt'}\nnc: 1\nnames: ['drone']\n")
        best = out / f"fold{k}" / "weights" / "best.pt"
        if not best.exists():
            m = YOLO(a.base)
            m.train(data=str(out / f"fold{k}.yaml"), epochs=a.epochs, imgsz=640, batch=a.batch, workers=a.workers,
                    project=str(out), name=f"fold{k}", exist_ok=True, optimizer="SGD", lr0=0.005,
                    close_mosaic=5, patience=50, seed=0, plots=False, verbose=False)
        m = YOLO(str(best))
        for s in range(0, len(te), 64):
            for src, r in zip(te[s:s + 64], m.predict(te[s:s + 64], imgsz=640, conf=0.05, verbose=False)):
                b = r.boxes
                preds[Path(src).name] = [[*map(float, xy), float(c)] for xy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())]
    flags = {"label_unfound": [], "loose": [], "extra": []}
    for e in items:
        im = cv2.imread(e["img"])
        h, w = im.shape[:2]
        gt = labels(e["img"], w, h)
        pr = preds.get(Path(e["img"]).name, [])
        f = set()
        for g in gt:
            if not any(p[4] >= 0.10 and iou(p, g) >= 0.30 for p in pr):
                f.add("label_unfound")
            ov = [iou(p, g) for p in pr if p[4] >= 0.50]
            if ov and 0.10 <= max(ov) < 0.55 and not any(iou(p, g) >= 0.55 for p in pr if p[4] >= 0.25):
                f.add("loose")
        for p in pr:
            if p[4] >= 0.50 and all(iou(p, g) < 0.10 for g in gt):
                f.add("extra")
        e["flags"] = sorted(f)
        e["gt"] = gt
        e["pred"] = [p for p in pr if p[4] >= 0.25]
        for x in f:
            flags[x].append(e["file"])
    flagged = [e for e in items if e["flags"]]
    print({k: len(v) for k, v in flags.items()}, "flagged images", len(flagged), "of", len(items))
    (out / "qa.json").write_text(json.dumps({"counts": {k: len(v) for k, v in flags.items()},
                                             "flagged": [{k: e[k] for k in ("file", "part", "src", "group", "flags",
                                                                            "gt", "pred")} for e in flagged]},
                                            indent=0))
    tiles = []
    for e in flagged:
        im = cv2.imread(e["img"])
        h, w = im.shape[:2]
        kk = 240 / max(w, h)
        im = cv2.resize(im, (int(w * kk), int(h * kk)))
        can = np.full((240, 240, 3), 30, np.uint8)
        can[:im.shape[0], :im.shape[1]] = im
        for x1, y1, x2, y2 in e["gt"]:
            cv2.rectangle(can, (int(x1 * kk), int(y1 * kk)), (int(x2 * kk), int(y2 * kk)), (0, 255, 0), 2)
        for x1, y1, x2, y2, c in e["pred"]:
            cv2.rectangle(can, (int(x1 * kk), int(y1 * kk)), (int(x2 * kk), int(y2 * kk)), (255, 0, 255), 1)
        cv2.rectangle(can, (0, 0), (239, 14), (0, 0, 0), -1)
        cv2.putText(can, f"{len(tiles)} {''.join(x[0] for x in e['flags'])}", (2, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(can)
        e["tile"] = len(tiles) - 1
    cols, per = 8, 48
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        while len(t) % cols:
            t.append(np.zeros((240, 240, 3), np.uint8))
        cv2.imwrite(str(out / f"flagged_{s // per:02d}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "flagged_index.json").write_text(json.dumps([e["file"] for e in flagged], indent=0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
screen_negatives.py — before negatives (images WITHOUT a drone) are used, every
box ft_v5 / ft_v7 put on them at >= THR is cut out and drawn on sheets, to be
looked at: a real drone there means the image is not a negative.

    python screen_negatives.py OUT_DIR DIR [DIR ...] [--thr 0.25]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--thr", type=float, default=0.25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for d in a.dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMG)
    hits = {}
    for m in ("ft_v5", "ft_v7"):
        model = YOLO(rf"C:\Github\drone_training\v8\models\{m}.pt")
        for s in range(0, len(paths), 64):
            chunk = paths[s:s + 64]
            for src, r in zip(chunk, model.predict(chunk, imgsz=640, conf=a.thr, verbose=False)):
                for xy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    hits.setdefault(src, []).append([m, *map(float, xy), float(c)])
    tiles, index = [], []
    for src, boxes in sorted(hits.items()):
        im = cv2.imread(src)
        h, w = im.shape[:2]
        for m, x1, y1, x2, y2, c in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1) * 1.2 + 24
            X0, Y0, X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))
            z = im[Y0:Y1, X0:X1].copy()
            cv2.rectangle(z, (int(x1 - X0), int(y1 - Y0)), (int(x2 - X0), int(y2 - Y0)), (0, 0, 255), 1)
            z = cv2.resize(z, (240, 240), interpolation=cv2.INTER_CUBIC)
            cv2.putText(z, f"{len(tiles)} {m[3:]} {c:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(z)
            index.append({"tile": len(tiles) - 1, "file": Path(src).name, "path": src, "model": m, "score": c})
    cols, per = 8, 48
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        while len(t) % cols:
            t.append(np.zeros((240, 240, 3), np.uint8))
        cv2.imwrite(str(out / f"screen_{s // per:02d}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "screen_index.json").write_text(json.dumps(index, indent=0))
    print(len(paths), "images,", len(hits), "with a box >=", a.thr, ",", len(tiles), "boxes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gt_assist.py — help complete station ground truth by eye. For each spec
FRAME:PX:PY[:HALF] (station_gt.json index, PUBLISHED px centre, half-size in
published px, default 45) a native crop is upscaled, ft_v5 / ft_v7 are run on
it at conf 0.05 (proposals only — every box is judged by eye), and the crop is
drawn with a published-px grid (tick every 5 px, label every 20). A 5th field
x1,y1,x2,y2 (published px) draws a candidate box instead of running models.

    python gt_assist.py OUT.jpg 76:400:300 78:205:280:60 ...
Prints every proposal as  FRAME model score  [x, y, w, h] published px.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

TS = Path(r"C:\Github\drone_training\v8\testsets")
gt = json.loads((TS / "station_gt.json").read_text())
specs = sys.argv[2:]
need_models = any(len(s.split(":")) < 5 for s in specs)
models = {}
if need_models:
    from ultralytics import YOLO
    models = {m: YOLO(rf"C:\Github\drone_training\v8\models\{m}.pt") for m in ("ft_v5", "ft_v7")}
S = 360
tiles = []
for spec in specs:
    p = spec.split(":")
    i, px, py = int(p[0]), float(p[1]), float(p[2])
    half = float(p[3]) if len(p) > 3 and p[3] else 45.0
    fr = gt[i]
    nat = cv2.imread(str(TS / "station" / fr["session"] / fr["nat"]))
    k = nat.shape[1] / 640.0
    x0, y0 = int(max(0, (px - half) * k)), int(max(0, (py - half) * k))
    x1, y1 = int(min(nat.shape[1], (px + half) * k)), int(min(nat.shape[0], (py + half) * k))
    crop = nat[y0:y1, x0:x1]
    z = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    sx, sy = S / (x1 - x0), S / (y1 - y0)

    def to_z(xp, yp):          # published px -> zoom px
        return int((xp * k - x0) * sx), int((yp * k - y0) * sy)

    for g in range(int((x0 / k) // 5 * 5), int(x1 / k) + 1, 5):
        zx, _ = to_z(g, 0)
        if 0 <= zx < S:
            big = g % 20 == 0
            cv2.line(z, (zx, 0), (zx, 10 if big else 4), (0, 255, 255), 1)
            if big:
                cv2.putText(z, str(g), (zx + 1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    for g in range(int((y0 / k) // 5 * 5), int(y1 / k) + 1, 5):
        _, zy = to_z(0, g)
        if 0 <= zy < S:
            big = g % 20 == 0
            cv2.line(z, (0, zy), (10 if big else 4, zy), (0, 255, 255), 1)
            if big:
                cv2.putText(z, str(g), (12, zy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    for o in fr["objects"]:
        bx, by, bw, bh = o["box"]
        col = {"D": (0, 255, 0), "U": (0, 255, 255), "F": (0, 0, 255)}[o["lab"]]
        cv2.rectangle(z, to_z(bx, by), to_z(bx + bw, by + bh), col, 1)
    if len(p) > 4:
        c = [float(v) for v in p[4].split(",")]
        cv2.rectangle(z, to_z(c[0], c[1]), to_z(c[2], c[3]), (255, 255, 0), 1)
    else:
        inp = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
        for m, col in (("ft_v5", (255, 160, 0)), ("ft_v7", (255, 0, 255))):
            r = models[m].predict(inp, imgsz=640, conf=0.05, verbose=False)[0]
            bb = r.boxes.xyxy.cpu().numpy()
            cc = r.boxes.conf.cpu().numpy()
            for b, s in sorted(zip(bb, cc), key=lambda t: -t[1])[:3]:
                bx1 = (x0 + b[0] * (x1 - x0) / 640) / k
                by1 = (y0 + b[1] * (y1 - y0) / 640) / k
                bx2 = (x0 + b[2] * (x1 - x0) / 640) / k
                by2 = (y0 + b[3] * (y1 - y0) / 640) / k
                print(f"{i} {m} {s:.2f} [{bx1:.1f}, {by1:.1f}, {bx2 - bx1:.1f}, {by2 - by1:.1f}]")
                cv2.rectangle(z, to_z(bx1, by1), to_z(bx2, by2), col, 1)
                cv2.putText(z, f"{m[-1]}{s:.2f}", (to_z(bx1, by1)[0], max(30, to_z(bx1, by1)[1] - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.putText(z, f"{i}", (S - 40, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    tiles.append(z)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros((S, S, 3), np.uint8))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])

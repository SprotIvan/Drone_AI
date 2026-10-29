"""Crops of a model's boxes (>= THR) on an image test set, to check by eye what
the 'false boxes' really are.   python _fp_sheet.py MODEL SET THR OUT.jpg"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

V8 = Path(r"C:\Github\drone_training\v8")
model, sset, thr, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
preds = json.loads((V8 / "eval" / "preds" / f"{model}.json").read_text())["sets"][sset]
tiles = []
for name, boxes in sorted(preds.items()):
    im = cv2.imread(str(V8 / "testsets" / sset / "images" / name))
    h, w = im.shape[:2]
    for x1, y1, x2, y2, s in boxes:
        if s < thr:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.3 + 30
        X0, Y0, X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))
        z = im[Y0:Y1, X0:X1].copy()
        cv2.rectangle(z, (int(x1 - X0), int(y1 - Y0)), (int(x2 - X0), int(y2 - Y0)), (0, 0, 255), 1)
        z = cv2.resize(z, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.putText(z, f"{len(tiles)} {s:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(z)
cols = 10
while len(tiles) % cols:
    tiles.append(np.zeros((200, 200, 3), np.uint8))
cv2.imwrite(out, np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(tiles), "tiles")

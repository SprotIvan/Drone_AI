"""Zoom regions of drone_photos images given in label_assist TILE px.
    python _zoom_tiles.py OUT.jpg IDX:TX:TY:THALF ...  (tile px; THALF half-size in tile px)
Each crop is cut from the ORIGINAL image and shown at 360x360 with its tile-px
grid (ticks every 10 tile px)."""
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"C:\Github\drone_training\v8\testsets\drone_photos\images")
names = sorted(p.name for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
S = 360
tiles = []
for spec in sys.argv[2:]:
    i, tx, ty, th = (float(v) for v in spec.split(":"))
    im = cv2.imread(str(SRC / names[int(i)]))
    h, w = im.shape[:2]
    k = 480 / max(w, h)
    x0, y0 = int(max(0, (tx - th) / k)), int(max(0, (ty - th) / k))
    x1, y1 = int(min(w, (tx + th) / k)), int(min(h, (ty + th) / k))
    z = cv2.resize(im[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_CUBIC)
    sx, sy = S / (x1 - x0), S / (y1 - y0)
    for g in range(int((x0 * k) // 10 * 10), int(x1 * k) + 1, 10):
        zx = int((g / k - x0) * sx)
        if 0 <= zx < S:
            cv2.line(z, (zx, 0), (zx, 10 if g % 50 == 0 else 4), (0, 255, 255), 1)
            if g % 50 == 0:
                cv2.putText(z, str(g), (zx + 1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for g in range(int((y0 * k) // 10 * 10), int(y1 * k) + 1, 10):
        zy = int((g / k - y0) * sy)
        if 0 <= zy < S:
            cv2.line(z, (0, zy), (10 if g % 50 == 0 else 4, zy), (0, 255, 255), 1)
            if g % 50 == 0:
                cv2.putText(z, str(g), (12, zy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(z, str(int(i)), (S - 50, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    tiles.append(z)
while len(tiles) % 5:
    tiles.append(np.zeros((S, S, 3), np.uint8))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[j:j + 5]) for j in range(0, len(tiles), 5)]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])

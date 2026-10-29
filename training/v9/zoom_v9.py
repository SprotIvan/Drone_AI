#!/usr/bin/env python3
"""
zoom_v9.py — a closer look at chosen objects: for each id, a 400 px crop of the native frame around the object
(box in red) next to the whole published frame (box in red), to decide doubtful tiles.

    python zoom_v9.py OUT.jpg ID [ID ...]          (ids of v9/label/objects.json; "n123" = objects_near.json)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

V9 = Path(r"C:\Github\drone_training\v9")
RAW = V9 / "raw_15_10"


def main():
    out, ids = sys.argv[1], sys.argv[2:]
    objs = {str(o["id"]): o for o in json.loads((V9 / "label" / "objects.json").read_text())}
    near = V9 / "label" / "objects_near.json"
    if near.exists():
        objs.update({"n" + str(o["id"]): o for o in json.loads(near.read_text())})
    rows = []
    for i in ids:
        o = objs[i]
        sess, pub = o["key"].split("/")
        nat = cv2.imread(str(RAW / sess / pub.replace("_pub.jpg", "_nat.jpg")))
        full = cv2.imread(str(RAW / sess / pub))
        k = nat.shape[1] / 640.0
        x, y, w, h = o["box"]
        X, Y, W, H = x * k, y * k, w * k, h * k
        half = max(W, H) * 2.2 + 70
        cx, cy = X + W / 2, Y + H / 2
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
        z = nat[y0:y1, x0:x1].copy()
        cv2.rectangle(z, (int(X - x0) - 3, int(Y - y0) - 3), (int(X + W - x0) + 3, int(Y + H - y0) + 3), (0, 0, 255), 1)
        z = cv2.resize(z, (480, 480), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(full, (int(x) - 4, int(y) - 4), (int(x + w) + 4, int(y + h) + 4), (0, 0, 255), 2)
        full = cv2.resize(full, (640, 480))
        sc = " ".join(f"{m}:{s:.2f}" for m, s in sorted(o["scores"].items()))
        cv2.putText(full, f"#{i} {sess[-6:]} {pub[9:17]} {sc}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        rows.append(np.hstack([z, full]))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()

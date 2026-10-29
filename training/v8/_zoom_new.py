"""Context for eval_v8 'new' objects: the published frame with the box + a x3
native crop around it.   python _zoom_new.py OUT.jpg 19,21,22 [--half 60]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EV = Path(r"C:\Github\drone_training\v8\eval\new_objects")
TS = Path(r"C:\Github\drone_training\v8\testsets")
objs = {o["id"]: o for o in json.loads((EV / "new_objects.json").read_text())}
ids = [int(x) for x in sys.argv[2].split(",")]
half_px = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rows = []
for i in ids:
    o = objs[i]
    sess, pub = o["key"].split("/")
    d = TS / "station" / sess
    p = cv2.imread(str(d / pub))
    nat = cv2.imread(str(d / pub.replace("_pub.jpg", "_nat.jpg")))
    x, y, w, h = o["box"]
    cv2.rectangle(p, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    cv2.putText(p, f"{i} {sess[16:]} {pub[9:15]}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    k = nat.shape[1] / 640
    cx, cy = (x + w / 2) * k, (y + h / 2) * k
    half = half_px or max(w, h) * k * 1.0 + 40
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
    z = nat[y0:y1, x0:x1].copy()
    cv2.rectangle(z, (int(x * k - x0), int(y * k - y0)), (int((x + w) * k - x0), int((y + h) * k - y0)), (0, 0, 255), 1)
    z = cv2.resize(z, (480, 480), interpolation=cv2.INTER_CUBIC)
    rows.append(np.hstack([p, z]))
cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

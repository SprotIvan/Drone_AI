"""Help place a manual drone box: crop the native frame around a rough point,
run ft_v5/ft_v7 on the crop (upscaled), and draw their boxes plus a 10 px
grid on a x3 zoom. Every box still has to be checked by eye on the zoom.
    python _manual_assist.py OUT.jpg FRAME:CX:CY[:HALF] ...   (native px)
Prints the model boxes in native and published (640x480) px.
"""
import sys
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
import cv2
import numpy as np
from ultralytics import YOLO
import station_additions as SA

frames, _ = SA.load()
models = {m: YOLO(rf"C:\Github\drone_training\v8\models\{m}.pt") for m in ("ft_v5", "ft_v7")}
tiles = []
for spec in sys.argv[2:]:
    p = spec.split(":")
    i, cx, cy = int(p[0]), int(p[1]), int(p[2])
    half = int(p[3]) if len(p) > 3 else 100
    fr = frames[i]
    nat = cv2.imread(str(SA.sess_dir(fr["session"]) / fr["nat"]))
    H, W = nat.shape[:2]
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    crop = nat[y0:y1, x0:x1].copy()
    k = 660 / (2 * half)
    z = cv2.resize(crop, (int((x1 - x0) * k), int((y1 - y0) * k)), interpolation=cv2.INTER_CUBIC)
    for gx in range((x0 // 10 + 1) * 10, x1, 10):
        c = (0, 255, 255) if gx % 50 == 0 else (90, 90, 90)
        cv2.line(z, (int((gx - x0) * k), 0), (int((gx - x0) * k), 8 if gx % 50 else 16), c, 1)
        if gx % 50 == 0:
            cv2.putText(z, str(gx), (int((gx - x0) * k) + 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    for gy in range((y0 // 10 + 1) * 10, y1, 10):
        c = (0, 255, 255) if gy % 50 == 0 else (90, 90, 90)
        cv2.line(z, (0, int((gy - y0) * k)), (8 if gy % 50 else 16, int((gy - y0) * k)), c, 1)
        if gy % 50 == 0:
            cv2.putText(z, str(gy), (18, int((gy - y0) * k) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    for cand in p[4:]:                        # a box to check, native x1,y1,x2,y2
        c1, c2, c3, c4 = (float(v) for v in cand.split(","))
        cv2.rectangle(z, (int((c1 - x0) * k), int((c2 - y0) * k)), (int((c3 - x0) * k), int((c4 - y0) * k)),
                      (0, 255, 0), 1)
    for m, col in (("ft_v5", (255, 160, 0)), ("ft_v7", (255, 0, 255))):
        if len(p) > 4:
            continue
        r = models[m].predict(crop, imgsz=640, conf=0.05, verbose=False)[0]
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            bx = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
            pk = 640 / W
            print(f"{i} {m} {s:.2f} native {[round(float(v)) for v in bx]} "
                  f"pub xywh {[round(float(v), 1) for v in (bx[0] * pk, bx[1] * pk, (bx[2] - bx[0]) * pk, (bx[3] - bx[1]) * pk)]}")
            cv2.rectangle(z, (int((bx[0] - x0) * k), int((bx[1] - y0) * k)),
                          (int((bx[2] - x0) * k), int((bx[3] - y0) * k)), col, 1)
            cv2.putText(z, f"{m[3:]} {s:.2f}", (int((bx[0] - x0) * k), int((bx[3] - y0) * k) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(z, str(i), (z.shape[1] - 60, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    tiles.append(z)
h = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
rows = [np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]
w = max(r.shape[1] for r in rows)
rows = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows]
cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

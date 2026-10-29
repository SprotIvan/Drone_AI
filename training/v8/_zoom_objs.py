"""Native-resolution zoom of EVERY model object (D, F or U) in the given
frames, to identify what a background candidate's false box is on.
    python _zoom_objs.py OUT.jpg 2328,2329,...
"""
import sys
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
import cv2
import numpy as np
import station_additions as SA

frames, byf = SA.load()
tiles = []
for i in [int(x) for x in sys.argv[2].split(",")]:
    fr = frames[i]
    nat = cv2.imread(str(SA.sess_dir(fr["session"]) / fr["nat"]))
    k = nat.shape[1] / 640
    for o in byf.get(i, []):
        (x, y, w, h), sc, m = SA.best_box(o)
        cx, cy = (x + w / 2) * k, (y + h / 2) * k
        half = max(w, h) * k * 1.5 + 30
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
        z = nat[y0:y1, x0:x1].copy()
        cv2.rectangle(z, (int(x * k - x0), int(y * k - y0)), (int((x + w) * k - x0), int((y + h) * k - y0)),
                      (0, 0, 255), 1)
        z = cv2.resize(z, (320, 320), interpolation=cv2.INTER_CUBIC)
        cv2.putText(z, f"{i} {o['lab']} {w:.0f}x{h:.0f} {sc:.2f}{m[3:]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
        tiles.append(z)
cols = 6
while len(tiles) % cols:
    tiles.append(np.zeros((320, 320, 3), np.uint8))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(tiles), "tiles")

"""Native 1320x990 station frames at half scale, 3x2 per sheet, for finding
drones no model boxed. Grid lines every 100 native px help read positions.
    python _native_sheet.py OUT_PREFIX 2460,2461,...
"""
import sys
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
import cv2
import numpy as np
import station_additions as SA

frames, _ = SA.load()
idx = [int(x) for x in sys.argv[2].split(",")]
tiles = []
for i in idx:
    fr = frames[i]
    nat = cv2.imread(str(SA.sess_dir(fr["session"]) / fr["nat"]))
    t = cv2.resize(nat, (nat.shape[1] // 2, nat.shape[0] // 2), interpolation=cv2.INTER_AREA)
    for g in range(100, nat.shape[1], 100):
        cv2.line(t, (g // 2, 0), (g // 2, 6), (0, 255, 255), 1)
    for g in range(100, nat.shape[0], 100):
        cv2.line(t, (0, g // 2), (6, g // 2), (0, 255, 255), 1)
    cv2.putText(t, f"{i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(t)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
for s in range(0, len(tiles), 6):
    rows = [np.hstack(tiles[s + j:s + j + 3]) for j in (0, 3)]
    cv2.imwrite(f"{sys.argv[1]}_{s // 6}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

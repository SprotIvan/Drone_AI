#!/usr/bin/env python3
"""
live_zoom.py — zoomed crops (from the native 1320x990 frame) around every box the station logged on its
saved frames, numbered, to tell by eye which boxes are on the drone and which are false.

    python live_zoom.py SESSION_DIR OUT_PREFIX
Writes OUT_PREFIX_NN.jpg sheets and OUT_PREFIX_index.json ([sheet tile id] -> frame, box).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def rows_of(d):
    out = []
    for l in (d / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    index, tiles = [], []
    for i, r in enumerate(rows_of(d)):
        nat = cv2.imread(str(d / r["files"]["nat"]))
        if nat is None or not r.get("boxes"):
            continue
        k = nat.shape[1] / r["pub_size"][0]
        for j, b in enumerate(r["boxes"]):
            x, y, w, h = (v * k for v in b[:4])
            cx, cy = x + w / 2, y + h / 2
            half = max(w, h) * 1.6 + 40
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
            z = nat[y0:y1, x0:x1].copy()
            cv2.rectangle(z, (int(x - x0) - 2, int(y - y0) - 2), (int(x + w - x0) + 2, int(y + h - y0) + 2), (0, 0, 255), 1)
            z = cv2.resize(z, (200, 200), interpolation=cv2.INTER_CUBIC)
            tid = len(tiles)
            cv2.putText(z, f"{tid} f{i} {b[4]:.2f} {max(b[2], b[3]):.0f}px", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1)
            tiles.append(z)
            index.append({"tile": tid, "frame": i, "pub": r["files"]["pub"], "reason": r["reason"], "box": b})
    for s in range(0, len(tiles), 48):
        t = tiles[s:s + 48]
        while len(t) % 8:
            t.append(np.zeros((200, 200, 3), np.uint8))
        cv2.imwrite(f"{out}_{s // 48:02d}.jpg", np.vstack([np.hstack(t[q:q + 8]) for q in range(0, len(t), 8)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    Path(f"{out}_index.json").write_text(json.dumps(index, indent=0))
    print(len(tiles), "boxes")


if __name__ == "__main__":
    main()

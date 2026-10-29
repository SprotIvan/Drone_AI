#!/usr/bin/env python3
"""
live_sheets.py — contact sheets of a live station session (frames_*/frames.jsonl): every saved published
frame with the boxes the station logged at save time, to see where the drone was and when it was missed.

    python live_sheets.py SESSION_DIR OUT_PREFIX [--every N]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    d = Path(a.session)
    rows = []
    for l in (d / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:                                # a session killed mid-write leaves a broken last line
            rows.append(json.loads(l))
        except ValueError:
            pass
    tiles = []
    for i, r in enumerate(rows):
        if i % a.every:
            continue
        im = cv2.imread(str(d / r["files"]["pub"]))
        if im is None:                      # file cut off when the session was killed
            print("unreadable:", r["files"]["pub"])
            continue
        for b in r.get("boxes", []):
            x, y, w, h = (int(round(v)) for v in b[:4])
            cv2.rectangle(im, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
            if len(b) > 4:
                cv2.putText(im, f"{b[4]:.2f}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
        t = datetime.fromtimestamp(r["t"]).strftime("%H:%M:%S")
        cv2.putText(im, f"#{i} {r['reason'][0].upper()} {t}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(im)
    for k in range(0, len(tiles), 36):
        t = tiles[k:k + 36]
        while len(t) % 6:
            t.append(np.zeros((240, 320, 3), np.uint8))
        cv2.imwrite(f"{a.out}_{k // 36:02d}.jpg", np.vstack([np.hstack(t[q:q + 6]) for q in range(0, len(t), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(len(rows), "frames,", len(tiles), "tiles")


if __name__ == "__main__":
    main()

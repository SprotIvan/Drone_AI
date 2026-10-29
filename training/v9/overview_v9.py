#!/usr/bin/env python3
"""
overview_v9.py — contact sheets of every Nth frame per session (published 640x480 at half size) with the
candidate boxes of ft_v5/ft_v7/ft_v8 (>= 0.30, cyan) and ft_v2 (>= 0.40, red), to see what each session
contains before labelling.

    python overview_v9.py N OUT_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAW = Path(r"C:\Github\drone_training\v9\raw_15_10")
CAND = Path(r"C:\Github\drone_training\v9\cands")


def main():
    step, out = int(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    c = {m: json.loads((CAND / f"{m}.json").read_text()) for m in ("ft_v2", "ft_v5", "ft_v7", "ft_v8")}
    keys = sorted(c["ft_v8"])
    tiles = []
    for i, key in enumerate(keys):
        if i % step:
            continue
        s, pub = key.split("/")
        im = cv2.imread(str(RAW / s / pub))
        for m, col, thr in (("ft_v2", (0, 0, 255), 0.40), ("ft_v5", (255, 255, 0), 0.30),
                            ("ft_v7", (255, 255, 0), 0.30), ("ft_v8", (255, 255, 0), 0.30)):
            for b in c[m][key]:
                if b[4] >= thr:
                    x, y, w, h = (int(round(v)) for v in b[:4])
                    cv2.rectangle(im, (x - 2, y - 2), (x + w + 2, y + h + 2), col, 1)
        im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{s[-6:]} {pub[9:17]}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(im)
    for k in range(0, len(tiles), 36):
        t = tiles[k:k + 36]
        while len(t) % 6:
            t.append(np.zeros((240, 320, 3), np.uint8))
        cv2.imwrite(str(out / f"ov_{k // 36:02d}.jpg"), np.vstack([np.hstack(t[q:q + 6]) for q in range(0, len(t), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(len(tiles), "tiles")


if __name__ == "__main__":
    main()

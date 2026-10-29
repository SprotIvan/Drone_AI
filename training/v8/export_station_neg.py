#!/usr/bin/env python3
"""
export_station_neg.py — checked drone-free station frames as negatives: per
frame the published 640x480 image + the 6 native 640x640 crop-mode windows
(smart_crop.tile_origins, as the station cuts them), each with an EMPTY label.

    python export_station_neg.py PICKED.json SESSION_DIR OUT_DIR

PICKED.json: mine_station_neg.py output (only "picked" -> "pub" is used).
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
from smart_crop import tile_origins  # noqa: E402


def main():
    picked = json.loads(Path(sys.argv[1]).read_text())["picked"]
    sess, out = Path(sys.argv[2]), Path(sys.argv[3])
    if out.exists():
        sys.exit(f"{out} exists")
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    n = 0
    for c in picked:
        pub = c["pub"]
        stem = f"st8neg_{sess.name[7:]}_{pub[:-len('_pub.jpg')]}"
        p = cv2.imread(str(sess / pub))
        nat = cv2.imread(str(sess / pub.replace("_pub.jpg", "_nat.jpg")))
        cv2.imwrite(str(out / "images" / f"{stem}_pub.jpg"), p, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / f"{stem}_pub.txt").write_text("")
        n += 1
        for i, (ox, oy) in enumerate(tile_origins((nat.shape[1], nat.shape[0]), 640)):
            cv2.imwrite(str(out / "images" / f"{stem}_tile{i}.jpg"), nat[oy:oy + 640, ox:ox + 640],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / f"{stem}_tile{i}.txt").write_text("")
            n += 1
    print(len(picked), "frames ->", n, "files ->", out)


if __name__ == "__main__":
    main()

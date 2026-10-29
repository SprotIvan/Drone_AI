#!/usr/bin/env python3
"""
preview_labels.py — contact sheets of images with their YOLO label boxes, for
review by eye. Every tile is numbered; a box is drawn green with its long side
in px. Images come from a zip (names under --prefix matching --match) or a
folder with images/ + labels/.

    python preview_labels.py OUT_PREFIX --zip DS.zip --match ua6_ --match uabg6_
    python preview_labels.py OUT_PREFIX --dir FOLDER [--sample N] [--cols 6] [--size 320]
"""
import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw(im, text, size):
    h, w = im.shape[:2]
    k = size / max(w, h)
    im2 = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    can = np.full((size, size, 3), 30, np.uint8)
    can[:im2.shape[0], :im2.shape[1]] = im2
    return can, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--zip")
    ap.add_argument("--dir")
    ap.add_argument("--prefix", default="images/")
    ap.add_argument("--match", action="append", default=[])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()
    if a.zip:
        z = zipfile.ZipFile(a.zip)
        names = sorted(n for n in z.namelist() if n.startswith(a.prefix) and Path(n).suffix.lower() in IMG
                       and (not a.match or any(m in Path(n).name for m in a.match)))
        read_img = lambda n: cv2.cvtColor(np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB")),
                                          cv2.COLOR_RGB2BGR)

        def read_lab(n):
            ln = n.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
            try:
                return z.read(ln).decode()
            except KeyError:
                return ""
    else:
        root = Path(a.dir)
        names = sorted(str(p) for p in (root / "images").rglob("*") if p.suffix.lower() in IMG
                       and (not a.match or any(m in p.name for m in a.match)))
        read_img = lambda n: cv2.imread(n)

        def read_lab(n):
            p = Path(n)
            lp = root / "labels" / p.relative_to(root / "images").with_suffix(".txt")
            return lp.read_text() if lp.exists() else ""
    if a.sample and len(names) > a.sample:
        names = sorted(random.Random(a.seed).sample(names, a.sample))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    per = a.cols * a.rows
    index = []
    for s in range(0, len(names), per):
        tiles = []
        for i, n in enumerate(names[s:s + per], start=s):
            im = read_img(n)
            can, k = draw(im, "", a.size)
            h, w = im.shape[:2]
            nb = 0
            for line in read_lab(n).splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x1, y1 = (cx - bw / 2) * w * k, (cy - bh / 2) * h * k
                x2, y2 = (cx + bw / 2) * w * k, (cy + bh / 2) * h * k
                cv2.rectangle(can, (int(x1) - 1, int(y1) - 1), (int(x2) + 1, int(y2) + 1), (0, 255, 0), 1)
                nb += 1
            cv2.rectangle(can, (0, 0), (a.size - 1, 16), (0, 0, 0), -1)
            cv2.putText(can, f"{i} {'bg' if not nb else nb} {Path(n).stem[:30]}", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            tiles.append(can)
            index.append({"i": i, "name": n, "boxes": nb})
        while len(tiles) % a.cols:
            tiles.append(np.zeros((a.size, a.size, 3), np.uint8))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{a.out}_{s // per:03d}.jpg",
                    np.vstack([np.hstack(tiles[j:j + a.cols]) for j in range(0, len(tiles), a.cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    Path(f"{a.out}_index.json").write_text(json.dumps(index, indent=0))
    print(len(names), "images ->", (len(names) + per - 1) // per, "sheets")


if __name__ == "__main__":
    main()

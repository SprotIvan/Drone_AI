#!/usr/bin/env python3
"""
label_assist.py — sheets to box drones BY EYE in an unlabelled image set.
Each tile: the image (long side 480) with a grid labelled in ORIGINAL image
px (every 100 px; ticks every 50), and ft_v5 / ft_v7 proposals (conf >= 0.05,
top 3 each, after NMS) drawn and numbered — a proposal is only a suggestion:
it is accepted, corrected or replaced by a box read off the grid.

    python label_assist.py IMAGES_DIR OUT_DIR [--per 12]
Writes OUT_DIR/la_NN.jpg and OUT_DIR/proposals.json:
  {image: [{"id": "5a", "model": "ft_v5", "box": [x1,y1,x2,y2], "score": s}, ...]}
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

T = 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    props = {p.name: [] for p in paths}
    for m, tag in (("ft_v5", "5"), ("ft_v7", "7")):
        model = YOLO(rf"C:\Github\drone_training\v8\models\{m}.pt")
        for s in range(0, len(paths), 32):
            chunk = [str(p) for p in paths[s:s + 32]]
            for src, r in zip(chunk, model.predict(chunk, imgsz=640, conf=0.05, verbose=False)):
                bb = sorted(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()),
                            key=lambda t: -t[1])[:3]
                for k, (b, c) in enumerate(bb):
                    props[Path(src).name].append({"id": f"{tag}{'abc'[k]}", "model": m,
                                                  "box": [round(v, 1) for v in b], "score": round(c, 3)})
    (out / "proposals.json").write_text(json.dumps(props, indent=0))
    tiles = []
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        k = T / max(w, h)
        im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        can = np.full((T, T, 3), 40, np.uint8)
        can[:im.shape[0], :im.shape[1]] = im
        for g in range(50, int(max(w, h)) + 1, 50):
            z = int(g * k)
            big = g % 100 == 0
            if z < im.shape[1]:
                cv2.line(can, (z, 0), (z, 9 if big else 4), (0, 255, 255), 1)
                if big:
                    cv2.putText(can, str(g), (z + 1, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            if z < im.shape[0]:
                cv2.line(can, (0, z), (9 if big else 4, z), (0, 255, 255), 1)
                if big:
                    cv2.putText(can, str(g), (11, z + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        for pr in props[p.name]:
            if pr["score"] < 0.10:
                continue
            x1, y1, x2, y2 = (int(v * k) for v in pr["box"])
            col = (255, 160, 0) if pr["model"] == "ft_v5" else (255, 0, 255)
            cv2.rectangle(can, (x1, y1), (x2, y2), col, 1)
            cv2.putText(can, f"{pr['id']}{pr['score']:.2f}", (x1 + 1, max(28, y1 - 2) if pr["model"] == "ft_v5"
                                                              else min(T - 3, y2 + 11)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
        cv2.rectangle(can, (0, T - 16), (T - 1, T - 1), (0, 0, 0), -1)
        cv2.putText(can, f"{i} {p.stem} {w}x{h}", (3, T - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(can)
    cols = 4
    for s in range(0, len(tiles), a.per):
        t = tiles[s:s + a.per]
        while len(t) % cols:
            t.append(np.zeros((T, T, 3), np.uint8))
        cv2.imwrite(str(out / f"la_{s // a.per:02d}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(len(paths), "images ->", out)


if __name__ == "__main__":
    main()

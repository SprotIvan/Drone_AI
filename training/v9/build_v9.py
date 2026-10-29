#!/usr/bin/env python3
"""
build_v9.py — dataset_v9 = dataset_v81 + the eye-checked frames of the owner's field tests (14.10 + 15.10).

    python build_v9.py          -> v9/dataset_v9/{images,labels}/{train,val}, data.yaml, STATS.md

Station frames (labels: v9/label/final_labels.json; D and H = class 0 "drone"; frames with a U object, a
flagged or unchecked frame are never used):
  positive  the published 640x480 frame + every native 640x640 window (smart_crop.tile_origins, as the station
            cuts them) that holds a whole drone box and cuts no drone box (> 15 % of a box outside = window skipped)
  negative  the published frame + the one window holding the strongest false box of ft_v5/v7/v8
            (no window if that box is cut by every window)
Train = v81 train (hard links) + train_pos_s/train_neg_s; val = v81 val (hard links) + station val frames.
The test sessions are not touched here.
"""
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
from smart_crop import tile_origins  # noqa: E402

V8 = Path(r"C:\Github\drone_training\v8")
V9 = Path(r"C:\Github\drone_training\v9")
RAW = V9 / "raw_15_10"
LAB = V9 / "label"
SRC = V8 / "dataset_v81"
OUT = V9 / "dataset_v9"
K = 1320 / 640.0


def yolo(box, w, h):
    x, y, bw, bh = box
    return f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def inside(box, ox, oy, size=640):
    """fraction of the box (native px) inside the window"""
    x, y, w, h = box
    iw = max(0.0, min(x + w, ox + size) - max(x, ox))
    ih = max(0.0, min(y + h, oy + size) - max(y, oy))
    return iw * ih / max(w * h, 1e-9)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists")
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    stats = Counter()
    # 1. v81, hard links for images, copies for labels
    for sp in ("train", "val"):
        for p in (SRC / "images" / sp).iterdir():
            os.link(p, OUT / "images" / sp / p.name)
            lp = SRC / "labels" / sp / (p.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, OUT / "labels" / sp / lp.name)
            stats[f"v81 {sp}"] += 1
    # 2. station frames
    fl = json.loads((LAB / "final_labels.json").read_text())
    labels, excluded = fl["labels"], set(fl["excluded"])
    sets = json.loads((LAB / "frame_sets.json").read_text())
    objs = json.loads((LAB / "frame_labels.json").read_text())
    groups = {"train": sets["train_pos_s"] + sets["train_neg_s"], "val": sets["val"]}
    manifest = []
    for sp, keys in groups.items():
        for k in keys:
            if k in excluded:
                continue
            sess, pub = k.split("/")
            drones = [b[:4] for b in labels[k] if b[4] in ("D", "H")]
            assert not any(b[4] == "U" for b in labels[k]), k
            stem = f"st9_{sess[-13:]}_{Path(pub).stem[:19]}"
            img = cv2.imread(str(RAW / sess / pub))
            cv2.imwrite(str(OUT / "images" / sp / f"{stem}_F.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / sp / f"{stem}_F.txt").write_text("\n".join(yolo(b, 640, 480) for b in drones) + ("\n" if drones else ""))
            kind = "pos" if drones else "neg"
            stats[f"station {sp} {kind} frames"] += 1
            stats[f"station {sp} boxes"] += len(drones)
            manifest.append({"split": sp, "file": f"{stem}_F.jpg", "src": k, "boxes": len(drones)})
            nat = cv2.imread(str(RAW / sess / pub.replace("_pub.jpg", "_nat.jpg")))
            nb = [[v * K for v in b] for b in drones]
            wins = []
            if drones:
                for j, (ox, oy) in enumerate(tile_origins((nat.shape[1], nat.shape[0]), 640)):
                    fr = [inside(b, ox, oy) for b in nb]
                    if any(0.0 < f < 0.85 for f in fr) or not any(f >= 0.85 for f in fr):
                        continue
                    wins.append((j, ox, oy, [b for b, f in zip(nb, fr) if f >= 0.85]))
            else:
                false = [o for o in objs.get(k, []) if o["lab"] == "F"
                         and max([v for m, v in o["scores"].items() if m != "ft_v2"] or [0.0]) >= 0.25]
                false.sort(key=lambda o: -max(v for m, v in o["scores"].items() if m != "ft_v2"))
                if false:
                    fb = [v * K for v in false[0]["box"]]
                    for j, (ox, oy) in enumerate(tile_origins((nat.shape[1], nat.shape[0]), 640)):
                        if inside(fb, ox, oy) >= 0.85:
                            wins.append((j, ox, oy, []))
                            break
            for j, ox, oy, bs in wins:
                win = nat[oy:oy + 640, ox:ox + 640]
                name = f"{stem}_T{j}"
                cv2.imwrite(str(OUT / "images" / sp / f"{name}.jpg"), win, [cv2.IMWRITE_JPEG_QUALITY, 95])
                loc = [[b[0] - ox, b[1] - oy, b[2], b[3]] for b in bs]
                (OUT / "labels" / sp / f"{name}.txt").write_text("\n".join(yolo(b, 640, 640) for b in loc) + ("\n" if loc else ""))
                stats[f"station {sp} {kind} windows"] += 1
                manifest.append({"split": sp, "file": f"{name}.jpg", "src": k, "window": j, "boxes": len(loc)})
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['drone']\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    lines = ["# dataset_v9", "", "v81 + eye-checked station frames of the 14.10/15.10 field tests (build_v9.py)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(stats.items())]
    (OUT / "STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

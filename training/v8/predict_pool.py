#!/usr/bin/env python3
"""
predict_pool.py — run one Ultralytics .pt model over a pool of images and store
every box >= --conf, in the image's own pixel coordinates.

    python predict_pool.py MODEL.pt OUT.json --zip DATASET.zip --prefix images/train/
    python predict_pool.py MODEL.pt OUT.json --dir FOLDER [--glob "*.jpg"]
    python predict_pool.py MODEL.pt OUT.json --list FILES.txt          (one path per line)
    python predict_pool.py MODEL.pt OUT.json --sessions FRAMES.json    (station frames:
                                       published frame + its 6 native crop windows)

Same inference for every model: imgsz 640, Ultralytics' own letterbox and
YOLO26 head, conf --conf (default 0.05), RGB as Ultralytics reads it, no TTA.
Output: {"meta": {...}, "preds": {key: [[x1, y1, x2, y2, score], ...]}}; for
--sessions the value is {"F": [...], "T0": [...], ... "T5": [...]} with window
boxes in WINDOW pixels (the caller maps them).
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def boxes_of(res):
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    sc = b.conf.cpu().numpy()
    return [[round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2),
             round(float(s), 4)] for (x1, y1, x2, y2), s in zip(xyxy, sc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip")
    src.add_argument("--dir")
    src.add_argument("--list")
    src.add_argument("--sessions")
    ap.add_argument("--prefix", default="images/")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--root", help="for --sessions: folder holding the session folders",
                    default=r"C:\Github\drone_training\station_raw")
    ap.add_argument("--root2", help="second session root (checked if the first lacks it)")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    t0 = time.time()
    preds = {}

    def run(keys, arrays):
        res = model.predict(arrays, imgsz=640, conf=a.conf, device=a.device, verbose=False,
                            batch=len(arrays))
        for k, r in zip(keys, res):
            preds[k] = boxes_of(r)

    if a.sessions:
        frames = json.loads(Path(a.sessions).read_text())
        roots = [Path(a.root)] + ([Path(a.root2)] if a.root2 else [])
        for i, fr in enumerate(frames):
            d = next((r / fr["session"] for r in roots if (r / fr["session"]).is_dir()), None)
            pub = cv2.imread(str(d / fr["pub"]))
            nat = cv2.imread(str(d / fr["nat"]))
            arrays = [pub] + [np.ascontiguousarray(nat[oy:oy + 640, ox:ox + 640]) for ox, oy in fr["origins"]]
            res = model.predict(arrays, imgsz=640, conf=a.conf, device=a.device, verbose=False, batch=len(arrays))
            preds[f"{fr['session']}/{fr['pub']}"] = {("F" if j == 0 else f"T{j - 1}"): boxes_of(r)
                                                     for j, r in enumerate(res)}
            if i % 200 == 0:
                print(f"  {i}/{len(frames)} frames", flush=True)
    else:
        if a.zip:
            z = zipfile.ZipFile(a.zip)
            names = sorted(n for n in z.namelist() if n.startswith(a.prefix)
                           and Path(n).suffix.lower() in IMG_EXT)
            load = lambda n: cv2.cvtColor(np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB")),
                                          cv2.COLOR_RGB2BGR)
        else:
            if a.dir:
                names = sorted(str(p) for p in Path(a.dir).rglob(a.glob) if p.suffix.lower() in IMG_EXT)
            else:
                names = [l.strip() for l in Path(a.list).read_text(encoding="utf-8").splitlines() if l.strip()]
            load = lambda n: cv2.imread(n)
        keys, arrays = [], []
        for i, n in enumerate(names):
            img = load(n)
            if img is None:
                preds[n] = None
                continue
            keys.append(n)
            arrays.append(img)
            if len(arrays) == a.batch:
                run(keys, arrays)
                keys, arrays = [], []
            if i % 2000 == 0:
                print(f"  {i}/{len(names)} images", flush=True)
        if arrays:
            run(keys, arrays)
    meta = {"model": str(a.model), "conf": a.conf, "imgsz": 640, "n": len(preds),
            "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "preds": preds}))
    print(json.dumps(meta))


if __name__ == "__main__":
    sys.exit(main())

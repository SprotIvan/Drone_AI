#!/usr/bin/env python3
"""
cand_v9.py — every readable frame of the 15.10 field tests through each model (.pt, on the laptop), the way
the station sees it: the published 640x480 frame + the 6 native 640x640 windows, boxes mapped to published
px and merged with smart_crop.merge_detections (station code). Everything >= 0.05 is kept: these are the
CANDIDATES that are then labelled by eye (nothing here is a label).

    python cand_v9.py              -> v9/cands/<model>.json  {"<session>/<pub>": [[x, y, w, h, score], ...]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
import eval_v8 as ev                                               # noqa: E402
from smart_crop import boxes_from_crop, merge_detections, tile_origins  # noqa: E402

RAW = Path(r"C:\Github\drone_training\v9\raw_15_10")
OUT = Path(r"C:\Github\drone_training\v9\cands")
MODELS = {"ft_v2": r"C:\Github\drone_training\v8\models\ft_v2.pt",
          "ft_v5": r"C:\Github\drone_training\v8\models\ft_v5.pt",
          "ft_v7": r"C:\Github\drone_training\v8\models\ft_v7.pt",
          "ft_v8": r"C:\Github\drone_training\v8\runs\v8A81\weights\best.pt",
          "ft_v9": r"C:\Github\drone_training\v9\runs\v9A\weights\best.pt"}
CONF = 0.05


def frames():
    sess = json.loads((RAW / "SESSIONS.json").read_text())
    out = []
    for s, info in sess.items():
        bad = set(info["unreadable"])
        for line in (RAW / s / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["files"]["pub"] not in bad:
                out.append((s, r["files"]["pub"], r["files"]["nat"]))
    return out


def main():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frames()
    names = sys.argv[1:] or list(MODELS)
    print(len(fr), "frames")
    for name, path in ((n, MODELS[n]) for n in names):
        model = YOLO(path)
        res = {}
        for i in range(0, len(fr), 64):
            chunk = fr[i:i + 64]
            pubs, wins, meta = [], [], []
            for s, pub, nat in chunk:
                p = cv2.imread(str(RAW / s / pub))
                n = cv2.imread(str(RAW / s / nat))
                can, r, l, t = ev.letterbox(p)
                pubs.append(can)
                origins = tile_origins((n.shape[1], n.shape[0]), 640)
                wins += [n[oy:oy + 640, ox:ox + 640] for ox, oy in origins]
                meta.append((f"{s}/{pub}", r, l, t, origins, (n.shape[1], n.shape[0])))
            pp = ev.run(model, pubs, CONF)
            tp = ev.run(model, wins, CONF)
            k = 0
            for (key, r, l, t, origins, nsize), boxes in zip(meta, pp):
                full = [np.array([(b[0] - l) / r, (b[1] - t) / r, (b[2] - b[0]) / r, (b[3] - b[1]) / r]) for b in boxes]
                groups = [(full, [b[4] for b in boxes])]
                sx = nsize[0] / 640.0
                for o in origins:
                    wb = [b for b in ev.nms(tp[k])]
                    k += 1
                    bb, ss = boxes_from_crop([np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]]) for b in wb],
                                             [b[4] for b in wb], tuple(o), 640, tuple(nsize), (sx, sx))
                    groups.append((bb, ss))
                mb, ms = merge_detections(groups, 0.45)
                res[key] = [[round(float(c), 2) for c in b[:4]] + [round(float(sc), 4)] for b, sc in zip(mb, ms)]
            print(f"{name}: {min(i + 64, len(fr))}/{len(fr)}", end="\r", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(res))
        n_b = sum(len(v) for v in res.values())
        print(f"\n{name}: {n_b} boxes >= {CONF} on {sum(1 for v in res.values() if v)} frames")


if __name__ == "__main__":
    main()

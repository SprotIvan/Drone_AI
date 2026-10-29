#!/usr/bin/env python3
"""
eval_v9.py — the field-test test set (sessions in CHANGELOG 16.10), every model through the station protocol
(published frame + 6 native windows, merged with smart_crop.merge_detections; cand_v9.py output).

    python eval_v9.py [MODEL ...]      default: ft_v2 ft_v5 ft_v7 ft_v8 ft_v9

Primary metric (rules 1-2): test PERIODIC frames (saved every 2 s regardless of detections; whole-frame
eye review, v9/label/final_labels.json). A labelled drone (D or H) is FOUND when a box >= thr is on it
(eval_v8.on: box centre inside the label extent + 12 px). A box on no D/H/U label is FALSE. U labels count
neither way. Thresholds 0.30 and 0.40.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
from eval_v8 import on  # noqa: E402

V9 = Path(r"C:\Github\drone_training\v9")
LAB = V9 / "label"
RUN_BY = {"001018": "ft_v5", "001621": "ft_v5", "021834": "ft_v8", "022253": "ft_v8", "024422": "ft_v8",
          "024514": "ft_v8"}


def main():
    models = sys.argv[1:] or ["ft_v2", "ft_v5", "ft_v7", "ft_v8", "ft_v9"]
    labels = json.loads((LAB / "final_labels.json").read_text())["labels"]
    keys = json.loads((LAB / "frame_sets.json").read_text())["test_periodic"]
    cands = {m: json.loads((V9 / "cands" / f"{m}.json").read_text()) for m in models}
    res = {}
    for m in models:
        for t in (0.30, 0.40):
            r = defaultdict(int)
            for k in keys:
                lab = [{"box": b[:4], "lab": b[4]} for b in labels[k]]
                boxes = [b for b in cands[m][k] if b[4] >= t]
                grp = RUN_BY[k.split("/")[0][-6:]]
                for o in lab:
                    if o["lab"] in ("D", "H"):
                        hit = any(on(b, o) for b in boxes)
                        r["drones"] += 1
                        r["found"] += hit
                        r[f"drones_{o['lab']}"] += 1
                        r[f"found_{o['lab']}"] += hit
                        r[f"drones_rec_{grp}"] += 1
                        r[f"found_rec_{grp}"] += hit
                nf = sum(1 for b in boxes if not any(on(b, o) for o in lab))
                r["false_boxes"] += nf
                r["frames_with_false"] += nf > 0
                r["frames"] += 1
            res[f"{m}@{t:.2f}"] = dict(r)
    (V9 / "eval_v9_test.json").write_text(json.dumps(res, indent=1))
    print(f"test periodic frames: {len(keys)}")
    print(f"{'model@thr':12s} {'found':>12s} {'not in hand D':>14s} {'in hand H':>12s} {'false boxes':>12s} {'frames w/ false':>16s}")
    for k, r in res.items():
        print(f"{k:12s} {r['found']:4d}/{r['drones']:<3d} ({100*r['found']/r['drones']:4.1f}%) "
              f"{r['found_D']:4d}/{r['drones_D']:<3d}     {r['found_H']:4d}/{r['drones_H']:<3d}   "
              f"{r['false_boxes']:8d}     {r['frames_with_false']:8d}/{r['frames']}")


if __name__ == "__main__":
    main()

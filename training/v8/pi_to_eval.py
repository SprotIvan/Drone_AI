#!/usr/bin/env python3
"""
pi_to_eval.py — HEF results from the Pi (compare_hef.py / run6.py image-set
preds, run_sessions_v8.py station frames) -> v8/eval/preds/<NAME>.json, the
format eval_v8.py score reads, so a HEF is scored by EXACTLY the code that
scored the .pt models.

    python pi_to_eval.py PI_OUT_DIR HEF_STEM NAME

PI_OUT_DIR holds <set>.<HEF_STEM>.rgb.preds.json for each image set (run6.py
--out <set>.json) and sess.<HEF_STEM>.json (run_sessions_v8.py --out sess).
Boxes there are (x, y, w, h) in the source image's px (HailoInference
un-letterboxes them); here they become [x1, y1, x2, y2, score].
"""
import json
import sys
from pathlib import Path

V8 = Path(r"C:\Github\drone_training\v8")


def xyxy(boxes, scores):
    return [[b[0], b[1], b[0] + b[2], b[1] + b[3], s] for b, s in zip(boxes, scores)]


def main():
    pi, stem, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sets = {}
    for f in sorted(pi.glob(f"*.{stem}.rgb.preds.json")):
        s = f.name.split(".")[0]
        sets[s] = {Path(k).name: xyxy(*v) for k, v in json.loads(f.read_text()).items()}
        print(s, len(sets[s]))
    sf = pi / f"sess.{stem}.json"
    if sf.exists():
        d = json.loads(sf.read_text())
        st = {}
        frames = {f"{fr['session']}/{fr['pub']}": fr for fr in
                  json.loads((V8 / "pi_v8" / "frames_v8.json").read_text())}
        for k, r in d["preds"].items():
            st[k] = {"full": xyxy(*r["F"]), "wins": [xyxy(*r[f"T{j}"]) for j in range(6)],
                     "origins": frames[k]["origins"], "native": [1320, 990]}
        sets["station"] = st
        print("station", len(st), d["meta"])
    (V8 / "eval" / "preds").mkdir(parents=True, exist_ok=True)
    (V8 / "eval" / "preds" / f"{name}.json").write_text(json.dumps({"model": stem, "source": "Pi HEF", "sets": sets}))


if __name__ == "__main__":
    main()

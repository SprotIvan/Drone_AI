#!/usr/bin/env python3
"""
verify_weights.py — are the ft_v8 checkpoints intact and from the runs they
claim to be? Per file: size, md5, torch.load, the run arguments stored inside
(start weights, data, epochs, patience), the last epoch in the stored training
record, parameter count, and a prediction on 4 known test images (2 with a
drone, 2 without) to see the network actually works.

    python verify_weights.py RUN_DIR [RUN_DIR ...]
"""
import hashlib
import sys
from pathlib import Path

import torch

TESTS = [r"C:\Github\drone_training\v8\testsets\drone_close\images",
         r"C:\Github\drone_training\v8\testsets\coco_neg\images"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    from ultralytics import YOLO
    imgs = []
    for d in TESTS:
        imgs += sorted(Path(d).iterdir())[:2]
    for run in sys.argv[1:]:
        run = Path(run)
        sd = {}
        for name in ("best.pt", "last.pt"):
            p = run / "weights" / name
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a = ck.get("train_args", {})
            tr = ck.get("train_results") or {}
            ep = tr.get("epoch", [])
            last_ep = int(ep[-1]) if len(ep) else None
            fit = None
            if len(ep):
                fit = 0.1 * tr["metrics/mAP50(B)"][-1] + 0.9 * tr["metrics/mAP50-95(B)"][-1]
            m = ck["model"]
            sd[name] = {k: v.float() for k, v in m.state_dict().items()}
            print(f"{run.name}/{name}: {p.stat().st_size} B md5 {md5(p)} | ultralytics {ck.get('version')} "
                  f"date {ck.get('date', '')[:19]}")
            print(f"   start {Path(str(a.get('model'))).name} data {Path(str(a.get('data'))).parent.name} "
                  f"epochs {a.get('epochs')} patience {a.get('patience')} name {a.get('name')} | "
                  f"record ends at epoch {last_ep} (fitness {fit:.4f}) | params {sum(t.numel() for t in sd[name].values())}")
            y = YOLO(str(p))
            res = y.predict([str(i) for i in imgs], imgsz=640, conf=0.25, verbose=False)
            print("   predict:", " | ".join(f"{Path(i).parent.parent.name}/{Path(i).name[:22]}: "
                                           f"{len(r.boxes)} box(es) {[round(float(c), 2) for c in r.boxes.conf][:3]}"
                                           for i, r in zip(imgs, res)))
        same = all(torch.equal(sd["best.pt"][k], sd["last.pt"][k]) for k in sd["best.pt"])
        print(f"   best.pt weights == last.pt weights: {same}")


if __name__ == "__main__":
    main()

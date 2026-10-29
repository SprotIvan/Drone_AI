#!/usr/bin/env python3
"""
train_v8.py — ft_v8 training on drone_dataset_v8 with the ft_v5 recipe
(train_colab_v5.ipynb / ft_v5 checkpoint train_args), changing only the data
and, for the control run, the starting weights:

    python train_v8.py A     start from ft_v5  -> runs/v8A   (the ft_v8 candidate)
    python train_v8.py B     start from ft_v7  -> runs/v8B   (control: same data, same protocol)
    python train_v8.py A81   start from ft_v5  -> runs/v8A81 (v8.1: dataset_v81, recipe of A)

Recipe (= ft_v5): imgsz 640, epochs 25, batch 32, optimizer auto (Ultralytics 8.4.66 picks
MuSGD lr 0.01 momentum 0.9 for > 10k iterations - as it did for ft_v5), cos_lr, warmup 3, close_mosaic 10,
mosaic 1.0, scale 0.5, translate 0.1, fliplr 0.5, hsv 0.015/0.7/0.4, seed 0,
deterministic, AMP. best.pt = best val fitness (Ultralytics default).
Local differences: workers (Windows laptop RAM), nothing else in the optimisation.

ONE CHANGE vs ft_v5, patience 8 -> 25 (= epochs, i.e. no early stop inside the budget):
the first run with patience 8 (runs/v8A_patience8) stopped after epoch 9 with
best = epoch 1 - the warm-up raises lr to 0.01 and val mAP50 dips (0.929 -> 0.834
at epoch 3) for longer than 8 epochs, so best.pt was ft_v5 + one warm-up epoch.
ft_v5's own run peaked at its LAST epoch (25).
"""
import json
import sys
import time
from pathlib import Path

V8 = Path(r"C:\Github\drone_training\v8")
START = {"A": V8 / "models" / "ft_v5.pt", "B": V8 / "models" / "ft_v7.pt", "A81": V8 / "models" / "ft_v5.pt"}
DATA = {"A": "dataset_v8", "B": "dataset_v8", "A81": "dataset_v81"}   # A81 = v8.1: same recipe as A, data build_v81.py
RECIPE = dict(imgsz=640, epochs=25, patience=25, batch=32, optimizer="auto", lr0=0.01, lrf=0.01,
              momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, cos_lr=True, close_mosaic=10,
              mosaic=1.0, mixup=0.0, copy_paste=0.0, scale=0.5, translate=0.1, degrees=0.0, fliplr=0.5,
              hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, seed=0, deterministic=True, amp=True, cache=False)


def main():
    from ultralytics import YOLO
    run = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cache = sys.argv[3] if len(sys.argv) > 3 else False   # 'disk': decoded .npy next to the images,
    #   same pixels as decoding the JPEG every time (load_image resizes either way) - only faster
    out = V8 / "runs"
    out.mkdir(exist_ok=True)
    meta = {"run": run, "start": str(START[run]), "data": str(V8 / DATA[run] / "data.yaml"),
            "recipe": RECIPE, "workers": workers, "cache": cache, "t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"v8{run}_meta.json").write_text(json.dumps(meta, indent=1))
    model = YOLO(str(START[run]))
    model.train(data=meta["data"], project=str(out), name=f"v8{run}", exist_ok=False, workers=workers,
                **{k: v for k, v in RECIPE.items() if k != "cache"}, cache=cache,
                plots=True)
    meta["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / f"v8{run}_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

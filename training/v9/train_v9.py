#!/usr/bin/env python3
"""
train_v9.py — ft_v9 candidates on dataset_v9 with the v8.1 recipe unchanged (train_v8.RECIPE).

    python train_v9.py A [workers]     start from ft_v8 (runs/v8A81/weights/best.pt) -> v9/runs/v9A
    python train_v9.py B [workers]     start from ft_v5                                -> v9/runs/v9B (only if A fails)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
from train_v8 import RECIPE  # noqa: E402

V8 = Path(r"C:\Github\drone_training\v8")
V9 = Path(r"C:\Github\drone_training\v9")
START = {"A": V8 / "runs" / "v8A81" / "weights" / "best.pt", "B": V8 / "models" / "ft_v5.pt"}


def main():
    from ultralytics import YOLO
    run = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = V9 / "runs"
    out.mkdir(exist_ok=True)
    meta = {"run": run, "start": str(START[run]), "data": str(V9 / "dataset_v9" / "data.yaml"),
            "recipe": RECIPE, "workers": workers, "t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"v9{run}_meta.json").write_text(json.dumps(meta, indent=1))
    model = YOLO(str(START[run]))
    model.train(data=meta["data"], project=str(out), name=f"v9{run}", exist_ok=False, workers=workers,
                **{k: v for k, v in RECIPE.items() if k != "cache"}, cache=False, plots=True)
    meta["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / f"v9{run}_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

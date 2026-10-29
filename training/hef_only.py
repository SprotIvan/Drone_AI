#!/usr/bin/env python3
"""
hef_only.py — just the HEF part of train_colab.ipynb (calibration set, the
DFC venv, both compiles) from three files, without the dataset or the
training run. For a runtime that cannot see the training account's Drive:
another Google account's Colab (hef_colab.ipynb), or a Kaggle kernel.

    python hef_only.py --inputs DIR --out DIR [--work DIR] [--recipes logits]

--inputs  searched recursively for
            hailo_dataflow_compiler-*.whl   the DFC 3.33.1 wheel
            best.onnx                       runs/ft_v2/weights on the Drive
                                            the model was trained on
            calib_src.zip                   the 1024 calibration images picked
                                            by `build_calib.py --pick-only`,
                                            with labels and calib_pick.txt
                                            (or that zip already unpacked)
          build_calib.py and compile_hef.py are taken from there too, or else
          from this script's own folder
--out     bestty_yolo26_<stamp>_<recipe>.hef, the quantized .har,
          compile_summary_*.json, the DFC logs. A recipe whose HEF is already
          there is skipped, so a rerun after a disconnect carries on.
--work    scratch: the unpacked images, the 5 GB calibration set, the venv
--recipes compile_hef.py recipes to build, in this order; only `logits` by
          default — the `sigmoid` recipe's fine-tune diverges. `zoo_fast` /
          `zoo` are Hailo's own layout for yolo26 (see compile_hef.py): put
          zoo_fast FIRST, so its HEF is on the Drive even if zoo's AdaRound
          runs out of memory or time
--onnx-md5 refuse to start unless best.onnx has this md5 (the Drive folder
          has held two different models' best.onnx)
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find(root, pattern, fallback=None):
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if hits:
        return hits[0]
    if fallback and Path(fallback).exists():
        return str(fallback)
    sys.exit(f"not found under {root}: {pattern}")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/hef_only")
    ap.add_argument("--recipes", nargs="+", default=["logits"],
                    choices=["logits", "sigmoid", "zoo_fast", "zoo"])
    ap.add_argument("--onnx-md5")
    args = ap.parse_args()
    inp, out, work = Path(args.inputs), Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    gpu = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        shell=True, capture_output=True, text=True)
    print("GPU:", gpu.stdout.strip() or gpu.stderr.strip(), flush=True)
    if gpu.returncode != 0:
        sys.exit("no GPU on this runtime: the compile would lose the "
                 "quantization-aware fine-tune the current HEF has")

    todo = [r for r in args.recipes if not glob.glob(f"{out}/*_{r}.hef")]
    print("HEF still to build:", todo or "none", flush=True)
    if not todo:
        return

    wheel = find(inp, "hailo_dataflow_compiler-*.whl")
    onnxes = sorted(glob.glob(f"{inp}/**/best.onnx", recursive=True))
    if len(onnxes) > 1:
        sys.exit(f"more than one best.onnx under {inp} - keep only the "
                 f"model to compile: {onnxes}")
    onnx = find(inp, "best.onnx")
    if args.onnx_md5:
        md5 = hashlib.md5(Path(onnx).read_bytes()).hexdigest()
        if md5 != args.onnx_md5:
            sys.exit(f"{onnx} has md5 {md5}, not {args.onnx_md5}: "
                     "this is not the model this run is meant to compile")
        print("best.onnx md5 OK:", md5, flush=True)
    build_calib = find(inp, "build_calib.py", HERE / "build_calib.py")
    compile_hef = find(inp, "compile_hef.py", HERE / "compile_hef.py")

    picks = glob.glob(f"{inp}/**/calib_pick.txt", recursive=True)
    if picks:                                   # the zip came unpacked
        src = Path(picks[0]).parent
    else:
        src = work / "calib_src"
        if not (src / "calib_pick.txt").exists():
            with zipfile.ZipFile(find(inp, "calib_src.zip")) as z:
                z.extractall(src)
    calib = work / "calib"
    if len(list(calib.glob("*.npy"))) < 1024:
        sh(f'"{sys.executable}" "{build_calib}" "{src}" "{calib}" '
           f'--pick "{src}/calib_pick.txt"')

    # The DFC in its own Python 3.10 venv: it pins TensorFlow 2.18 and a
    # matplotlib (3.5.2) whose wheels stop at Python 3.10.
    vpy = work / "hailo_venv" / "bin" / "python"
    uv = f'"{sys.executable}" -m uv'
    sh(f'"{sys.executable}" -m pip -q install uv')
    sh(f'{uv} venv -q --allow-existing --python 3.10 "{work}/hailo_venv"')
    sh(f'{uv} pip install -q --python "{vpy}" "{wheel}" psutil')
    tf = subprocess.check_output(
        [str(vpy), "-c", "import tensorflow as t; print(t.__version__)"],
        stderr=subprocess.DEVNULL, text=True).strip()
    print("DFC pins TensorFlow", tf, "- adding its CUDA libraries", flush=True)
    sh(f'{uv} pip install -q --python "{vpy}" "tensorflow[and-cuda]=={tf}"')

    # Named explicitly: an inherited empty CUDA_VISIBLE_DEVICES hides the GPU.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    failed = []
    for recipe in todo:
        r = subprocess.run([str(vpy), compile_hef, "--recipe", recipe,
                            "--onnx", onnx, "--calib", str(calib),
                            "--out", str(out),
                            "--work", str(work / f"hef_work_{recipe}")],
                           env=env)
        print(f"== {recipe}: exit {r.returncode}", flush=True)
        if r.returncode:
            failed.append(recipe)
    sh(f'ls -la "{out}"')
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()

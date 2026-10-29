#!/usr/bin/env python3
"""
make_ft_v8_package.py — the self-contained folder for compiling a model to
a Hailo-8L HEF in Google Colab (.pt -> ONNX -> HEF), for the owner to upload.

    python make_ft_v8_package.py RUN_DIR DATASET_DIR OUT_DIR --report REPORT.md [--model-name ft_v8]

RUN_DIR      Ultralytics run (weights/best.pt, args.yaml)
DATASET_DIR  the dataset the run trained on (images/train + labels/train): the
             1024 calibration images are picked from ITS train split
             (build_calib.py --pick-only: stratified by drone size, <= 10 %
             backgrounds) — never from val or test
OUT_DIR      must not exist

Runs colab_export_verify.py on THIS machine first (stages 1-3 for real; its
ONNX md5 goes into samples_expected.json so Colab can say whether its export is
byte-identical). The DFC wheel is HARD-LINKED (no second 512 MB copy on disk).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
WHEEL = Path(r"C:\Github\drone_training\drone_hef\hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl")
PINS = ["ultralytics==8.4.66", "onnx==1.23.2", "onnxslim==0.1.98", "onnxruntime==1.23.2"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(model_name, pt_md5, folder=None):
    """folder: the Drive folder name the package is uploaded as (default: model_name)."""
    folder = folder or model_name
    F = f"/content/drive/MyDrive/{folder}"
    cells = [
        cell("markdown", f"""
# {model_name}: `best.pt` → ONNX → HEF (Hailo-8L)

Folder **`MyDrive/{folder}/`** (uploaded whole from the laptop) → this notebook →
**`MyDrive/{folder}/out/`**: `best.onnx`, `onnx_check.json`, and in `out/hef/` the
`bestty_yolo26_<date>_zoo_fast.hef` with its quantized `.har`, `compile_summary_zoo_fast.json`
and the compiler logs.

`Runtime → Change runtime type → T4 GPU`, then **Runtime → Run all**. Every stage
stops the run if its check fails. About 45–70 min in total. If the session disconnects,
reconnect (T4 GPU) and **Run all** again: finished stages are not repeated where avoidable
(cell 5 finishes from a saved quantized HAR; it skips when a HEF is already there).

| stage | cell | what is proven |
|---|---|---|
| 0 | 1–2 | GPU present; Drive folder complete |
| 1 | 4 | `best.pt` is this model (md5 `{pt_md5}`) and predicts as on the laptop |
| 2–3 | 4 | ONNX exported exactly as before (opset 11, simplify, 640×640); the 6 head convolutions the compiler needs exist; ONNX = `.pt` |
| 4–5 | 5 | 1024 calibration images → Hailo Dataflow Compiler **3.33.1** (Python 3.10 venv), recipe `zoo_fast`, target `hailo8l` |
| 6 | 6 | the HEF exists, its size and md5; the optimizer ran its quantization-aware fine-tune and did not lower its level |
"""),
        cell("code", """
import subprocess, sys, platform
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True)
print(r.stdout or r.stderr)
print('Python', sys.version.split()[0], '|', platform.platform())
assert r.returncode == 0, ('NO GPU on this runtime: Runtime -> Change runtime type -> T4 GPU. '
                           'If Colab offers only Pro, this account is out of free GPU time - use another account.')
"""),
        cell("code", f"""
from google.colab import drive
drive.mount('/content/drive')
import glob, hashlib, json, os, shutil, zipfile
PT_MD5 = '{pt_md5}'
F = '{F}'
if not os.path.isfile(f'{{F}}/best.pt'):
    # Drive may have renamed the folder (e.g. '{folder} (1)'), or it went one level deeper:
    # find it by the md5 of its best.pt
    found = [os.path.dirname(p) for p in sorted(glob.glob('/content/drive/MyDrive/*/best.pt')
                                                + glob.glob('/content/drive/MyDrive/*/*/best.pt'))
             if hashlib.md5(open(p, 'rb').read()).hexdigest() == PT_MD5]
    assert found, f'{{F}} not found and no folder in My Drive holds this best.pt - upload the whole folder {folder} into My Drive'
    F = found[0]
    print('folder found by the md5 of best.pt:', F, ('| also: ' + str(found[1:])) if len(found) > 1 else '')
need = ['best.pt', 'samples_expected.json', 'calib_src.zip', 'scripts/build_calib.py',
        'scripts/compile_hef.py', 'scripts/hef_only.py', 'scripts/colab_export_verify.py']
missing = [n for n in need if not os.path.exists(f'{{F}}/{{n}}')]
wheels = glob.glob(f'{{F}}/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl')
assert not missing, f'missing in {{F}}: {{missing}}'
assert hashlib.md5(open(f'{{F}}/best.pt', 'rb').read()).hexdigest() == PT_MD5, 'best.pt in ' + F + ' is not this model'
assert wheels, 'hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl missing in ' + F
OUT = f'{{F}}/out'
os.makedirs(OUT, exist_ok=True)
def run(cmd, env=None):
    # a shell command whose output streams into the cell; returns its exit code
    rc = '/content/_rc.txt'
    get_ipython().system(cmd + f' ; echo $? > {{rc}}')
    return int(open(rc).read().strip())
print('folder OK:', F)
"""),
        cell("code", "\n".join([f"!pip -q install {' '.join(PINS)}",
                                "import ultralytics, onnx, onnxruntime",
                                "print('ultralytics', ultralytics.__version__, '| onnx', onnx.__version__, "
                                "'| onnxruntime', onnxruntime.__version__)"])),
        cell("code", """
# stages 1-3: verify best.pt, export ONNX, verify ONNX (own process - see the script's docstring)
exp = json.load(open(f'{F}/samples_expected.json'))
S = '/content/samples'
shutil.rmtree(S, ignore_errors=True); os.makedirs(S)
with zipfile.ZipFile(f'{F}/calib_src.zip') as z:
    for short, s in exp['samples'].items():
        with open(f'{S}/{short}', 'wb') as f:
            f.write(z.read(f'images/train/' + s['src']))
rc = run(f'python {F}/scripts/colab_export_verify.py --pt {F}/best.pt --expected {F}/samples_expected.json '
         f'--samples {S} --out {OUT}')
assert rc == 0, 'stages 1-3 failed - see the messages above; nothing was compiled'
print(json.dumps(json.load(open(f'{OUT}/onnx_check.json')), indent=1)[:2000])
"""),
        cell("code", """
# stages 4-5: calibration set + Hailo compile (hef_only.py: Python 3.10 venv for DFC 3.33.1 + its TensorFlow).
# After a disconnect just Run all again: if out/hef/ holds the quantized HAR of an interrupted run
# (optimization finished, compile not), the compile is finished from that HAR - no GPU, no calibration.
H = f'{OUT}/hef'
hefs = sorted(glob.glob(f'{H}/*_zoo_fast.hef'))
hars = sorted(glob.glob(f'{H}/*_zoo_fast_quantized.har'))
if hefs:
    print('HEF already in out/hef - nothing to compile:', [os.path.basename(x) for x in hefs])
elif hars:
    print('resuming from the saved HAR:', os.path.basename(hars[-1]))
    V = '/content/hef_tmp/hailo_venv/bin/python'
    for c in ('python -m pip -q install uv',
              'python -m uv venv -q --allow-existing --python 3.10 /content/hef_tmp/hailo_venv',
              f'python -m uv pip install -q --python {V} "{wheels[0]}" psutil',
              f'{V} {F}/scripts/compile_hef.py --from-har "{hars[-1]}" --out {H} --recipe zoo_fast'):
        rc = run(c)
        assert rc == 0, (f'failed: {c}\\nIf the HAR itself is damaged (the session died while saving it), '
                         'delete it from out/hef and Run all again - that redoes the whole compile.')
else:
    INP = '/content/hef_inputs'
    shutil.rmtree(INP, ignore_errors=True); os.makedirs(INP)
    shutil.copy(f'{OUT}/best.onnx', f'{INP}/best.onnx')
    os.symlink(wheels[0], f'{INP}/' + os.path.basename(wheels[0]))
    os.symlink(f'{F}/calib_src.zip', f'{INP}/calib_src.zip')
    for s in ('build_calib.py', 'compile_hef.py'):
        shutil.copy(f'{F}/scripts/{s}', f'{INP}/{s}')
    onnx_md5 = hashlib.md5(open(f'{INP}/best.onnx', 'rb').read()).hexdigest()
    rc = run(f'CUDA_VISIBLE_DEVICES=0 python {F}/scripts/hef_only.py --inputs {INP} --out {H} '
             f'--work /content/hef_tmp --recipes zoo_fast --onnx-md5 {onnx_md5}')
    assert rc == 0, ('compile failed - see the log above. If the session died after the optimization '
                     '(out/hef/ holds a *_zoo_fast_quantized.har), Run all again: this cell then finishes from the HAR')
"""),
        cell("code", """
# stage 6: what was produced
hefs = sorted(glob.glob(f'{OUT}/hef/*_zoo_fast.hef'))
assert hefs, 'no HEF in out/hef'
h = hefs[-1]
size = os.path.getsize(h)
print('HEF :', h)
print('size:', size, 'B (ft_v5 / ft_v7 HEFs of the same recipe: about 11.5 MB)')
print('md5 :', hashlib.md5(open(h, 'rb').read()).hexdigest())
s = json.load(open(f'{OUT}/hef/compile_summary_zoo_fast.json'))
print(json.dumps({k: s.get(k) for k in ('hef', 'dfc', 'calibration_images', 'calibration_as', 'optimization_level',
                                        'optimization_level_reduced', 'qft_lines', 'optimize_min', 'total_min')}, indent=1))
assert not s.get('optimization_level_reduced'), 'the optimizer LOWERED its level - do not use this HEF'
assert s.get('qft_lines'), 'no quantization-aware fine-tune in the log - do not use this HEF'
assert 5e6 < size < 3e7, 'HEF size far from the expected ~11.5 MB - inspect before use'
print('\\nOK. Download out/hef/ - next steps in README.md (copy to the Pi, test before the station uses it).')
for f in sorted(os.listdir(f'{OUT}/hef')):
    print(f'  {f}  {os.path.getsize(f"{OUT}/hef/{f}")} B')
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--report", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--model-name", default="ft_v8")
    a = ap.parse_args()
    run, ds, out = Path(a.run), Path(a.dataset), Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists")
    (out / "scripts").mkdir(parents=True)
    (out / "laptop_check").mkdir()
    shutil.copy2(run / "weights" / "best.pt", out / "best.pt")
    pt_md5 = md5(out / "best.pt")
    (out / "best.pt.md5").write_text(pt_md5 + "\n")
    shutil.copy2(run / "args.yaml", out / "args.yaml")
    import torch
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    (out / "model.yaml").write_text(yaml.safe_dump(dict(ck["model"].yaml), sort_keys=False))
    (out / "data.yaml").write_text("# class list the model was trained with (export needs only best.pt)\n"
                                   "nc: 1\nnames: ['drone']\n")
    # calibration pick from the run's own train split
    tmp = out / "_calib_tmp"
    subprocess.run([sys.executable, str(TRAINING / "build_calib.py"), str(ds), str(tmp), "--pick-only"], check=True)
    pick = (tmp / "calib_pick.txt").read_text().splitlines()
    with zipfile.ZipFile(out / "calib_src.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(tmp / "calib_pick.txt", "calib_pick.txt")
        for line in pick:
            _, name = line.split("\t")
            z.write(ds / "images" / "train" / name, f"images/train/{name}")
            z.write(ds / "labels" / "train" / (Path(name).stem + ".txt"), f"labels/train/{Path(name).stem}.txt")
    shutil.rmtree(tmp)
    # 12 sample images from the calibration pick (= train split): 10 where the model is confident of a
    # drone (a box >= 0.5, so the ONNX-vs-.pt check compares real boxes) of every size, 2 backgrounds
    sys.path.insert(0, str(HERE))
    import colab_export_verify as C
    from ultralytics import YOLO
    model = YOLO(str(out / "best.pt"))
    want = {"<16": 2, "16-32": 3, "32-64": 2, ">=64": 3, "background": 2}
    samples = []
    for line in pick:
        why, name = line.split("\t")
        if want.get(why, 0) <= 0:
            continue
        if why != "background" and not any(b[4] >= 0.5 for b in
                                             C.pt_boxes(model, cv2.imread(str(ds / "images" / "train" / name)))):
            continue
        want[why] -= 1
        samples.append(name)
    sdir = out / "laptop_check" / "samples"
    sdir.mkdir()
    exp = {"pt_md5": pt_md5, "samples": {}}
    for k, n in enumerate(samples):           # short local names: Windows MAX_PATH
        short = f"sample_{k}{Path(n).suffix.lower()}"
        shutil.copy2(ds / "images" / "train" / n, sdir / short)
        exp["samples"][short] = {"src": n, "boxes": [[round(v, 2) for v in b[:4]] + [round(b[4], 4)] for b in
                                                      C.pt_boxes(model, cv2.imread(str(sdir / short)))]}
    (out / "samples_expected.json").write_text(json.dumps(exp, indent=1))
    # stages 1-3 for real, on this machine
    r = subprocess.run([sys.executable, str(HERE / "colab_export_verify.py"), "--pt", str(out / "best.pt"),
                        "--expected", str(out / "samples_expected.json"), "--samples", str(sdir),
                        "--out", str(out / "laptop_check")])
    if r.returncode:
        sys.exit("stages 1-3 FAILED on the laptop - package not finished")
    exp["laptop_onnx_graph_md5"] = (out / "laptop_check" / "best.onnx.graph_md5").read_text().strip()
    (out / "samples_expected.json").write_text(json.dumps(exp, indent=1))
    (out / "laptop_check" / "best.onnx").rename(out / "laptop_check" / "best_laptop.onnx")
    (out / "laptop_check" / "best_cut_heads.onnx").unlink()
    for s in ("build_calib.py", "compile_hef.py", "hef_only.py"):
        shutil.copy2(TRAINING / s, out / "scripts" / s)
    shutil.copy2(HERE / "colab_export_verify.py", out / "scripts" / "colab_export_verify.py")
    (out / "requirements_export.txt").write_text("\n".join(PINS) + "\n# DFC: hailo_dataflow_compiler-3.33.1 "
                                                 "(wheel in this folder), its own Python 3.10 venv - hef_only.py\n")
    os.link(WHEEL, out / WHEEL.name)
    (out / f"{a.model_name}_colab.ipynb").write_text(json.dumps(notebook(a.model_name, pt_md5), indent=1),
                                                     encoding="utf-8")
    shutil.copy2(a.report, out / "REPORT.md")
    shutil.copy2(a.readme, out / "README.md")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"package -> {out} ({total / 2**20:.0f} MB incl. the 488 MB wheel)")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  {f.stat().st_size} B")


if __name__ == "__main__":
    main()

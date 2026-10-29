#!/usr/bin/env python3
"""Writes train_colab.ipynb and hef_colab.ipynb (kept as code so the
notebooks stay reviewable)."""
import json
from pathlib import Path

CELLS = []
HERE = Path(__file__).resolve().parent
# Embedded whole, so the notebook needs nothing but Drive.
BUILD_CALIB = (HERE / "build_calib.py").read_text(encoding="utf-8")
COMPILE_HEF = (HERE / "compile_hef.py").read_text(encoding="utf-8")


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Fine-tune the station's YOLO26n on the merged drone dataset — round 5 (ft_v5)

**Before you run:** `Runtime → Change runtime type → GPU (T4)`.

Put these in Google Drive, folder **`MyDrive/drone_training/`**:
- `drone_dataset_v5.zip` — built by `training/build_dataset.py`: v3 plus
  frames from the station's own cameras (`training/station_frames.py`):
  a Mavic flown in front of the station (143 frames, every drone box checked
  by eye) and only the HARD backgrounds — the windows where ft_v2 boxed
  something that is not a drone (wires, a pole lamp, car lights, window
  frames). Round 4 had ~2 960 station backgrounds against 91 drone images and
  lost 8 points of recall (06.10.2026); the easy ones are left out here.
- `best_ft_v2.pt` — the ft_v2 model (`runs/ft_v2/weights/best.pt`, renamed):
  round 5 starts from it, as rounds 3 and 4 did
- `hailo_dataflow_compiler-3.33.1-….whl` — the Hailo Dataflow Compiler wheel
  (the version the current HEF was built with; the Pi runs HailoRT 4.23.0)

Everything the training writes goes straight to Drive (`drone_training/runs/`),
so a Colab disconnect loses nothing — see the **Resume** cell. The HEF part at
the end can run in a later session: it reads everything from Drive.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/drone_training'
DATA_ZIP = f'{DRIVE}/drone_dataset_v5.zip'
BASE_PT = f'{DRIVE}/best_ft_v2.pt'
RUNS = f'{DRIVE}/runs'
RUN_NAME = 'ft_v5'
import os
assert os.path.exists(DATA_ZIP), f'upload {DATA_ZIP}'
assert os.path.exists(BASE_PT), f'upload {BASE_PT}'
""")

md("""
Same Ultralytics version the station's models were trained with (8.4.66),
so nothing about the model format changes.
""")

code("""
!pip -q install ultralytics==8.4.66
import ultralytics; ultralytics.checks()
""")

md("Unzip to the Colab disk (much faster to read than Drive).")

code("""
!rm -rf /content/drone_dataset && mkdir -p /content/drone_dataset
!unzip -q "$DATA_ZIP" -d /content/drone_dataset
DATA_YAML = '/content/drone_dataset/data.yaml'
# The dataset root must be the folder of data.yaml. Ultralytics resolves a
# relative 'path:' against the CURRENT directory, so drop any such line.
import re
_y = open(DATA_YAML).read()
open(DATA_YAML, 'w').write(re.sub(r'(?m)^path:.*\\n', '', _y))
!cat /content/drone_dataset/data.yaml
!for s in train val; do echo "$s: $(ls /content/drone_dataset/images/$s | wc -l) images"; done
!cat /content/drone_dataset/SOURCES.md
""")

md("A look at random training images with their boxes — check the labels make sense.")

code("""
import random, glob, cv2
import matplotlib.pyplot as plt
imgs = random.sample(glob.glob('/content/drone_dataset/images/train/*.jpg'), 12)
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
for ax, p in zip(axes.flat, imgs):
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
    lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    for row in open(lab).read().strip().splitlines():
        _, cx, cy, bw, bh = map(float, row.split())
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        cv2.rectangle(im, (x0, y0), (int(x0 + bw * w), int(y0 + bh * h)), (255, 0, 0), max(1, w // 300))
    ax.imshow(im); ax.set_title(p.split('/')[-1][:40], fontsize=8); ax.axis('off')
plt.tight_layout()
""")

md("""
## Baseline — the CURRENT model on the NEW validation set
This is the honest starting point. The model's own training report said
mAP50 0.994; at the station it scored 0.04 on a drone. This number is what
the fine-tune has to beat.
""")

code("""
from ultralytics import YOLO
base = YOLO(BASE_PT)
base_metrics = base.val(data=DATA_YAML, imgsz=640, batch=32, plots=False)
print('BASELINE  mAP50 %.3f  mAP50-95 %.3f  P %.3f  R %.3f' % (
    base_metrics.box.map50, base_metrics.box.map, base_metrics.box.mp, base_metrics.box.mr))
""")

md("## Smoke test — 1 epoch on 5 % of the data (a minute), before the long run")

code("""
import os, time
LAST = f'{RUNS}/{RUN_NAME}/weights/last.pt'
if os.path.exists(LAST):
    print('the run already exists - smoke test skipped')
else:
    t0 = time.time()
    YOLO(BASE_PT).train(data=DATA_YAML, epochs=1, imgsz=640, batch=32, fraction=0.05,
                        project='/content/smoke', name='smoke', exist_ok=True, plots=False)
    print(f'smoke test: {(time.time() - t0) / 60:.1f} min')
""")

md("""
## Fine-tune from the ft_v2 model
Same image size (640) and architecture (YOLO26n), so the Hailo compile path
does not change. ft_v2 already knows the drone data; round 5 mostly has to
learn the station's scene, so 25 epochs with `patience=8` (one epoch took
~9 min on a Colab T4 on 25.09.2026 with the v2 set).

The BEFORE/AFTER below is on the whole val split. 614 of its 2 009 images
were in ft_v2's own training set (the split moved between rounds), so
BEFORE is ft_v2 partly on images it has seen: the fair comparison is made on
the Pi afterwards, on the 1 395 images neither model has seen.

**Safe to re-run, and after a disconnect just Run all again**: the cell
continues an unfinished run from `last.pt`, skips a finished one, and
starts fresh only when there is no run yet.
""")

code("""
import os, torch
from ultralytics import YOLO
LAST = f'{RUNS}/{RUN_NAME}/weights/last.pt'
# Ultralytics marks a finished run by stripping its checkpoint to epoch -1.
state = 'new'
if os.path.exists(LAST):
    ep = torch.load(LAST, map_location='cpu', weights_only=False).get('epoch', -1)
    state = 'finished' if ep == -1 else f'unfinished (last epoch {ep + 1})'
print('run', RUN_NAME, '->', state)
if state == 'new':
    YOLO(BASE_PT).train(
        data=DATA_YAML, imgsz=640, epochs=25, patience=8, batch=32,
        optimizer='auto', cos_lr=True, close_mosaic=10,
        mosaic=1.0, scale=0.5, translate=0.1, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=RUNS, name=RUN_NAME, exist_ok=True, seed=0, workers=2)
elif state.startswith('unfinished'):
    YOLO(LAST).train(resume=True)
""")

md("## Before / after on the same validation set")

code("""
from ultralytics import YOLO
best_pt = f'{RUNS}/{RUN_NAME}/weights/best.pt'
base = YOLO(BASE_PT)                 # again: after a disconnect it is gone
new = YOLO(best_pt)
if 'base_metrics' not in globals():
    base_metrics = base.val(data=DATA_YAML, imgsz=640, batch=32, plots=False)
new_metrics = new.val(data=DATA_YAML, imgsz=640, batch=32, plots=True)
for name, m in (('BEFORE (base model)', base_metrics), ('AFTER  (best.pt)   ', new_metrics)):
    print('%s  mAP50 %.3f  mAP50-95 %.3f  P %.3f  R %.3f' % (name, m.box.map50, m.box.map, m.box.mp, m.box.mr))
""")

md("""
## The test that matters — frames from the station
If you recorded frames on the station (`visual.frame_record`), put the
drone frames WITH labels (and backgrounds without) in
`MyDrive/drone_training/station_val/` (`*.jpg` + same-name `*.txt`).
These were never trained on. Scores are per image: the drone's best score,
and the strongest box on background frames (false positives).
""")

code("""
import glob, os
SV = f'{DRIVE}/station_val'
files = sorted(glob.glob(f'{SV}/*.jpg')) if os.path.isdir(SV) else []
print(len(files), 'station frames')
for label, m in (('BEFORE', base), ('AFTER ', new)):
    drone_best, bg_worst = [], []
    for f in files:
        r = m.predict(f, imgsz=640, conf=0.01, verbose=False)[0]
        s = [float(c) for c in r.boxes.conf]
        (drone_best if os.path.exists(f[:-4] + '.txt') else bg_worst).append(max(s) if s else 0.0)
    if drone_best:
        print(label, 'drone frames: median best score %.3f, found at >=0.4 in %d/%d' % (
            sorted(drone_best)[len(drone_best) // 2], sum(x >= 0.4 for x in drone_best), len(drone_best)))
    if bg_worst:
        print(label, 'background frames: false positives >=0.4 in %d/%d' % (
            sum(x >= 0.4 for x in bg_worst), len(bg_worst)))
""")

md("""
## Export ONNX for the Hailo Dataflow Compiler
The station's current HEF ends at the six raw head convolutions (box +
class for strides 8/16/32 — see `HEAD_CONFIG` in `TWO_CAMERAS_FIXED.py`).
Compile the new ONNX the same way; the cell prints the candidate end nodes.
""")

code("""
from ultralytics import YOLO
# Written next to best.pt, i.e. on Drive: runs/<RUN_NAME>/weights/best.onnx
onnx_path = YOLO(f'{RUNS}/{RUN_NAME}/weights/best.pt').export(
    format='onnx', imgsz=640, opset=11, simplify=True, dynamic=False)
print(onnx_path)
!pip -q install onnx
import onnx
g = onnx.load(onnx_path).graph
heads = [n.name for n in g.node if n.op_type == 'Conv' and 'one2one' in n.name]
print('last convs of the one2one heads (end nodes for the DFC parse):')
for n in heads[-6:]:
    print('  ', n)
""")

md("""
## HEF part — safe to stop and Run all again later
A HEF already on Drive is not compiled again; the steps below are skipped
once it is there. A compile interrupted half-way starts over.
""")

code("""
import glob, os
# The ONNX export above switched this notebook's process to CPU
# (Ultralytics select_device('cpu') sets CUDA_VISIBLE_DEVICES=''), and every
# command started from here would inherit it: the DFC would see no GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
HEF_OUT = f'{RUNS}/{RUN_NAME}/hef'
ONNX = f'{RUNS}/{RUN_NAME}/weights/best.onnx'
TODO = [r for r in ('zoo_fast',) if not glob.glob(f'{HEF_OUT}/*_{r}.hef')]
# A quantized HAR without its HEF: an earlier session optimized and then died
# in the compile. Only the compile is left - no calibration set, no GPU.
HARS = sorted(glob.glob(f'{HEF_OUT}/*_zoo_fast_quantized.har'))
print('HEF still to build:', TODO or 'none - it is on Drive',
      '| compile only, from', HARS[-1].split('/')[-1] if TODO and HARS else '')
sh = get_ipython().system            # what `!` does; f-strings below fill the {…}
""")

md("""
## Calibration set — images from the dataset, fed the way the station feeds the chip
1024 images from the **train** split (not val: the optimizer fine-tunes the
quantized weights on them), stratified by drone size, plus the station's own
background frames; letterboxed to 640 with grey 114 like `HailoInference`.
""")

code("%%writefile /content/build_calib.py\n" + BUILD_CALIB)

code("""
if TODO and not HARS:
    sh('python /content/build_calib.py /content/drone_dataset /content/calib --n 1024')
    import shutil; shutil.copy('/content/calib/calib_list.txt', f'{RUNS}/{RUN_NAME}/')
    from IPython.display import Image as _Img, display
    display(_Img('/content/calib/calib_sheet.jpg', width=900))
else:
    print('calibration not needed:', 'the HEF is on Drive' if not TODO
          else 'the quantized HAR is on Drive, only the compile is left')
""")

md("""
## Hailo Dataflow Compiler — its own venv, from the wheel on Drive
The DFC pins its own TensorFlow; it gets a separate Python so it cannot break
Ultralytics. The last line must list a GPU: without one the optimizer skips
the quantization-aware fine-tune the current HEF was built with.
""")

code('''
import glob, re, subprocess, zipfile
from packaging.specifiers import SpecifierSet
wheels = sorted(glob.glob(f'{DRIVE}/hailo_dataflow_compiler-*.whl'))
assert wheels, f'put the DFC wheel (hailo_dataflow_compiler-3.33.1-...whl) in {DRIVE}'
WHEEL = wheels[-1]
DFC_VER = re.search(r'hailo_dataflow_compiler-([0-9.]+)', WHEEL).group(1)
print(WHEEL)
if DFC_VER != '3.33.1':
    print(f'⚠️ DFC {DFC_VER}, not 3.33.1: the Pi runs HailoRT 4.23.0 (the DFC 3.33 '
          'release). A HEF from another DFC version may refuse to load there.')
with zipfile.ZipFile(WHEEL) as z:
    meta = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
m = re.search(r'^Requires-Python: (.+)$', meta, re.M)
REQ_PY = m.group(1).strip() if m else ''
PY = next(v for v in ['3.10', '3.11', '3.12', '3.9', '3.8']
          if not REQ_PY or SpecifierSet(REQ_PY).contains(v + '.0'))
print('Requires-Python:', REQ_PY or '(not stated)', '-> venv Python', PY)
VPY = '/content/hailo_venv/bin/python'
''')

code("""
if TODO:
    sh('apt-get -qq update > /dev/null && apt-get -qq install -y graphviz libgraphviz-dev > /dev/null')
    sh('pip -q install uv')
    sh(f'uv venv -q --allow-existing --python {PY} /content/hailo_venv')
    sh(f'uv pip install -q --python {VPY} "{WHEEL}" psutil')
    TF_VER = subprocess.check_output([VPY, '-c', 'import tensorflow as t; print(t.__version__)'],
                                     stderr=subprocess.DEVNULL).decode().strip()
    print('DFC pins TensorFlow', TF_VER, '- adding its CUDA libraries')
    sh(f'uv pip install -q --python {VPY} "tensorflow[and-cuda]=={TF_VER}"')
    check = ("import tensorflow as tf, hailo_sdk_client as h; print('DFC', "
             "getattr(h, '__version__', '?'), '| GPU:', tf.config.list_physical_devices('GPU'))")
    sh(f'{VPY} -c "{check}" 2>&1 | tail -1')
else:
    print('the HEF is on Drive - DFC install skipped')
""")

md("""
## Compile the HEF
Recipe **`zoo_fast`** — that of the station's current
`bestty_yolo26_1002_1255_zoo_fast.hef` (Hailo's own yolo26 layout, 16-bit
heads, the default level-2 optimization), so old and new HEF differ only in
the weights and the calibration images. Measured on the Pi on 02.10.2026, it
runs the detector ~12 % faster than the `logits` layout in crop mode.

ft_v2's `zoo_fast` build took ~37 min on a Colab T4 (optimization 17.5 min).
The HEF goes to Drive (`runs/<RUN_NAME>/hef/`) when it is done. The script
stops rather than build a HEF with a weaker optimization than the current one.

**If the session runs out of memory here** (round 4 did, right after
training): `Runtime → Restart session`, then Run all again — the finished
training is skipped and the compile gets the whole RAM.
""")

code("%%writefile /content/compile_hef.py\n" + COMPILE_HEF)

code("""
if 'zoo_fast' in TODO and HARS:
    sh(f'{VPY} /content/compile_hef.py --recipe zoo_fast --from-har "{HARS[-1]}" '
       f'--out "{HEF_OUT}"')
elif 'zoo_fast' in TODO:
    sh(f'{VPY} /content/compile_hef.py --recipe zoo_fast --onnx "{ONNX}" '
       f'--calib /content/calib --out "{HEF_OUT}"')
else:
    print('zoo_fast HEF already on Drive - skipped')
sh(f'ls -la "{HEF_OUT}"')
""")

md("""
## Next
Download from `MyDrive/drone_training/runs/<RUN_NAME>/`:
`hef/bestty_yolo26_<stamp>_zoo_fast.hef`, `hef/compile_summary_zoo_fast.json`
and `weights/best.pt` → `D:/d/DocumentsFromC/GitHub/drone_training/new_model_v5/`.
The HEF is then compared on the Pi with the current one on the same images
(`training/compare_hef.py`), and live on the station.
""")

def write(name):
    nb = {"cells": list(CELLS), "metadata": {
        **({} if name.startswith("har_") else {"accelerator": "GPU"}),
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4,
        "nbformat_minor": 0}
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote", out)
    CELLS.clear()


write("train_colab.ipynb")

# ── hef_colab.ipynb: the HEF part alone, for another Google account ──────
# (the training account's free GPU quota ran out; that account's Drive is
# not visible from here, so three files are carried over by hand)

# ft_v2 recompiled with Hailo's own yolo26 layout (compile_hef.py, recipes
# zoo_fast and zoo). The md5 is that of ft_v2's ONNX: checked on 02.10.2026
# against ft_v2's best.pt (identical scores on val images); ft_v3's ONNX,
# which the Drive folder held last, has another md5.
HEF_RECIPES = "zoo_fast zoo"
HEF_INPUTS = "drone_hef_v2zoo"
HEF_OUT = "hef_zoo"
FT_V2_ONNX_MD5 = "a3dba8517af77bc0531cf3c8ad94ea11"

md(f"""
# HEF only — ft_v2 `best.onnx` → HEF in Hailo's own yolo26 layout

For a Google account **other than the one the model was trained on**. Drag
the laptop's folder `drone_training/{HEF_INPUTS}/` whole into **My Drive**, so
that **`MyDrive/{HEF_INPUTS}/`** holds:
- `best.onnx` — **ft_v2**. The notebook checks its md5 `{FT_V2_ONNX_MD5}` and
  refuses any other model (ft_v3's has a different one)
- `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl`
- `calib_src.zip` — the 1024 calibration images (the same file as for ft_v2's current HEF)
- this notebook

Then `Runtime → Change runtime type → T4 GPU` and **Run all**. Two HEFs are
built one after the other into `MyDrive/{HEF_INPUTS}/{HEF_OUT}/`:
1. **`zoo_fast`** — Hailo's layout (no forced context switch, 16-bit heads),
   the optimization of the current HEF. Its last build took ~37 min on a T4.
2. **`zoo`** — Hailo's script as is, + AdaRound. Its duration on Colab is
   **not known**, and it may run out of memory. If it fails, `zoo_fast` is
   already saved.

After a disconnect, Run all again: a HEF already there is skipped.
""")

code("""
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
assert r.returncode == 0, ('NO GPU on this runtime: Runtime -> Change runtime type -> T4 GPU. '
                           'If Colab offers Pro instead, this account is out of free GPU time.')
""")

code("""
from google.colab import drive
drive.mount('/content/drive')
import glob, hashlib, os
INPUTS = '/content/drive/MyDrive/%s'
assert os.path.isdir(INPUTS), f'{INPUTS} not found - drag the whole folder %s into My Drive'""" % (HEF_INPUTS, HEF_INPUTS) + """
for pattern in ('best.onnx', 'hailo_dataflow_compiler-*.whl', 'calib_src.zip'):
    assert glob.glob(f'{INPUTS}/{pattern}'), f'put {pattern} in {INPUTS}'
onnxes = glob.glob(f'{INPUTS}/**/best.onnx', recursive=True)
assert len(onnxes) == 1, f'keep only ONE best.onnx (ft_v2) in {INPUTS}: {onnxes}'
md5 = hashlib.md5(open(onnxes[0], 'rb').read()).hexdigest()
assert md5 == '%s', f'best.onnx md5 {md5} - this is NOT ft_v2 (ft_v2 is %s)'
print('inputs OK, best.onnx is ft_v2')
""" % (FT_V2_ONNX_MD5, FT_V2_ONNX_MD5))

code("%%writefile /content/build_calib.py\n" + BUILD_CALIB)
code("%%writefile /content/compile_hef.py\n" + COMPILE_HEF)
code("%%writefile /content/hef_only.py\n"
     + (HERE / "hef_only.py").read_text(encoding="utf-8"))

code(f"""
!python /content/hef_only.py --inputs "{{INPUTS}}" --out "{{INPUTS}}/{HEF_OUT}" --work /content/hef_tmp --recipes {HEF_RECIPES} --onnx-md5 {FT_V2_ONNX_MD5}
""")

md(f"""
## Next
Download everything from `MyDrive/{HEF_INPUTS}/{HEF_OUT}/` (the `.hef` files and
`compile_summary_*.json`) → `D:/d/DocumentsFromC/GitHub/drone_training/new_model_zoo/`.
Before anything goes on the station, each HEF is measured on the Pi:
contexts and FPS (`hailortcli`), and detection quality against ft_v2's
current HEF (`compare_hef.py`).
""")

write("hef_colab.ipynb")

# ── har_colab.ipynb: the compile alone, from a quantized HAR ─────────────
# ft_v5 (06.10.2026): training and optimization finished, the session died
# in the compile, and the next session had no GPU. The HAR on Drive holds
# the optimized model; compiling it needs neither the GPU nor the dataset.
HAR_RUN = "ft_v5"

md(f"""
# HEF from the quantized HAR — {HAR_RUN}, compile only

For a run whose training and optimization finished but whose compile did
not (the session ran out of memory, or the next one had no GPU). Nothing is
trained or optimized again: `runs/{HAR_RUN}/hef/*_zoo_fast_quantized.har` on
Drive already holds the quantized model, and the compile runs on the CPU.

**Any runtime works, a GPU is not needed.** `Runtime → Run all`; allow Drive.
ft_v2's compile alone took ~20 min on Colab (02.10.2026: 37.8 min in all,
17.5 of them optimizing).
""")

code(f"""
from google.colab import drive
drive.mount('/content/drive')
import glob, os
DRIVE = '/content/drive/MyDrive/drone_training'
HEF_OUT = f'{{DRIVE}}/runs/{HAR_RUN}/hef'
HARS = sorted(glob.glob(f'{{HEF_OUT}}/*_zoo_fast_quantized.har'))
assert HARS, f'no *_zoo_fast_quantized.har in {{HEF_OUT}}'
HAR = HARS[-1]
WHEELS = sorted(glob.glob(f'{{DRIVE}}/hailo_dataflow_compiler-*.whl'))
assert WHEELS, f'put the DFC wheel in {{DRIVE}}'
print('HAR  ', HAR, os.path.getsize(HAR) // 2**20, 'MB')
print('wheel', WHEELS[-1])
print('already there:', [os.path.basename(p) for p in glob.glob(f'{{HEF_OUT}}/*.hef')] or 'no HEF')
""")

code("%%writefile /content/compile_hef.py\n" + COMPILE_HEF)

code("""
# The DFC in its own Python 3.10 venv (it pins TensorFlow 2.18); no CUDA
# libraries - the compile does not use the GPU.
!apt-get -qq update > /dev/null && apt-get -qq install -y graphviz libgraphviz-dev > /dev/null
!pip -q install uv
!uv venv -q --allow-existing --python 3.10 /content/hailo_venv
!uv pip install -q --python /content/hailo_venv/bin/python "{WHEELS[-1]}" psutil
!/content/hailo_venv/bin/python -c "import hailo_sdk_client as h; print('DFC', getattr(h, '__version__', '?'))" 2>&1 | tail -1
""")

code("""
!/content/hailo_venv/bin/python /content/compile_hef.py --recipe zoo_fast --from-har "{HAR}" --out "{HEF_OUT}"
!ls -la "{HEF_OUT}"
""")

md(f"""
## Next
Download from `MyDrive/drone_training/runs/{HAR_RUN}/hef/`: the `.hef`,
`compile_summary_zoo_fast.json` and the `zoo_fast_*.log` files, and from
`runs/{HAR_RUN}/weights/` the `best.pt`, into
`D:/d/DocumentsFromC/GitHub/drone_training/upload_v5/`. The output above
lists what is in the HAR — copy it too. The HEF is then measured on the Pi
against the station's current one.
""")

write("har_colab.ipynb")

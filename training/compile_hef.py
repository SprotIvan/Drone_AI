#!/usr/bin/env python3
"""
compile_hef.py — ONNX -> HEF for the station's Hailo-8L with DFC 3.33.1,
the compiler the station's current HEF (bestty_yolo260508.hef) was built
with, so that a difference between HEFs comes from what was changed on
purpose, not from the compiler.

    <dfc venv>/bin/python compile_hef.py --onnx best.onnx --calib <dir> \\
        --out <dir> --recipe logits|sigmoid|zoo_fast|zoo
    <dfc venv>/bin/python compile_hef.py --from-har <...>_zoo_fast_quantized.har \\
        --out <dir> --recipe zoo_fast

--from-har  the compile alone, from the quantized HAR a run saves right
          after the optimization — for a session that died in the compile
          (Colab ran out of memory there after training, ft_v4 and ft_v5).
          Needs no GPU and no calibration set.

--calib   a directory of .npy files, one image each: float32 (640, 640, 3),
          RGB, 0..255, letterboxed like the station (build_calib.py)
--out     receives bestty_yolo26_<stamp>_<recipe>.hef, the quantized .har,
          the DFC logs and compile_summary_<recipe>.json
--recipe  logits   AS THE CURRENT HEF: the class outputs leave the chip as
                   logits and the station applies the sigmoid. Measured on
                   the Pi (25.09.2026): that HEF's class logits are
                   quantized in steps of ~0.69-0.79, so above 0.4 it can
                   only report 0.5 / 0.665 / 0.69 / 0.80 / 0.83 / 0.887 /
                   0.915, never more.
          sigmoid  the sigmoid moves ONTO the chip (conv64/80/94), so what
                   is quantized is 0..1 in steps of ~0.004. NOT USED: its
                   quantization-aware fine-tune diverges on this model —
                   28.09.2026, total distill loss 2.09 -> 6.51 -> 5.57 ->
                   5.34 over the 4 epochs, conv94's pinned at 1.0000,
                   while `logits` fell 1.17 -> 1.13 -> 1.04 -> 1.00; an
                   earlier compile this way (5.08.2026) logged an SNR of
                   -1.1 dB on the same conv94.

          zoo_fast HAILO'S OWN LAYOUT, our optimization. Hailo's model-zoo
                   script for yolo26n/yolo26s on Hailo-8L (hailo_model_zoo
                   v2.18 = DFC 3.33.0 / HailoRT 4.23,
                   cfg/alls/generic/yolo26n.alls) sets NO
                   context_switch_param and keeps the six head convs, their
                   outputs and dw1/dw6/dw7/dw8 in 16 bit. Measured on the
                   station's Pi 5 (hailortcli, batch 1, 02.10.2026): their
                   yolo26n HEF has 4 contexts and runs 47.7 FPS (PCIe Gen 2)
                   / 78.2 FPS (Gen 3), ours (`logits`) 6 contexts, 37.8 /
                   65.2 FPS. Which of the differences makes the 4 contexts is
                   NOT known. The optimization stays the default one (level 2,
                   as `logits`), so this recipe only changes the layout.
          zoo      Hailo's script as it is: on top of zoo_fast, optimization
                   level 4 with AdaRound (calibset 1024, batch 8). The DFC's
                   own docstring calls AdaRound "time consuming and requires
                   strong system" — on Colab's 12.7 GB it may run out of
                   memory; its duration there is not known.

COMMON TO ALL (as the current HEF)
  * model name `bestty_yolo26`: the HEF's outputs are then named
    bestty_yolo26/conv61 ... conv94, which HEAD_CONFIG in
    TWO_CAMERAS_FIXED.py expects (another name and the station falls back
    to shape-based auto-detection);
  * the graph cut at the six one2one head convolutions (box + class for
    strides 8/16/32);
  * normalization([0,0,0],[255,255,255]): the station feeds UINT8 0..255;
  * context_switch_param(mode=enabled) — logits/sigmoid only;
  * the default optimization — level 2 with a GPU and >= 1024 images
    (statistics on 64 images, 4 epochs of fine-tune on 1024) — all but zoo.
  The station's decoder detects either output kind by itself.
  Unlike the current HEF, the layer noise analysis (a report, not a step
  that changes the model) is switched off: it ran out of memory on Colab.

REFUSES rather than quietly producing a worse HEF when
  * an end node or one of the six output layers is not where it should be;
  * TensorFlow in this venv sees no GPU, or there are fewer than 1024
    calibration images: the optimizer would silently drop to level 0 (no
    quantization-aware fine-tune) and say so only in its log.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("USER", "colab")        # the DFC reads it
# ⚠️ GPU selection, before the DFC is imported.
#  * Ultralytics' ONNX export calls select_device("cpu"), which sets
#    CUDA_VISIBLE_DEVICES="" in the notebook's process, and every command
#    started from the notebook afterwards inherits it: TensorFlow saw no GPU
#    on a T4 runtime (Colab, 25 and 28.09.2026). An empty or negative value
#    is therefore taken as "not chosen", not as "no GPU".
#  * The DFC's own selector (hailo_model_optimization/__init__.py) runs only
#    when the variable is unset, and rejects a GPU with > 5 % of its memory
#    in use — the notebook's PyTorch holds ~1.4 GB of the T4 — so it is
#    bypassed by naming the device.
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in ("", "-1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "bestty_yolo26"
END_NODES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
]
# (box layer, class layer, channels) per stride — HEAD_CONFIG's names.
HEADS = [("conv61", "conv64", 8), ("conv77", "conv80", 16),
         ("conv91", "conv94", 32)]
NORMALIZATION = (
    "norm_layer = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n")
# The layer noise analysis only REPORTS per-output SNR ("does not modify the
# structure of the model's graph", CheckerConfig in the 3.33.1 wheel). On
# Colab (12.7 GB) it was killed for memory (exit -9) right after the
# fine-tune had finished, twice (28.09.2026) — so it is off.
CHECKER_OFF = "model_optimization_config(checker_cfg, policy=disabled)\n"
MODEL_SCRIPT = (NORMALIZATION + "context_switch_param(mode=enabled)\n"
                + CHECKER_OFF)
# Hailo's 16-bit layers for yolo26n/s (cfg/alls/generic/yolo26n.alls, model
# zoo v2.18), names as in their script. check_layers() refuses to start if
# the parsed model does not have them.
ZOO_16BIT = (
    "quantization_param([dw1, dw6, dw7, dw8], precision_mode=a16_w16)\n"
    "quantization_param([conv61, conv77, conv91, conv64, conv80, conv94], "
    "precision_mode=a16_w16)\n"
    "quantization_param([output_layer1, output_layer2, output_layer3, "
    "output_layer4, output_layer5, output_layer6], precision_mode=a16_w16)\n"
)
RECIPES = {
    "logits": MODEL_SCRIPT,
    "sigmoid": MODEL_SCRIPT + "".join(
        f"change_output_activation({cls}, sigmoid)\n" for _, cls, _ in HEADS),
    "zoo_fast": NORMALIZATION + ZOO_16BIT + CHECKER_OFF,
    "zoo": (NORMALIZATION
            + "model_optimization_config(calibration, batch_size=8, "
              "calibset_size=1024)\n"
            + "model_optimization_flavor(optimization_level=4, "
              "compression_level=0)\n"
            + "post_quantization_optimization(adaround, policy=enabled, "
              "batch_size=8)\n"
            + ZOO_16BIT + CHECKER_OFF),
}
MIN_CALIB = 1024
IMG_BYTES = 640 * 640 * 3 * 4


def check_onnx(path):
    import onnx
    names = {n.name for n in onnx.load(path).graph.node}
    missing = [n for n in END_NODES if n not in names]
    if missing:
        heads = sorted(n for n in names if "one2one" in n and n.endswith("Conv"))
        sys.exit("end nodes not in the ONNX: %s\none2one convs present:\n  %s"
                 % (missing, "\n  ".join(heads)))


def check_heads(runner):
    """The six output layers must be HEAD_CONFIG's convs, box = 4 channels,
    class = 1 channel, at 80/40/20 cells. (Checked against the HN of the
    current model's HAR: output_layer1..6 <- conv61, 64, 77, 80, 91, 94.)"""
    layers = runner.get_hn_dict()["layers"]
    feeding = set()
    for layer in layers.values():
        if layer.get("type") == "output_layer":
            feeding.update(layer.get("input", []))
    problems = []
    for box, cls, stride in HEADS:
        cells = 640 // stride
        for short, ch in ((box, 4), (cls, 1)):
            full = f"{MODEL_NAME}/{short}"
            if full not in layers:
                problems.append(f"{full}: no such layer")
                continue
            if feeding and full not in feeding:
                problems.append(f"{full}: not an output")
            shape = (layers[full].get("output_shapes") or [[None]])[0]
            if list(shape[-3:]) != [cells, cells, ch]:
                problems.append(f"{full}: shape {shape}, expected "
                                f"[.., {cells}, {cells}, {ch}]")
    if problems:
        sys.exit("parsed model does not match HEAD_CONFIG:\n  "
                 + "\n  ".join(problems))
    print("heads OK: " + ", ".join(f"{b}/{c} (stride {s})"
                                   for b, c, s in HEADS))


def check_layers(runner, recipe):
    """Every layer a quantization_param names must exist in the parsed
    model — checked BEFORE the optimizer runs, so a name that differs from
    Hailo's yolo26n costs seconds, not an hour of Colab."""
    script = "".join(l for l in RECIPES[recipe].splitlines(True)
                     if l.startswith("quantization_param"))
    wanted = sorted(set(re.findall(r"\b(?:dw|conv|output_layer)\d+\b",
                                   script)))
    if not wanted:
        return
    layers = runner.get_hn_dict()["layers"]
    missing = [n for n in wanted if f"{MODEL_NAME}/{n}" not in layers]
    if missing:
        dws = sorted((n.split("/")[-1] for n in layers
                      if n.split("/")[-1].startswith("dw")),
                     key=lambda s: int(s[2:]) if s[2:].isdigit() else 0)
        sys.exit(f"recipe {recipe}: layers not in the parsed model: "
                 f"{missing}\n depthwise layers present: {dws}")
    print(f"16-bit layers OK: {', '.join(wanted)}")


def build_runner(onnx_path, recipe):
    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(onnx_path, MODEL_NAME,
                                start_node_names=["images"],
                                end_node_names=END_NODES)
    check_heads(runner)
    check_layers(runner, recipe)
    runner.load_model_script(RECIPES[recipe])
    return runner


def calib_strategies(files):
    """Ways to hand the calibration set to runner.optimize, most proven
    first. Both are read by the DFC one image at a time through a
    generator (hailo_model_optimization/acceleras/utils/dataset_util.py in
    the 3.33.1 wheel), so the array costs its own size in RAM, not twice."""
    import numpy as np
    import psutil
    avail = psutil.virtual_memory().available
    # The array's 5 GB plus room for the optimizer and the compiler: with
    # only 4 GB of room the process was killed for memory on Colab.
    need = len(files) * IMG_BYTES + (10 << 30)

    def as_array():                 # what built the current HEF
        a = np.empty((len(files), 640, 640, 3), np.float32)
        for i, f in enumerate(files):
            a[i] = np.load(f)
        return a

    def as_dir():                   # DFC "npy_dir": one image per file,
        return str(files[0].parent)     # memory-mapped as it is read

    if avail >= need:
        return [("in-memory array", as_array), ("npy directory", as_dir)]
    print(f"RAM available {avail / 2**30:.1f} GB < {need / 2**30:.1f} GB "
          "for the in-memory array: the directory first")
    return [("npy directory", as_dir), ("in-memory array", as_array)]


def log_facts(text):
    """What the DFC's client log says about the optimization it ran."""
    lines = text.splitlines()
    return {
        "optimization_level": [l for l in lines
                               if "optimization level" in l.lower()][:5],
        "optimization_level_reduced": [l for l in lines
                                       if "Reducing optimization level" in l],
        "qft_lines": [l for l in lines if "for finetune" in l],
        "snr_db": dict(re.findall(r"(output_layer\d+) SNR:\s*(-?[\d.]+) dB",
                                  text)),
    }


def save_logs(out, recipe):
    for f in Path(".").glob("*.log"):
        shutil.copy(f, out / f"{recipe}_{f.name}")


def show_har(har):
    """A HAR is a tar archive: list it, and quote whatever its text members
    say about the optimization — the HAR is all that is left when the
    session that optimized it died before the compile finished."""
    import tarfile
    keys = re.compile(r"optimization_level|flavor|finetune|qft|Reducing|"
                      r"calibset|sdk_version|state", re.I)
    with tarfile.open(har) as t:
        for m in t.getmembers():
            print(f"  {m.name}  {m.size} B")
            if m.isfile() and m.size < (4 << 20) and not m.name.endswith(".npz"):
                text = t.extractfile(m).read().decode("utf-8", "replace")
                hits = [l.strip()[:200] for l in text.splitlines() if keys.search(l)]
                for l in hits[:12]:
                    print("     ", l)


def compile_from_har(args, out, dfc):
    """The compile alone, from a quantized HAR an earlier run saved: no GPU,
    no calibration set, no optimization — the weights are already the ones
    the optimizer produced. Only for recipes whose model script has no
    compile-time command (context_switch_param is one), so that compiling
    the HAR is the same as the compile that run would have done."""
    har = Path(args.from_har).resolve()
    if "context_switch_param" in RECIPES[args.recipe]:
        sys.exit(f"--from-har: recipe {args.recipe} has a compile-time command "
                 "that a HAR may not carry; optimize again instead")
    if not har.name.endswith(f"_{args.recipe}_quantized.har"):
        sys.exit(f"{har.name} is not a {args.recipe} quantized HAR")
    print("HAR", har)
    show_har(har)
    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(har=str(har))
    check_heads(runner)
    t0 = time.time()
    hef = out / (har.name[:-len("_quantized.har")] + ".hef")
    hef.write_bytes(runner.compile())
    # The optimization's own log, if the run that wrote the HAR saved it.
    opt_log = out / f"{args.recipe}_hailo_sdk.client.log"
    facts = (log_facts(opt_log.read_text(errors="replace")) if opt_log.exists()
             else {"optimization_level": "NOT KNOWN: the log of the run that "
                   "optimized this HAR was not saved"})
    summary = {"hef": hef.name, "recipe": args.recipe, "dfc": dfc,
               "compiled_from_har": har.name,
               "model_script": RECIPES[args.recipe], **facts,
               "compile_min": round((time.time() - t0) / 60, 1)}
    (out / f"compile_summary_{args.recipe}.json").write_text(
        json.dumps(summary, indent=1))
    for f in Path(".").glob("*.log"):
        shutil.copy(f, out / f"{args.recipe}_compile_{f.name}")
    print(json.dumps(summary, indent=1))
    print(f"\nSUCCESS: {hef}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--calib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--work", help="scratch folder for the DFC (default: "
                    "/content/hef_work_<recipe> on Colab, else under --out)")
    ap.add_argument("--from-har", help="compile this <...>_<recipe>_quantized"
                    ".har (saved by a run whose compile did not finish) "
                    "instead of optimizing again; needs no GPU")
    args = ap.parse_args()
    if not args.from_har and not (args.onnx and args.calib):
        ap.error("--onnx and --calib are required unless --from-har")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else (
        (Path("/content") if Path("/content").exists() else out)
        / f"hef_work_{args.recipe}")
    shutil.rmtree(work, ignore_errors=True)   # its logs must be this run's
    work.mkdir(parents=True)
    os.chdir(work)                      # the DFC writes its logs to the cwd

    import hailo_sdk_client
    import tensorflow as tf
    dfc = getattr(hailo_sdk_client, "__version__", "?")
    print("DFC", dfc, "| TensorFlow", tf.__version__)
    if args.from_har:
        compile_from_har(args, out, dfc)
        return
    onnx_path = str(Path(args.onnx).resolve())
    if not tf.config.list_physical_devices("GPU"):
        sys.exit("TensorFlow in this venv sees no GPU: the optimizer would "
                 "drop to level 0 (no quantization-aware fine-tune). Fix the "
                 "CUDA install first.")
    files = sorted(Path(args.calib).glob("*.npy"))
    if len(files) < MIN_CALIB:
        sys.exit(f"{len(files)} calibration images < {MIN_CALIB}: the "
                 "optimizer would drop to a lower level")
    check_onnx(onnx_path)

    t0 = time.time()
    runner, used, errors = None, None, []
    for label, make in calib_strategies(files):
        runner = build_runner(onnx_path, args.recipe)
        try:
            print(f"optimize: calibration as {label}", flush=True)
            runner.optimize(make())
            used = label
            break
        except Exception as exc:          # try the next way of feeding it
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print("  failed —", errors[-1][:500], flush=True)
    if used is None:
        sys.exit("optimize failed every way:\n  " + "\n  ".join(errors))
    t_opt = time.time() - t0

    stamp = time.strftime("%m%d_%H%M")
    base = f"{MODEL_NAME}_{stamp}_{args.recipe}"
    runner.save_har(str(out / f"{base}_quantized.har"))
    # The optimization log goes next to the HAR NOW: a session that dies in
    # the compile (ft_v4 and ft_v5 did, out of memory) leaves the HAR, and
    # without this log nothing says at which level it was optimized.
    save_logs(out, args.recipe)
    hef = out / f"{base}.hef"
    hef.write_bytes(runner.compile())
    t_all = time.time() - t0

    log = Path("hailo_sdk.client.log")
    text = log.read_text(errors="replace") if log.exists() else ""
    summary = {
        "hef": hef.name, "recipe": args.recipe,
        "model_script": RECIPES[args.recipe], "dfc": dfc, "onnx": onnx_path,
        "calibration_images": len(files), "calibration_as": used,
        "failed_ways": errors, **log_facts(text),
        "optimize_min": round(t_opt / 60, 1),
        "total_min": round(t_all / 60, 1),
    }
    (out / f"compile_summary_{args.recipe}.json").write_text(
        json.dumps(summary, indent=1))
    save_logs(out, args.recipe)
    print(json.dumps(summary, indent=1))
    if summary["optimization_level_reduced"]:
        print("\n⚠️  THE OPTIMIZER REDUCED ITS LEVEL — this HEF is not built "
              "the way the current one was. Do not deploy it.")
    print(f"\nSUCCESS: {hef}")


if __name__ == "__main__":
    main()

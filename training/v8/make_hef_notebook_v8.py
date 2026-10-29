#!/usr/bin/env python3
"""
make_hef_notebook_v8.py — hef_colab_v8.ipynb: the existing hef_colab.ipynb
(build_calib.py, compile_hef.py, hef_only.py embedded unchanged) pointed at
MyDrive/drone_hef_v8 and locked to ft_v8's best.onnx md5, recipe zoo_fast
(the recipe of ft_v5's and ft_v7's HEFs: bestty_yolo26_*_zoo_fast.hef).

    python make_hef_notebook_v8.py HEF_INPUTS_DIR
HEF_INPUTS_DIR must hold best.onnx and best.onnx.md5 (export_v8.py output).
"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "hef_colab.ipynb"
OLD_MD5 = "a3dba8517af77bc0531cf3c8ad94ea11"


def main():
    inp = Path(sys.argv[1])
    md5 = (inp / "best.onnx.md5").read_text().strip()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.startswith("%%writefile"):
            continue                       # the three scripts, unchanged
        s = s.replace("drone_hef_v2zoo", "drone_hef_v8").replace(OLD_MD5, md5)
        s = s.replace("ft_v2 `best.onnx`", "ft_v8 `best.onnx`").replace("(ft_v2)", "(ft_v8)")
        s = s.replace("this is NOT ft_v2 (ft_v2 is", "this is NOT ft_v8 (ft_v8 is")
        s = s.replace("best.onnx is ft_v2", "best.onnx is ft_v8")
        s = s.replace("--recipes zoo_fast zoo", "--recipes zoo_fast").replace('"{INPUTS}/hef_zoo"', '"{INPUTS}/hef_v8"')
        if c["cell_type"] == "markdown" and s.startswith("# HEF only"):
            s = ("# HEF only — ft_v8 `best.onnx` → `bestty_yolo26_<stamp>_zoo_fast.hef`\n\n"
                 "Drag the laptop's folder `drone_training/v8/hef_inputs_v8/` whole into **My Drive** "
                 "and rename it **`drone_hef_v8`**, so that `MyDrive/drone_hef_v8/` holds:\n"
                 f"- `best.onnx` — **ft_v8**, md5 `{md5}` (the notebook refuses any other)\n"
                 "- `hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl` (copy it from "
                 "`drone_training/drone_hef/`)\n"
                 "- `calib_src.zip` — 1024 calibration images picked from the ft_v8 TRAIN split "
                 "(`build_calib.py --pick-only`)\n"
                 "- this notebook\n\n"
                 "Then `Runtime → Change runtime type → T4 GPU` and **Run all**. One HEF, recipe "
                 "`zoo_fast` (as ft_v5's and ft_v7's), lands in `MyDrive/drone_hef_v8/hef_v8/` "
                 "(~20-30 min: optimization with the quantization-aware fine-tune, then the compile).\n")
        if c["cell_type"] == "markdown" and s.startswith("## Next"):
            s = ("## Next\nDownload `MyDrive/drone_hef_v8/hef_v8/` (the `.hef`, the quantized `.har`, "
                 "`compile_summary_zoo_fast.json`, the logs) → `drone_training/v8/hef_v8/`. On the Pi "
                 "the HEF is then measured against ft_v5's and ft_v7's on the same frames before it "
                 "goes anywhere near the station.\n")
        c["source"] = [l for l in s.splitlines(keepends=True)]
    out = inp / "hef_colab_v8.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    left = [l for c in nb["cells"] if not "".join(c["source"]).startswith("%%writefile")
            for l in c["source"] if "v2zoo" in l or OLD_MD5 in l]
    assert not left, left
    print("->", out, "md5", md5)


if __name__ == "__main__":
    main()

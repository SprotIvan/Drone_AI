#!/usr/bin/env python3
"""
make_upload_folder.py — the lean folder the owner drags into the root of
My Drive to compile ft_v8 in Colab: only what the notebook uses, copied
(byte-checked) from the verified package v8/ft_v8.

    python make_upload_folder.py SRC_PACKAGE DST_FOLDER

Copied: best.pt, calib_src.zip, samples_expected.json, the DFC 3.33.1 wheel,
scripts/ (4), requirements_export.txt, config/ (model/args/data.yaml - for
reference), README.md (adapted to this folder) and the notebook generated for
the Drive folder name = DST_FOLDER's name.
Left out: laptop_check/ (laptop results, test samples - Colab extracts its
samples from calib_src.zip), best.pt.md5 (the md5 is inside the notebook and
samples_expected.json), the reports and the change log (stay in v8/ft_v8).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ft_v8_package as M  # noqa: E402

FILES = ["best.pt", "calib_src.zip", "samples_expected.json",
         "hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl", "requirements_export.txt",
         "scripts/build_calib.py", "scripts/compile_hef.py", "scripts/hef_only.py",
         "scripts/colab_export_verify.py"]
CONFIG = ["model.yaml", "args.yaml", "data.yaml"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def readme(text, folder, model="ft_v8"):
    """The package README adapted to the upload folder - every replacement must hit."""
    text = text.replace(model, "ft_v8")      # the replacements below are written for ft_v8
    reps = [
        ("Чому обрано — `REPORT.md`.", "Чому обрано — `drone_training\\v8\\ft_v8\\REPORT.md` на ноутбуці."),
        ("(`laptop_check/onnx_check.json`)", "(`drone_training\\v8\\ft_v8\\laptop_check\\onnx_check.json` на ноутбуці)"),
        (" (`best.pt.md5`) |", " (md5 записаний і в ноутбуці — комірка 2 звіряє) |"),
        ("| `model.yaml`, `args.yaml`, `data.yaml` |", "| `config/model.yaml`, `config/args.yaml`, `config/data.yaml` |"),
        ("перетягніть усю папку\n   `ft_v8` (з Провідника) у корінь «Мого диска». Має вийти `Мій диск/ft_v8/best.pt` і т. д.",
         f"перетягніть усю папку\n   `{folder}` (з робочого столу) у корінь «Мого диска». Має вийти `Мій диск/{folder}/best.pt` і т. д."),
        ("`ft_v8/ft_v8_colab.ipynb`", f"`{folder}/ft_v8_colab.ipynb`"),
        ("   - 2 — підключення Drive і перевірка, що всі файли на місці;",
         f"   - 2 — підключення Drive і перевірка, що всі файли на місці (папка `Мій диск/{folder}`; якщо Drive її\n"
         "     перейменував, напр. на «(1)», — знаходить її за md5 `best.pt`);"),
        ("`Мій диск/ft_v8/out/`", f"`Мій диск/{folder}/out/`"),
        ("`ft_v8 → out → hef`", f"`{folder} → out → hef`"),
    ]
    for old, new in reps:
        assert old in text, f"README: expected text not found: {old[:70]!r}"
        text = text.replace(old, new)
    drop = ["| `laptop_check/` |", "| `REPORT.md` |", "| `REPORT_V8.md` |", "| `CHANGELOG.md` |"]
    lines = [l for l in text.splitlines() if not any(l.startswith(d) for d in drop)]
    assert len(lines) == len(text.splitlines()) - len(drop), "README: a table row to drop was not found"
    text = "\n".join(lines) + "\n"
    text += ("\n## Чого тут немає (навмисно)\n"
             "Лише те, що використовує ноутбук. Звіти (`REPORT.md`, `REPORT_V8.md`), журнал `CHANGELOG.md` і\n"
             "результати перевірки на ноутбуці (`laptop_check/`) лежать у `C:\\Github\\drone_training\\v8\\ft_v8\\`.\n")
    if model != "ft_v8":
        text = text.replace("drone_training\\v8\\ft_v8", "drone_training\\v9\\" + model).replace("ft_v8", model)
    return text


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        sys.exit(f"{dst} exists - not overwritten")
    (dst / "scripts").mkdir(parents=True)
    (dst / "config").mkdir()
    for f in FILES:
        shutil.copy2(src / f, dst / f)
        assert md5(src / f) == md5(dst / f), f"copy differs: {f}"
    for f in CONFIG:
        shutil.copy2(src / f, dst / "config" / f)
        assert md5(src / f) == md5(dst / "config" / f), f"copy differs: {f}"
    pt_md5 = md5(dst / "best.pt")
    model = src.name                         # ft_v8 / ft_v9 ... = the package folder name
    nb = M.notebook(model, pt_md5, folder=dst.name)
    (dst / f"{model}_colab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (dst / "README.md").write_text(readme((src / "README.md").read_text(encoding="utf-8"), dst.name, model),
                                   encoding="utf-8")
    files = sorted(p for p in dst.rglob("*") if p.is_file())
    for p in files:
        print(f"  {p.relative_to(dst)}  {p.stat().st_size} B")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.0f} MB -> {dst}; best.pt md5 {pt_md5}")


if __name__ == "__main__":
    main()

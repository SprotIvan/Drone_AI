#!/usr/bin/env python3
"""
build_v8.py — assemble drone_dataset_v8 (one class: drone) from
  * drone_dataset_v5.zip train (read only, never modified), minus confirmed
    label errors, one fixed double label;
  * v5 val images that are NOT the same photo / scene as a train image,
    split by visual cluster into  val  (model selection, threshold choice)
    and  test_inet  (reported, never used for selection);
  * parts/station_train   (station_additions.py export, by-eye decisions)
  * parts/closeups/{train,val}  minus the close-up QA rejects
  * parts/coco_neg/{train,val}  minus the COCO screening rejects

    python build_v8.py DECISIONS.json OUT_DIR

DECISIONS.json: {"closeup_reject": {file: reason}, "coco_reject": {file: reason}}
Writes OUT_DIR/images|labels/{train,val}, OUT_DIR/data.yaml, OUT_DIR/manifest.json,
OUT_DIR/stats.json + STATS.md, and v8/testsets/test_inet/ (images + labels).
"""
import io
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

V5ZIP = Path(r"C:\Users\Nazar\Downloads\drone_dataset_v5.zip")
V8 = Path(r"C:\Github\drone_training\v8")
PARTS = V8 / "parts"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXCLUDE = {"images/train/station_018446.jpg":
           "background with an unlabelled dark object in the sky (drone or bird) - doubtful"}
FIX_FIRST_BOX = {"images/train/drone-detection-project__drone-vs-bird-d_000820.jpg":
                 "two labels on one object (IoU > 0.9) - first label kept"}


def origin(name):
    s = Path(name).stem
    if s.startswith("st8_"):
        return "station_add"
    if s.startswith("close__"):
        return "closeup:" + s.split("__")[1]
    if s.startswith("coco_"):
        return "coco:" + s.split("_")[1]
    if "__" in s:
        return s.split("__")[0]
    return s.rsplit("_", 1)[0]


def bits(a):
    d = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in d), 2)


def feat(data):
    with Image.open(io.BytesIO(data)) as im:
        g = im.convert("L")
        return (bits(np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)),
                np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.int16))


def split_val(names, z):
    """Visual clusters (dHash <= 12 bits AND thumbnails within 12 grey levels,
    any source) + runs of neighbouring indices of one source -> components;
    components go to val / test_inet alternately (seeded), so no near-identical
    picture is on both sides."""
    H, T = [], []
    for n in names:
        h, t = feat(z.read(n))
        H.append(h)
        T.append(t)
    H = np.array(H, dtype=np.uint64)
    T = np.stack(T)
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        d = np.bitwise_count(H ^ H[i])
        js = np.flatnonzero(d <= 12)
        js = js[js > i]
        if len(js):
            mad = np.abs(T[js] - T[i]).mean(axis=(1, 2))
            for j in js[mad <= 12]:
                parent[find(int(j))] = find(i)
    idx = defaultdict(list)
    for i, n in enumerate(names):
        stem = Path(n).stem
        src, _, num = stem.rpartition("_")
        if num.isdigit():
            idx[src].append((int(num), i))
    for src, lst in idx.items():
        lst.sort()
        for (a, i), (b, j) in zip(lst, lst[1:]):
            if b - a <= 3:
                parent[find(j)] = find(i)
    comps = defaultdict(list)
    for i in range(len(names)):
        comps[find(i)].append(names[i])
    order = sorted(comps.values(), key=lambda c: (-len(c), c[0]))
    rng = random.Random(8)
    val, test, nv, nt = [], [], 0, 0
    for c in order:
        if nv < nt or (nv == nt and rng.random() < 0.5):
            val += c
            nv += len(c)
        else:
            test += c
            nt += len(c)
    return val, test, len(order)


def box_stats(lines, w, h, c):
    for line in lines:
        p = line.split()
        if len(p) != 5:
            continue
        bw, bh = float(p[3]), float(p[4])
        k = 640 / max(w, h)
        side = max(bw * w, bh * h) * k
        c["boxes"] += 1
        c["box<16px"] += side < 16
        c["box16-32px"] += 16 <= side < 32
        c["box32-96px"] += 32 <= side < 96
        c["box96-320px"] += 96 <= side < 320
        c["box>=320px"] += side >= 320
        c["box_area>15%"] += bw * bh > 0.15


def main():
    dec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    if out.exists():
        sys.exit(f"{out} exists - remove it first (never built over)")
    lc = json.loads((V8 / "audit" / "label_check" / "label_check.json").read_text())
    missing = set(lc["all"]["missing_label"])
    audit = json.loads((V8 / "audit" / "v5_audit.json").read_text())
    leaked = {p[0] for k in ("dup_cross_split", "scene_cross_split") for p in audit["near_pairs"][k]}
    z = zipfile.ZipFile(V5ZIP)
    names = sorted(n for n in z.namelist() if n.startswith("images/") and Path(n).suffix.lower() in IMG)
    log = Counter()
    manifest = []
    plan = []                                   # (split, dst name, read_img, label text, origin note)

    def zlab(n):
        try:
            return z.read(n.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt").decode()
        except KeyError:
            return ""

    v5val = []
    for n in names:
        split = n.split("/")[1]
        if n in missing:
            log[f"v5 {split}: dropped - drone without a label (label_check missing_label, checked by eye)"] += 1
            continue
        if n in EXCLUDE:
            log[f"v5 {split}: dropped - {EXCLUDE[n]}"] += 1
            continue
        if split == "val":
            if n in leaked:
                log["v5 val: dropped - same photo / scene as a v5 train image"] += 1
                continue
            if Path(n).name.startswith("user_photos_"):
                log["v5 val: moved out - the owner's 7 photos are the separate owner7 test"] += 1
                continue
            v5val.append(n)
            continue
        lab = zlab(n)
        if n in FIX_FIRST_BOX:
            lab = lab.splitlines()[0] + "\n"
            log[f"v5 train: fixed - {FIX_FIRST_BOX[n]}"] += 1
        plan.append(("train", Path(n).name, ("zip", n), lab, "v5 train"))
    val, test, ncomp = split_val(v5val, z)
    log[f"v5 val clean: {len(v5val)} -> val {len(val)} / test_inet {len(test)} ({ncomp} visual clusters)"] += 1
    for n in val:
        plan.append(("val", Path(n).name, ("zip", n), zlab(n), "v5 val (clean)"))
    tdir = V8 / "testsets" / "test_inet"
    if tdir.exists():
        shutil.rmtree(tdir)
    (tdir / "images").mkdir(parents=True)
    (tdir / "labels").mkdir(parents=True)
    for n in test:
        (tdir / "images" / Path(n).name).write_bytes(z.read(n))
        (tdir / "labels" / (Path(n).stem + ".txt")).write_text(zlab(n))
    for part_dir, split, rej_key, note in ((PARTS / "station_train", "train", None, "station additions"),
                                           (PARTS / "closeups" / "train", "train", "closeup_reject", "close-ups"),
                                           (PARTS / "closeups" / "val", "val", "closeup_reject", "close-ups"),
                                           (PARTS / "coco_neg" / "train", "train", "coco_reject", "COCO negatives"),
                                           (PARTS / "coco_neg" / "val", "val", "coco_reject", "COCO negatives")):
        rej = dec.get(rej_key, {}) if rej_key else {}
        for p in sorted((part_dir / "images").iterdir()):
            if p.suffix.lower() not in IMG:
                continue
            if p.name in rej:
                log[f"{note} {split}: dropped - {rej[p.name]}"] += 1
                continue
            lab = (part_dir / "labels" / (p.stem + ".txt")).read_text()
            plan.append((split, p.name, ("file", p), lab, note))
    seen = set()
    for split, name, _, _, _ in plan:
        assert (split, name) not in seen, name
        seen.add((split, name))
    stats = defaultdict(Counter)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for split, name, (kind, src), lab, note in plan:
        data = z.read(src) if kind == "zip" else Path(src).read_bytes()
        (out / "images" / split / name).write_bytes(data)
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text(lab)
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
        lines = [l for l in lab.splitlines() if l.strip()]
        for key in (split, f"{split}|{origin(name)}"):
            c = stats[key]
            c["images"] += 1
            c["drone_images" if lines else "background_images"] += 1
            box_stats(lines, w, h, c)
        manifest.append({"split": split, "file": name, "from": note,
                         "src": src if kind == "zip" else str(src), "boxes": len(lines)})
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "nc: 1\nnames: ['drone']\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=0))
    st = {"log": dict(log), "splits": {k: dict(v) for k, v in sorted(stats.items())}}
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    md = ["# drone_dataset_v8 — stats", "", "## Build log", ""]
    md += [f"- {k}: {v}" for k, v in log.items()]
    for split in ("train", "val"):
        md += ["", f"## {split}", "", "| origin | images | drone | background | boxes | <16px | 16-32 | 32-96 | 96-320 | >=320 | area>15% |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(k for k in stats if k.startswith(split + "|")) + [split]:
            c = stats[k]
            md.append(f"| {k.split('|')[-1] if '|' in k else '**all**'} | {c['images']} | {c['drone_images']} | "
                      f"{c['background_images']} | {c['boxes']} | {c['box<16px']} | {c['box16-32px']} | "
                      f"{c['box32-96px']} | {c['box96-320px']} | {c['box>=320px']} | {c['box_area>15%']} |")
    (out / "STATS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

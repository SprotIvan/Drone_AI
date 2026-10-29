#!/usr/bin/env python3
"""
select_birds.py — v8.1 negatives "birds close up": photos of the drone-vs-bird
source that hold BIRDS ONLY (no drone label), bird box >= --min-area of the
frame. v5's builder dropped every close-up, birds included (largest box > 15 %),
so v8 got 2 247 close-up drones and no close-up birds at all.

    python select_birds.py WORK_DIR OUT_DIR --n 600 [--min-area 0.05]

Dropped: the photos of the bg_birds_dvb test set (by file name) and any photo
that is the same photo / scene as an image of ANY v8 test set or of the v8 val
(dHash <= 6 and 64x64 thumbnails within 6 grey levels, round-7 rule);
near-duplicates among the candidates (one kept). Kept photos are copied with
an EMPTY label (= no drone) to OUT_DIR/images, labels; manifest.json.
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_dataset as B  # noqa: E402
from fetch_datasets import SOURCES  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
V8 = Path(r"C:\Github\drone_training\v8")
SLUG = "drone-detection-project/drone-vs-bird-detection"


def feats(path):
    with Image.open(path) as im:
        im.load()
        h, hm = B.dhash_pair(im)
        th = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.int16)
    return h, hm, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--min-area", type=float, default=0.05)
    a = ap.parse_args()
    work, out = Path(a.work_dir), Path(a.out_dir)
    if out.exists():
        sys.exit(f"{out} exists")
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    stats = Counter()
    # test photos by name + every test / val image for the near-duplicate check
    test_names = set()
    for line in (Path(r"C:\Github\drone_training\eval_v5\bg_birds_dvb\source.txt")
                 .read_text(encoding="utf-8").splitlines()):
        if " " in line:
            test_names.add(Path(line.split(" ", 1)[1].replace("\\", "/")).name)
    ex = [p for p in (V8 / "testsets").rglob("*") if p.suffix.lower() in IMG and "_excluded_leak" not in p.parts
          and "station" not in p.parts]
    ex += [p for p in (V8 / "dataset_v8" / "images" / "val").iterdir() if p.suffix.lower() in IMG]
    exf = [feats(p) for p in ex]
    EH = np.array([f[0] for f in exf], dtype=np.uint64)
    ETH = np.stack([f[2] for f in exf])
    print(len(ex), "test/val images to keep out", flush=True)

    def leaks(f):
        h, hm, th = f
        for hh, thx in ((h, th), (hm, th[:, ::-1])):
            d = np.bitwise_count(EH ^ np.uint64(hh))
            cand = np.flatnonzero(d <= 6)
            if len(cand) and (np.abs(ETH[cand] - thx).mean(axis=(1, 2)) <= 6).any():
                return True
        return False

    slug, version, licence, cmap, note = next(s for s in SOURCES if s[0] == SLUG)
    name = f"{slug.replace('/', '__')}__v{version}"
    items = B.collect_source(work / "raw" / name, name, slug, cmap, Counter(), B.FileCache(out / "cache.json"))
    cand = [it for it in items if not it["boxes"] and it["other_area"] >= a.min_area]
    stats["bird-only photos with a bird >= min area"] = len(cand)
    keep, KH, KTH = [], [], []
    rng = random.Random(8)
    rng.shuffle(cand)
    for it in cand:
        if it["path"].name in test_names:
            stats["drop: a bg_birds_dvb test photo"] += 1
            continue
        f = feats(it["path"])
        if leaks(f):
            stats["drop: same photo/scene as a test or val image"] += 1
            continue
        if KH:
            d = np.bitwise_count(np.array(KH, dtype=np.uint64) ^ np.uint64(f[0]))
            c = np.flatnonzero(d <= 6)
            if len(c) and (np.abs(np.stack([KTH[i] for i in c]) - f[2]).mean(axis=(1, 2)) <= 6).any():
                stats["drop: near-duplicate of a kept photo"] += 1
                continue
        keep.append(it)
        KH.append(f[0])
        KTH.append(f[2])
        if len(keep) >= a.n:
            break
    manifest = []
    for it in keep:
        stem = f"birdneg__{it['path'].stem}"[:160]
        dst = out / "images" / (stem + it["path"].suffix.lower())
        shutil.copy2(it["path"], dst)
        (out / "labels" / (stem + ".txt")).write_text("")
        manifest.append({"file": dst.name, "src": str(it["path"]), "bird_area": round(it["other_area"], 4)})
    stats["kept"] = len(keep)
    areas = sorted(m["bird_area"] for m in manifest)
    stats["bird area q10/q50/q90 (x1000)"] = "/".join(str(int(areas[int(q * (len(areas) - 1))] * 1000))
                                                      for q in (0.1, 0.5, 0.9)) if areas else "-"
    (out / "manifest.json").write_text(json.dumps({"stats": dict(stats), "items": manifest}, indent=1))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()

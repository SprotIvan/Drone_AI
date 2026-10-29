#!/usr/bin/env python3
"""
build_v81.py — dataset_v81 = dataset_v8 with three changes in TRAIN only
(plan in CHANGELOG, written before the build):

  1. close-ups: ~700 of v8's 2 247 kept — per-source quota proportional to
     v8's mix; inside a source, groups (video / series / single photo) shuffled
     (seed 8) and taken one image per group round-robin, so the kept ones are
     as varied as possible;
  2. + parts/birds_neg (600 bird-only close-ups of drone-vs-bird, no drone);
  3. + parts/station_neg160513 (131 checked drone-free frames of train
     recording 160513, frame + 6 windows = 917 files).
  val = dataset_v8 val, unchanged.

    python build_v81.py OUT_DIR [--closeups 700]

Images are HARD-LINKED (no copy on disk), labels copied. Writes data.yaml,
manifest.json (every train/val file and where it came from) and STATS.md.
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v8 import box_stats, origin  # noqa: E402

V8 = Path(r"C:\Github\drone_training\v8")
DS8 = V8 / "dataset_v8"
ADD = [V8 / "parts" / "birds_neg", V8 / "parts" / "station_neg160513"]
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--closeups", type=int, default=700)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists")
    sel = json.loads((V8 / "parts" / "closeups" / "selection.json").read_text())
    group = {e["file"]: (e["src"].split("__")[1], e["group"]) for e in sel["items"]["train"]}
    train8 = sorted(p.name for p in (DS8 / "images" / "train").iterdir() if p.suffix.lower() in IMG)
    close = [n for n in train8 if n.startswith("close__")]
    assert all(n in group for n in close), "a v8 close-up missing from selection.json"
    by_src = defaultdict(lambda: defaultdict(list))
    for n in close:
        s, g = group[n]
        by_src[s][g].append(n)
    total = len(close)
    rng = random.Random(8)
    keep = set()
    for s in sorted(by_src):
        quota = round(a.closeups * sum(len(v) for v in by_src[s].values()) / total)
        groups = sorted(by_src[s])
        rng.shuffle(groups)
        lists = {g: sorted(by_src[s][g]) for g in groups}
        for g in groups:
            rng.shuffle(lists[g])
        taken = 0
        while taken < quota and any(lists.values()):
            for g in groups:
                if lists[g] and taken < quota:
                    keep.add(lists[g].pop())
                    taken += 1
    log = Counter()
    plan = []          # (split, src image path, src label path, origin note)
    for n in train8:
        if n.startswith("close__") and n not in keep:
            log["v8 close-ups: left out (v8.1 keeps ~700)"] += 1
            continue
        plan.append(("train", DS8 / "images" / "train" / n, DS8 / "labels" / "train" / (Path(n).stem + ".txt"),
                     "close-up kept" if n.startswith("close__") else "v8 train"))
    log["v8 close-ups: kept"] = len(keep)
    for d in ADD:
        for p in sorted((d / "images").iterdir()):
            if p.suffix.lower() in IMG:
                plan.append(("train", p, d / "labels" / (p.stem + ".txt"), d.name))
                log[f"added: {d.name}"] += 1
    for p in sorted((DS8 / "images" / "val").iterdir()):
        if p.suffix.lower() in IMG:
            plan.append(("val", p, DS8 / "labels" / "val" / (p.stem + ".txt"), "v8 val"))
    seen = set()
    for split, p, _, _ in plan:
        assert (split, p.name) not in seen, p.name
        seen.add((split, p.name))
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    stats = defaultdict(Counter)
    manifest = []
    for split, p, lp, note in plan:
        os.link(p, out / "images" / split / p.name)
        shutil.copy2(lp, out / "labels" / split / lp.name)
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        with Image.open(p) as im:
            w, h = im.size
        o = "birds_neg" if p.name.startswith("birdneg__") else "station_neg160513" if p.name.startswith(
            "st8neg_") else origin(p.name)
        for key in (split, f"{split}|{o}"):
            c = stats[key]
            c["images"] += 1
            c["drone_images" if lines else "background_images"] += 1
            box_stats(lines, w, h, c)
        manifest.append({"split": split, "file": p.name, "from": note, "src": str(p), "boxes": len(lines)})
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "nc: 1\nnames: ['drone']\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=0))
    md = ["# drone_dataset_v81 — stats", "", "## Build log", ""] + [f"- {k}: {v}" for k, v in log.items()]
    for split in ("train", "val"):
        md += ["", f"## {split}", "", "| origin | images | drone | background | boxes | <16px | 16-32 | 32-96 | "
               "96-320 | >=320 | area>15% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(k for k in stats if k.startswith(split + "|")) + [split]:
            c = stats[k]
            md.append(f"| {k.split('|')[-1] if '|' in k else '**all**'} | {c['images']} | {c['drone_images']} | "
                      f"{c['background_images']} | {c['boxes']} | {c['box<16px']} | {c['box16-32px']} | "
                      f"{c['box32-96px']} | {c['box96-320px']} | {c['box>=320px']} | {c['box_area>15%']} |")
    (out / "STATS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    print("train", dict(stats["train"]))
    print("val", dict(stats["val"]))


if __name__ == "__main__":
    main()

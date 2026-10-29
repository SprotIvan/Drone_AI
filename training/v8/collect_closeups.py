#!/usr/bin/env python3
"""
collect_closeups.py — candidate CLOSE-UP drone photos for ft_v8: the photos of
the raw public sources that build_dataset.py dropped with --max-box-area 0.15
(largest drone box > 15 % of the frame), read with build_dataset.py's own
class roles, augmented-copy handling and box cleaning.

    python collect_closeups.py WORK_DIR OUT_DIR --exclude TEST_DIR [--exclude ...]

Every candidate is hashed (build_dataset.dhash_pair + 64x64 thumbnail) and
DROPPED if it is the same photo or the same scene (dHash <= 6, thumbnails
within 6 grey levels) as any image under an --exclude folder (the independent
test sets) — the test must not leak into training. Kept candidates are copied
to OUT_DIR/images + OUT_DIR/labels with a manifest (source path, licence,
box areas). Nothing in WORK_DIR is modified.
"""
import argparse
import json
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


def feats(path):
    try:
        with Image.open(path) as im:
            im.load()
            g = im.convert("L")
            h, hm = B.dhash_pair(im)
            th = np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.uint8)
            return h, hm, th, im.size
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--min-area", type=float, default=0.15)
    a = ap.parse_args()
    work, out = Path(a.work_dir), Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    cache = B.FileCache(out / "cache.json")

    # ── the excluded (test) images ──
    ex = [p for d in a.exclude for p in Path(d).rglob("*") if p.suffix.lower() in IMG]
    exf = [f for f in (feats(p) for p in ex) if f]
    EH = np.array([f[0] for f in exf], dtype=np.uint64)
    ETH = np.stack([f[2] for f in exf]).astype(np.int16) if exf else None
    print(f"{len(exf)} test images to keep out", flush=True)

    def leaks(f):
        if ETH is None:
            return False
        h, hm, th, _ = f
        th = th.astype(np.int16)
        for hh, thx in ((h, th), (hm, th[:, ::-1])):
            d = np.bitwise_count(EH ^ np.uint64(hh))
            cand = np.flatnonzero(d <= 6)
            if len(cand) and (np.abs(ETH[cand] - thx).mean(axis=(1, 2)) <= 6).any():
                return True
        return False

    manifest, stats = [], Counter()
    for slug, version, licence, cmap, note in SOURCES:
        name = f"{slug.replace('/', '__')}__v{version}"
        if "NC" in licence or slug in B.BAD_LABELS or slug in B.EXCLUDED:
            continue
        src_dir = work / "raw" / name
        if not (src_dir / "data.yaml").exists():
            continue
        items = B.collect_source(src_dir, name, slug, cmap, Counter(), cache)
        for it in items:
            if not it["boxes"]:
                continue
            area = max(bw * bh for _, _, bw, bh in it["boxes"])
            if area <= a.min_area or it["other_area"] > a.min_area:
                continue                      # not a close-up (or a big bird in it)
            f = feats(it["path"])
            if f is None:
                stats["unreadable"] += 1
                continue
            if leaks(f):
                stats[f"{name}: same photo/scene as a test image -> dropped"] += 1
                continue
            stem = f"close__{name}__{it['path'].stem}"[:180]
            dst = out / "images" / (stem + it["path"].suffix.lower())
            shutil.copy2(it["path"], dst)
            (out / "labels" / (stem + ".txt")).write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in it["boxes"]))
            manifest.append({"file": dst.name, "src": name, "licence": licence, "path": str(it["path"]),
                             "boxes": len(it["boxes"]), "max_area": round(area, 4), "size": f[3]})
            stats[f"{name}: kept"] += 1
        cache.save()
        print({k: v for k, v in stats.items() if k.startswith(name)}, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("TOTAL kept", len(manifest), dict(stats))


if __name__ == "__main__":
    main()

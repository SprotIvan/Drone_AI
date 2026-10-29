#!/usr/bin/env python3
"""
audit_dataset.py — read-only audit of a YOLO detection dataset (a folder or a
.zip with images/<split>/, labels/<split>/, data.yaml), as built by
training/build_dataset.py.

    python audit_dataset.py DATASET(.zip|dir) OUT_DIR [--name v5]

Nothing is changed. Writes OUT_DIR/<name>_audit.json (every finding, per file)
and OUT_DIR/<name>_audit.md (the summary).

CHECKS
  files     unreadable / truncated / zero-size images, images without a label
            file, label files without an image, non-image files
  yaml      data.yaml present, nc / names, split paths that exist
  labels    wrong field count, non-numeric values, class id != 0, centre or
            size outside 0..1, zero / negative width or height, box spilling
            over the frame edge (> 0.5 % of the frame), boxes under 2 px on the
            long side at the image's own resolution, a box covering > 90 % of
            the frame, two boxes on the same object (IoU > 0.9)
  dupes     byte-identical images (md5) inside a split and across splits;
            near-duplicates by build_dataset.py's own 64-bit dHash (<= 4 bits,
            direct or mirrored) with the same boxes; same-scene pairs
            (dHash <= 6 and 64x64 thumbnails within 6 grey levels) across
            train/val — the leak build_dataset.py's clustering is meant to stop
  balance   drones per long side at 640 px, per source, per image brightness,
            light vs dark drones (box mean luminance vs a ring around it),
            boxes per image, backgrounds per source
"""
import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False          # a truncated file is a finding
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_BINS = [("<8", 0, 8), ("8-16", 8, 16), ("16-32", 16, 32), ("32-64", 32, 64),
             ("64-128", 64, 128), (">=128", 128, 1e9)]


class Store:
    """Uniform read access to a dataset folder or zip."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = self.path.suffix.lower() == ".zip"
        if self.zip:
            with zipfile.ZipFile(self.path) as z:
                self.names = [n for n in z.namelist() if not n.endswith("/")]
        else:
            self.names = [p.relative_to(self.path).as_posix()
                          for p in self.path.rglob("*") if p.is_file()]

    def read(self, name, z=None):
        if self.zip:
            return z.read(name)
        return (self.path / name).read_bytes()


def source_of(name):
    stem = Path(name).stem
    return stem.split("__")[0] if "__" in stem else re.sub(r"_\d+$", "", stem)


def bits(a):
    d = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in d), 2)


def parse_label(text):
    rows, problems = [], []
    for k, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5:
            problems.append(f"line {k + 1}: {len(p)} fields")
            continue
        try:
            c = float(p[0])
            cx, cy, w, h = (float(v) for v in p[1:])
        except ValueError:
            problems.append(f"line {k + 1}: not numeric")
            continue
        if c != 0:
            problems.append(f"line {k + 1}: class {p[0]}")
        rows.append((cx, cy, w, h))
    return rows, problems


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    u = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / u if u > 0 else 0.0


def job(args):
    """One image: integrity, label checks, hashes, brightness, per-box stats."""
    store_path, img_name, lab_name = args
    st = Store.__new__(Store)
    st.path, st.zip = Path(store_path), str(store_path).lower().endswith(".zip")
    z = zipfile.ZipFile(st.path) if st.zip else None
    out = {"img": img_name, "lab": lab_name, "issues": []}
    try:
        data = st.read(img_name, z)
        out["bytes"] = len(data)
        out["md5"] = hashlib.md5(data).hexdigest()
        if not data:
            out["issues"].append("zero-size image")
            return out
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception as exc:
            out["issues"].append(f"unreadable image: {exc}")
            return out
        w, h = im.size
        out["w"], out["h"] = w, h
        g = im.convert("L")
        a = np.asarray(g, dtype=np.float32)
        out["mean_lum"] = float(a.mean())
        g9 = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
        out["dh"], out["dhm"] = bits(g9), bits(g9[:, ::-1])
        out["th"] = np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.uint8).tobytes().hex()
        boxes = []
        if lab_name is None:
            out["issues"].append("no label file")
        else:
            text = st.read(lab_name, z).decode("utf-8", "replace")
            rows, probs = parse_label(text)
            out["issues"] += [f"label {p}" for p in probs]
            for cx, cy, bw, bh in rows:
                if not (0 <= cx <= 1 and 0 <= cy <= 1):
                    out["issues"].append(f"centre outside 0..1 ({cx:.4f},{cy:.4f})")
                if bw <= 0 or bh <= 0:
                    out["issues"].append(f"zero/negative size ({bw:.4f}x{bh:.4f})")
                    continue
                if bw > 1 or bh > 1:
                    out["issues"].append(f"size > 1 ({bw:.4f}x{bh:.4f})")
                spill = max(0 - (cx - bw / 2), (cx + bw / 2) - 1, 0 - (cy - bh / 2), (cy + bh / 2) - 1)
                if spill > 0.005:
                    out["issues"].append(f"box over the edge by {spill:.3f}")
                lp = max(bw * w, bh * h)
                if lp < 2:
                    out["issues"].append(f"box under 2 px ({lp:.1f})")
                if bw * bh > 0.9:
                    out["issues"].append(f"box covers {bw * bh:.0%} of the frame")
                # luminance of the box vs a ring around it (light / dark drone)
                x0, y0 = int(max(0, (cx - bw / 2) * w)), int(max(0, (cy - bh / 2) * h))
                x1, y1 = int(min(w, (cx + bw / 2) * w)) or 1, int(min(h, (cy + bh / 2) * h)) or 1
                inner = a[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
                px, py = max(2, (x1 - x0) // 2), max(2, (y1 - y0) // 2)
                ring = a[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
                rmean = (ring.sum() - inner.sum()) / max(1, ring.size - inner.size)
                boxes.append({"cx": cx, "cy": cy, "w": bw, "h": bh,
                              "long640": lp * 640.0 / max(w, h),
                              "area": bw * bh,
                              "lum": float(inner.mean()) if inner.size else 0.0,
                              "ring": float(rmean)})
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if iou(rows[i], rows[j]) > 0.9:
                        out["issues"].append("two boxes on one object (IoU>0.9)")
        out["boxes"] = boxes
    finally:
        if z:
            z.close()
    return out


def size_bin(px):
    for name, lo, hi in SIZE_BINS:
        if lo <= px < hi:
            return name
    return SIZE_BINS[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--name", default="dataset")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = Store(a.dataset)
    rep = {"dataset": str(a.dataset), "files": len(st.names), "yaml": None}

    # ── data.yaml ──
    yaml_names = [n for n in st.names if n.endswith("data.yaml")]
    if yaml_names:
        with (zipfile.ZipFile(st.path) if st.zip else open(st.path / yaml_names[0], "rb")) as h:
            text = (h.read(yaml_names[0]) if st.zip else h.read()).decode()
        rep["yaml"] = text
    splits = sorted({n.split("/")[1] for n in st.names if n.startswith("images/") and n.count("/") >= 2})
    rep["splits"] = splits

    # ── pairing ──
    imgs = {s: sorted(n for n in st.names if n.startswith(f"images/{s}/")
                      and Path(n).suffix.lower() in IMG_EXT) for s in splits}
    labs = {s: {Path(n).stem: n for n in st.names if n.startswith(f"labels/{s}/") and n.endswith(".txt")}
            for s in splits}
    other = [n for n in st.names if n.startswith("images/") and Path(n).suffix.lower() not in IMG_EXT]
    rep["non_image_files_in_images"] = other
    orphans = {s: sorted(set(labs[s]) - {Path(n).stem for n in imgs[s]}) for s in splits}
    rep["orphan_labels"] = {s: v for s, v in orphans.items() if v}
    work = [(str(st.path), n, labs[s].get(Path(n).stem)) for s in splits for n in imgs[s]]
    split_of = {n: s for s in splits for n in imgs[s]}
    print(f"{len(work)} images in {splits}", flush=True)

    with ProcessPoolExecutor() as pool:
        res = list(pool.map(job, work, chunksize=64))
    print("read all images", flush=True)

    # ── per-image issues ──
    issues = {r["img"]: r["issues"] for r in res if r["issues"]}
    kinds = Counter(re.sub(r"[\d.(),x%-]+", "#", i) for v in issues.values() for i in v)

    # ── exact duplicates ──
    by_md5 = defaultdict(list)
    for r in res:
        if "md5" in r:
            by_md5[r["md5"]].append(r["img"])
    md5_groups = [g for g in by_md5.values() if len(g) > 1]
    md5_cross = [g for g in md5_groups if len({split_of[n] for n in g}) > 1]

    # ── near duplicates / same scene across splits (train vs every other) ──
    ok = [r for r in res if "dh" in r]
    near = {"dup_same_split": 0, "dup_cross_split": [], "scene_cross_split": []}
    if "train" in splits:
        tr = [r for r in ok if split_of[r["img"]] == "train"]
        H = np.array([r["dh"] for r in tr], dtype=np.uint64)
        TH = np.stack([np.frombuffer(bytes.fromhex(r["th"]), dtype=np.uint8).reshape(64, 64) for r in tr])
        for r in ok:
            if split_of[r["img"]] == "train":
                continue
            th = np.frombuffer(bytes.fromhex(r["th"]), dtype=np.uint8).reshape(64, 64).astype(np.int16)
            best = None
            for hh, thx in ((r["dh"], th), (r["dhm"], th[:, ::-1])):
                d = np.bitwise_count(H ^ np.uint64(hh))
                cand = np.flatnonzero(d <= 6)
                if len(cand):
                    mads = np.abs(TH[cand].astype(np.int16) - thx).mean(axis=(1, 2))
                    for i, m in zip(cand, mads):
                        k = "dup" if d[i] <= 4 and m <= 3 else "scene" if m <= 6 else None
                        if k and (best is None or (k == "dup", -m) > best[0]):
                            best = ((k == "dup", -float(m)), k, tr[i]["img"])
            if best:
                key = "dup_cross_split" if best[1] == "dup" else "scene_cross_split"
                near[key].append([r["img"], best[2]])

    # ── balance ──
    bal = {}
    for s in splits:
        rs = [r for r in ok if split_of[r["img"]] == s]
        boxes = [b for r in rs for b in r.get("boxes", [])]
        src_img = Counter(source_of(r["img"]) for r in rs)
        src_bg = Counter(source_of(r["img"]) for r in rs if not r.get("boxes"))
        lum = Counter("dark (<70)" if r["mean_lum"] < 70 else "bright (>180)" if r["mean_lum"] > 180
                      else "normal" for r in rs)
        tone = Counter("lighter than surround" if b["lum"] - b["ring"] > 15 else
                       "darker than surround" if b["lum"] - b["ring"] < -15 else "similar"
                       for b in boxes)
        bal[s] = {"images": len(rs), "backgrounds": sum(1 for r in rs if not r.get("boxes")),
                  "boxes": len(boxes),
                  "size_long_side_at_640": dict(Counter(size_bin(b["long640"]) for b in boxes)),
                  "boxes_per_image": dict(Counter(min(len(r.get("boxes", [])), 5) for r in rs)),
                  "image_brightness": dict(lum), "drone_tone": dict(tone),
                  "large_box_area_gt_15pct": sum(1 for b in boxes if b["area"] > 0.15),
                  "images_per_source": dict(src_img.most_common()),
                  "backgrounds_per_source": dict(src_bg.most_common())}

    rep.update({"images": len(res), "issue_kinds": dict(kinds.most_common()),
                "images_with_issues": len(issues), "issues": issues,
                "md5_duplicate_groups": len(md5_groups),
                "md5_duplicate_images_extra": sum(len(g) - 1 for g in md5_groups),
                "md5_cross_split_groups": md5_cross,
                "near": {"dup_cross_split": len(near["dup_cross_split"]),
                         "scene_cross_split": len(near["scene_cross_split"])},
                "near_pairs": near, "balance": bal})
    (out / f"{a.name}_audit.json").write_text(json.dumps(rep, indent=1))

    # ── markdown ──
    L = [f"# Audit: {a.dataset}", "", f"- files in archive/folder: {rep['files']}",
         f"- splits: {', '.join(splits)}; images: {len(res)}",
         f"- data.yaml: {'present' if rep['yaml'] else 'MISSING'}",
         f"- non-image files under images/: {len(other)}",
         f"- label files without an image: {sum(len(v) for v in orphans.values())}",
         f"- images with at least one finding: {len(issues)}", "", "## Findings by kind", ""]
    L += [f"- {k}: {v}" for k, v in kinds.most_common()] or ["- none"]
    L += ["", "## Duplicates and leakage", "",
          f"- byte-identical groups: {len(md5_groups)} ({rep['md5_duplicate_images_extra']} extra copies),"
          f" of them across splits: {len(md5_cross)}",
          f"- val/test images with a near-duplicate (same photo) in train: {len(near['dup_cross_split'])}",
          f"- val/test images with a same-scene frame in train: {len(near['scene_cross_split'])}", ""]
    for s, b in bal.items():
        L += [f"## Split `{s}`", "", f"- images {b['images']}, backgrounds {b['backgrounds']}, boxes {b['boxes']}",
              f"- drone long side at 640: {b['size_long_side_at_640']}",
              f"- boxes larger than 15% of the frame: {b['large_box_area_gt_15pct']}",
              f"- boxes per image (5 = 5+): {b['boxes_per_image']}",
              f"- image brightness: {b['image_brightness']}",
              f"- drone vs surround: {b['drone_tone']}",
              f"- images per source: {b['images_per_source']}",
              f"- backgrounds per source: {b['backgrounds_per_source']}", ""]
    (out / f"{a.name}_audit.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    sys.exit(main())

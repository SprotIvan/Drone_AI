#!/usr/bin/env python3
"""
build_dataset.py — merge the downloaded drone datasets into ONE clean,
single-class YOLO dataset for fine-tuning the station's YOLO26n.

    python build_dataset.py <work_dir> [--include-nc] [--val 0.1] [--seed 0]

<work_dir>/raw/<source>/      what fetch_datasets.py downloaded
<work_dir>/station/           frames from the station (optional):
    *.jpg|*.png               with a same-name .txt  -> labelled drone frame
                              without a .txt          -> background (negative)
<work_dir>/drone_dataset/     OUTPUT: images/{train,val}, labels/{train,val},
                              data.yaml, SOURCES.md, stats.json

WHAT IT DOES, AND WHY
  1. Augmented copies are removed — only where it is safe. Roboflow names
     an exported image <photo>.rf.<hash>.jpg, but a name is NOT unique per
     photo: many projects hold different photos with the same name ("141.jpg"
     from different folders), even inside one split, and a valid/test photo
     often shares its name with an unrelated train photo. So names group
     copies only in the TRAIN split of projects listed in AUGMENTED_TRAIN,
     after checking no name has more copies than the augmentation made; the
     least-augmented copy (least impulse noise and rotation padding) is kept.
  2. Classes are merged by the class map in fetch_datasets.SOURCES: drone
     classes -> "drone"; "background" classes (birds) lose their boxes and
     the image may stay as a negative; "drop" classes exclude the image.
     A class the map does not name stops the build.
  3. Images with NO boxes in a public source are excluded: in a sample of
     24, three held an unlabelled drone or aircraft — as "background" they
     would teach that a drone is not a drone. Negatives come from the
     station's own frames instead, where it is known that there is none.
  4. Close-ups are excluded (--max-box-area): the station never sees a
     drone — or a bird — that fills a sixth of the frame.
  5. Near-duplicates are removed ACROSS all sources: same flip-invariant
     64-bit difference hash (within DUP_HAMMING) AND the same boxes. The
     boxes matter: in a video from a static camera every frame has almost
     the same hash, and without them the whole video would collapse into
     one frame; with them, frames are kept whenever the drone has moved.
  6. The train/val split is done by CLUSTERS of similar frames (consecutive
     video frames stay on one side). Letting near-identical frames sit in
     both train and val is what makes a validation score meaningless — the
     current model scored mAP50 0.994 on its own validation and 0.04 on a
     drone at the station. Clusters link frames of ONE source only: the
     64-bit hash cannot tell plain-sky scenes apart — frames from different
     sources share IDENTICAL hashes — so linking across sources chained
     16 511 of 21 883 images into one cluster. Each source gives ~10 % of
     its images to val, from clusters no bigger than a quarter of that
     share; its big look-alike chains stay in train.
  7. Images larger than 1280 px on their long side are downscaled to 1280
     (JPEG q95): training runs at 640, so nothing usable is lost, and the
     upload to Colab halves.

Hashes and copy scores are cached in <work_dir>/build_cache.json (keyed by
path, size and modification time): a rerun takes seconds, not ~12 minutes.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_datasets import SOURCES  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1280
DUP_HAMMING = 4          # <= this = the same photo (re-upload, re-encode)
DUP_BOX_TOL = 0.02       # ...if its boxes also match within this (of frame)
CLUSTER_HAMMING = 6      # <= this, same source = same scene: one side only

# Projects whose Roboflow export holds N augmented copies of every TRAIN
# photo (valid/test are never augmented). Checked on the downloads: every
# train name occurs exactly N times (zhejiang: 2-3), valid/test names once.
AUGMENTED_TRAIN = {
    "yogith-nams8/anti-uav-s8wri": 3,
    "get/uav-detect-pfiqs": 3,
    "zhejiang-university-china-dliq1/drones-detection-with-yolov8": 3,
}

# Downloaded, but NOT trained on (synth_small_drones.py may still cut
# drones out of them). Remove a line to bring a source back.
EXCLUDED = {
    "crypt2/dji-drones": "augmentation baked into the files before upload: "
                         "about half of a 24-image sample was rotated "
                         "90-180 deg (sky below ground), smeared by "
                         "replicate padding or heavily noised — scenes the "
                         "station never sees; the other sources are clean",
}

# Not used at all — neither trained on nor cut out for the synthetic set —
# because the LABELS are wrong, not the images.
BAD_LABELS = {
    "passants-workspace/dut-anti-uav-detection":
        "boxes 1.5-3x larger than the drone and often stretched vertically: "
        "22 of its 24 val drones counted as missed while the ft_v2 HEF boxed "
        "the drone tightly (error analysis 29.09.2026) — loose boxes teach "
        "loose boxes",
}


# ── Hashing ────────────────────────────────────────────────────

def dhash_pair(img: Image.Image):
    """64-bit difference hashes of the image and of its mirror image.
    Kept apart, not merged into one flip-invariant hash: a match must say
    WHICH way round two images match, or the boxes cannot be compared."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR),
                   dtype=np.int16)
    def bits(a):
        d = (a[:, 1:] > a[:, :-1]).flatten()
        return int("".join("1" if b else "0" for b in d), 2)
    return bits(g), bits(g[:, ::-1])


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FileCache:
    """Per-file results kept between runs, keyed by what was computed, the
    path, its size and its modification time; entries not used by a run
    are dropped when it saves."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.used = set()

    def map(self, tag, fn, paths, chunksize=32):
        """[fn(p) for p in paths], computing only what is not cached."""
        keys = []
        for p in paths:
            st = os.stat(p)
            keys.append(f"{tag}|{p}|{st.st_size}|{st.st_mtime_ns}")
        todo = [(k, p) for k, p in zip(keys, paths) if k not in self.data]
        if todo:
            with Pool() as pool:
                res = pool.map(fn, [p for _, p in todo], chunksize=chunksize)
            self.data.update(zip((k for k, _ in todo), res))
        self.used.update(keys)
        return [self.data[k] for k in keys]

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: self.data[k] for k in self.used}),
                       encoding="utf-8")
        tmp.replace(self.path)


class HashIndex:
    """The hashes kept so far; `near` gives the indices of those within
    `limit` bits. Brute force — XOR and popcount against all of them in
    numpy: exact at any limit, and as fast on sky frames as on anything
    else. (Buckets of 8-bit hash chunks took ~17 minutes here: plain-sky
    frames all share the same few chunk values.)"""

    def __init__(self, capacity: int):
        self.items = np.zeros(max(1, capacity), dtype=np.uint64)
        self.n = 0

    def near(self, h: int, limit: int):
        d = np.bitwise_count(self.items[:self.n] ^ np.uint64(h))
        return np.flatnonzero(d <= limit).tolist()

    def add(self, h: int) -> int:
        self.items[self.n] = h
        self.n += 1
        return self.n - 1


# ── Reading one source ─────────────────────────────────────────

def read_names(data_yaml: Path):
    """Class names from a Roboflow data.yaml, without needing PyYAML."""
    text = data_yaml.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",")]
    names, block = [], False
    for line in text.splitlines():
        if line.startswith("names:"):
            block = True
            continue
        if block:
            mm = re.match(r"\s+(?:-\s*|\d+:\s*)(.+)", line)
            if not mm:
                break
            names.append(mm.group(1).strip().strip("'\""))
    return names


def parse_label(path: Path):
    """[(cls, cx, cy, w, h)] from a YOLO txt; polygons become their bbox."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
        else:                                   # polygon x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        out.append((cls, cx, cy, w, h))
    return out


def hash_one(path):
    """((hash, mirror hash), (w, h)) or None — top level so a process pool
    can run it."""
    try:
        with Image.open(path) as im:
            im.load()
            return dhash_pair(im), im.size
    except Exception:
        return None


def artifact_score(path):
    """How visibly augmented one copy is: the share of impulse-noise pixels
    plus the share of the frame border that is flat padding (rotation
    fill). Meaningful only between copies of ONE photo."""
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            a = np.asarray(g, dtype=np.int16)
            m = np.asarray(g.filter(ImageFilter.MedianFilter(3)),
                           dtype=np.int16)
    except Exception:
        return float("inf")
    noise = float((np.abs(a - m) > 48).mean())
    b = max(2, min(a.shape) // 32)
    ring = np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                           a[:, :b].ravel(), a[:, -b:].ravel()])
    fill = float(((ring <= 2) | (ring >= 253)).mean())
    return noise + fill


def same_boxes(a, b, tol=DUP_BOX_TOL):
    """True when two box lists mark the same objects in the same places."""
    if len(a) != len(b):
        return False
    free = list(b)
    for cx, cy, w, h in a:
        for j, (cx2, cy2, w2, h2) in enumerate(free):
            if abs(cx - cx2) <= tol and abs(cy - cy2) <= tol \
                    and abs(w - w2) <= max(tol, 0.3 * w) \
                    and abs(h - h2) <= max(tol, 0.3 * h):
                del free[j]
                break
        else:
            return False
    return True


def mirrored(boxes):
    return [(1 - cx, cy, w, h) for cx, cy, w, h in boxes]


def write_one(job):
    """Copy or downscale one image and write its label file."""
    src, dst, label_path, boxes = job
    src, dst = Path(src), Path(dst)
    with Image.open(src) as im:
        w, h = im.size
        if max(w, h) <= MAX_SIDE and src.suffix.lower() in (".jpg", ".jpeg") \
                and im.mode == "RGB":
            shutil.copyfile(src, dst)           # byte-exact, no re-encode
        else:
            im = im.convert("RGB")
            if max(w, h) > MAX_SIDE:
                s = MAX_SIDE / max(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            im.save(dst, quality=95)
    with open(label_path, "w") as f:
        for cx, cy, bw, bh in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


ROLES = ("drone", "background", "drop")


def class_roles(names, cmap, source):
    """The role of every class index; stops on a class the map does not
    give a role — a silent default is how drones become "background"."""
    roles = []
    for n in names:
        role = cmap.get(n, cmap.get("*"))
        if role not in ROLES:
            raise SystemExit(f"{source}: class {n!r} has role {role!r} — "
                             f"inspect it and give it one of {ROLES} in "
                             f"fetch_datasets.SOURCES")
        roles.append(role)
    return roles


def clean_box(cx, cy, w, h):
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    w, h = x1 - x0, y1 - y0
    if w < 0.002 or h < 0.002:
        return None
    return ((x0 + x1) / 2, (y0 + y1) / 2, w, h)


def images_in(folder: Path):
    return [p for p in sorted(folder.glob("*")) if p.suffix.lower() in IMG_EXT]


def one_copy_per_photo(train, n_aug, source, count, cache):
    """The least-augmented copy of every train photo of an AUGMENTED_TRAIN
    project; stops if the names do not look like a plain x n_aug export."""
    groups = defaultdict(list)
    for p in train:
        groups[p.name.split(".rf.")[0]].append(p)
    over = sum(len(v) > n_aug for v in groups.values())
    if over:
        raise SystemExit(f"{source}: {over} train names have more than "
                         f"{n_aug} copies — not a plain x{n_aug} export; "
                         f"check it before grouping by name")
    multi = [str(p) for v in groups.values() if len(v) > 1 for p in v]
    score = dict(zip(multi, cache.map("artifact", artifact_score, multi)))
    count["augmented_copies_dropped"] += len(train) - len(groups)
    return [min(v, key=lambda p: score.get(str(p), 0.0))
            for v in groups.values()]


def collect_source(src_dir: Path, name: str, slug: str, cmap: dict,
                   count: Counter, cache: FileCache):
    """One item per photo: {"src", "path", "boxes", "other_area"}.
    `count` receives why images were left out."""
    roles = class_roles(read_names(src_dir / "data.yaml"), cmap, name)
    photos = images_in(src_dir / "valid" / "images") \
        + images_in(src_dir / "test" / "images")
    train = images_in(src_dir / "train" / "images")
    n_aug = AUGMENTED_TRAIN.get(slug)
    photos += one_copy_per_photo(train, n_aug, name, count, cache) \
        if n_aug else train
    items = []
    for img in photos:
        raw = parse_label(img.parent.parent / "labels" / (img.stem + ".txt"))
        if not raw:
            count["no_boxes_excluded"] += 1
            continue
        boxes, other_area, had_drone, drop = [], 0.0, False, False
        for cls, cx, cy, w, h in raw:
            if cls >= len(roles):
                raise SystemExit(f"{name}: {img.name} uses class {cls}; its "
                                 f"data.yaml names {len(roles)}")
            if roles[cls] == "drone":
                had_drone = True
                b = clean_box(cx, cy, w, h)
                if b:
                    boxes.append(b)
            elif roles[cls] == "background":
                other_area = max(other_area, w * h)
            else:
                drop = True
        if drop:
            count["drop_class_excluded"] += 1
            continue
        if had_drone and not boxes:        # only degenerate drone boxes:
            count["degenerate_excluded"] += 1   # not a background either
            continue
        items.append({"src": name, "path": img, "boxes": boxes,
                      "other_area": other_area})
    return items


# ── Main ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--include-nc", action="store_true")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-box-area", type=float, default=0.15,
                    help="drop public images whose largest box (drone, or "
                         "bird kept as background) covers more than this "
                         "fraction of the image (close-ups); 0 disables")
    ap.add_argument("--stats-only", action="store_true",
                    help="print the per-source size statistics, write nothing")
    ap.add_argument("--zip", action="store_true",
                    help="also pack the dataset into <work_dir>/drone_dataset.zip "
                         "(its CONTENTS at the zip root, as the notebook expects)")
    args = ap.parse_args()
    work = Path(args.work_dir)
    out = work / "drone_dataset"
    stats = Counter()
    per_src = defaultdict(Counter)
    cache = FileCache(work / "build_cache.json")

    items = []
    used_sources = []
    for slug, version, licence, cmap, note in SOURCES:
        if "NC" in licence and not args.include_nc:
            continue
        name = f"{slug.replace('/', '__')}__v{version}"
        if slug in BAD_LABELS:
            print(f"not used {name}: {BAD_LABELS[slug]}")
            continue
        if slug in EXCLUDED:
            print(f"excluded {name}: {EXCLUDED[slug]}")
            # still credited: the synthetic set holds drones cut out of it
            used_sources.append((name, slug, version, licence,
                                 "not trained on directly (see "
                                 "build_dataset.EXCLUDED); drones cut out "
                                 "of it for the synthetic set"))
            continue
        src_dir = work / "raw" / name
        if not (src_dir / "data.yaml").exists():
            print(f"missing {name} — skipped")
            continue
        got = collect_source(src_dir, name, slug, cmap, per_src[name], cache)
        # ⚠️ Close-ups: the station never sees a drone (or a bird) this
        # large, and whole-frame boxes teach nothing about a 10-30 px
        # target. Such an image is DROPPED, not kept as a background — it
        # contains a drone. (synth_small_drones.py reuses them as cut-outs.)
        kept_src = []
        for it in got:
            area = max([bw * bh for _, _, bw, bh in it["boxes"]]
                       + [it["other_area"]])
            if area > 0.20:
                per_src[name]["over_20pct_area"] += 1
            if area > 0.15:
                per_src[name]["over_15pct_area"] += 1
            if args.max_box_area and area > args.max_box_area:
                per_src[name]["close_ups_excluded"] += 1
                continue
            kept_src.append(it)
        items += kept_src
        used_sources.append((name, slug, version, licence, note))
        c = per_src[name]
        print(f"read {name}: {len(got)} photos with boxes | excluded: "
              f"{c['augmented_copies_dropped']} augmented copies, "
              f"{c['no_boxes_excluded']} without boxes, "
              f"{c['drop_class_excluded']} drop-class, "
              f"{c['degenerate_excluded']} degenerate, "
              f"{c['close_ups_excluded']} close-ups", flush=True)

    for folder, label, note in (
            ("station", "this station", "frames from the station's own cameras"),
            ("synthetic", "synthetic", "small drones pasted onto station "
                                       "backgrounds (synth_small_drones.py)"),
            ("synthetic_clutter", "synthetic", "small drones pasted onto the "
             "TRAIN backgrounds — sky, trees, buildings, weather, birds "
             "(synth_small_drones.py --clutter)")):
        root = work / folder
        if not root.exists():
            continue
        for img in sorted(root.rglob("*")):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = img.with_suffix(".txt")
            boxes = [clean_box(*b[1:]) for b in parse_label(lab)]
            items.append({"src": folder, "path": img,
                          "boxes": [b for b in boxes if b],
                          "other_area": 0.0, "force_train": True})
        used_sources.append((folder, label, "-", "own data", note))

    # Fixed-split additions (prepare_extra.py): backgrounds without drones
    # (<work>/negatives/<part>/<source>/) and the owner's labelled photos
    # (<work>/user_photos/<part>/). Their train/val split was made by the
    # preparer — by time for fixed cameras, so no val frame has a near-copy
    # in train — and is kept as is. Held-out backgrounds are what measures
    # false positives on scenes the model never saw.
    neg_manifest = {}
    if (work / "negatives" / "manifest.json").exists():
        neg_manifest = json.loads(
            (work / "negatives" / "manifest.json").read_text())
    for folder in ("negatives", "user_photos"):
        root = work / folder
        if not root.exists():
            continue
        srcs = set()
        for part in ("train", "val"):
            for img in sorted((root / part).rglob("*")):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                src = (f"negatives/{img.parent.name}" if folder == "negatives"
                       else folder)
                boxes = [clean_box(*b[1:])
                         for b in parse_label(img.with_suffix(".txt"))]
                items.append({"src": src, "path": img,
                              "boxes": [b for b in boxes if b],
                              "other_area": 0.0,
                              "force_train": part == "train",
                              "force_val": part == "val"})
                srcs.add(src)
        for src in sorted(srcs):
            name = src.split("/")[-1]
            m = neg_manifest.get(name, {})
            used_sources.append((
                src, m.get("kaggle", "the owner's own photos"), "-",
                m.get("licence", "own data"),
                "backgrounds, no drone (prepare_extra.py)"
                if folder == "negatives" else
                "the owner's drone, labelled (prepare_extra.py)"))

    # Hash, drop near-duplicates, cluster for the split.
    index = HashIndex(len(items))
    index_boxes, index_src = [], []
    kept = []
    parent = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    print(f"hashing {len(items)} images...", flush=True)
    hashes = cache.map("dhash", hash_one, [str(it["path"]) for it in items],
                       chunksize=64)
    cache.save()
    for n, (it, hs) in enumerate(zip(items, hashes)):
        if hs is None:
            stats["unreadable"] += 1
            continue
        (h, hm), it["size"] = hs
        # Only DIRECT hashes are indexed. A kept image matching this one's
        # hash is the same way round; one matching its MIRROR hash is a
        # flipped copy, whose boxes are compared mirrored. An image that
        # matches both ways (a near-symmetric scene) is compared directly.
        direct = set(index.near(h, DUP_HAMMING))
        flipped = set(index.near(hm, DUP_HAMMING)) - direct
        if not (it.get("force_train") or it.get("force_val")) and (
                any(same_boxes(it["boxes"], index_boxes[j]) for j in direct)
                or any(same_boxes(mirrored(it["boxes"]), index_boxes[j])
                       for j in flipped)):
            stats["near_duplicates_dropped"] += 1
            per_src[it["src"]]["duplicates"] += 1
            continue
        k = index.add(h)
        index_boxes.append(it["boxes"])
        index_src.append(it["src"])
        parent.append(k)
        for j in set(index.near(h, CLUSTER_HAMMING)) \
                | set(index.near(hm, CLUSTER_HAMMING)):
            if j != k and index_src[j] == it["src"]:
                parent[find(j)] = find(k)
        it["k"] = k
        kept.append(it)
        if n % 5000 == 0:
            print(f"  hashed {n}/{len(items)}", flush=True)

    # Box sizes per source: the LONG side in px as it appears in a
    # 640-letterboxed training image.
    edges = [0, 8, 16, 32, 64, 128, 1e9]
    labels = ["<8", "8-16", "16-32", "32-64", "64-128", ">=128"]
    size_table, long_px = {}, defaultdict(list)
    for it in kept:
        w, h = it["size"]
        scale = 640.0 / max(w, h)
        row = size_table.setdefault(it["src"], Counter())
        row["images"] += 1
        row["backgrounds"] += 0 if it["boxes"] else 1
        for _, _, bw, bh in it["boxes"]:
            px = max(bw * w, bh * h) * scale
            long_px[it["src"]].append(px)
            for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
                if lo <= px < hi:
                    row[lab] += 1
    print("\nBOX LONG SIDE AT 640 px, per source (after all exclusions and "
          "dedup):")
    print(f"  {'source':44s} {'images':>7s} {'bg':>5s} {'dups':>6s} "
          + " ".join(f"{l:>7s}" for l in labels) + "  median")
    for src, row in sorted(size_table.items()):
        med = np.median(long_px[src]) if long_px[src] else 0.0
        print(f"  {src[:44]:44s} {row['images']:7d} {row['backgrounds']:5d} "
              f"{per_src[src]['duplicates']:6d} "
              + " ".join(f"{row[l]:7d}" for l in labels) + f"  {med:6.1f}")
    allpx = np.array([p for v in long_px.values() for p in v])
    if len(allpx):
        print(f"  {'ALL':44s} {len(kept):7d} "
              f"{sum(r['backgrounds'] for r in size_table.values()):5d} "
              f"{stats['near_duplicates_dropped']:6d} "
              + " ".join(f"{int(((allpx >= lo) & (allpx < hi)).sum()):7d}"
                         for lo, hi in zip(edges[:-1], edges[1:]))
              + f"  {np.median(allpx):6.1f}")
    print("\nclose-ups per source (images whose largest box covers >15% / "
          ">20% of the frame):")
    for src, c in sorted(per_src.items()):
        if c["over_15pct_area"] or c["over_20pct_area"]:
            print(f"  {src[:52]:52s} >15%: {c['over_15pct_area']:6d}   "
                  f">20%: {c['over_20pct_area']:6d}")
    # Split by cluster, per source (clusters never span sources): each
    # public source gives ~args.val of its images to val, drawn from
    # clusters no bigger than a quarter of that share. A bigger cluster — a
    # long static-camera video, or a chain of look-alike sky frames — stays
    # in train whole rather than turning val into copies of one scene.
    rng = random.Random(args.seed)
    clusters = defaultdict(list)
    for it in kept:
        clusters[find(it["k"])].append(it)
    by_src = defaultdict(list)
    for c in sorted(clusters):
        if not any(it.get("force_train") or it.get("force_val")
                   for it in clusters[c]):
            by_src[clusters[c][0]["src"]].append(c)
    val_keys = set()
    print("\nval split per source (images in val / of the source; largest "
          "cluster kept in train):")
    for src, keys in sorted(by_src.items()):
        n_src = sum(len(clusters[c]) for c in keys)
        target, n = args.val * n_src, 0
        cap = max(1.0, target / 4)
        rng.shuffle(keys)
        for c in keys:
            if n >= target:
                break
            if len(clusters[c]) <= cap:
                val_keys.add(c)
                n += len(clusters[c])
        big = max(len(clusters[c]) for c in keys)
        print(f"  {src[:44]:44s} {n:5d} / {n_src:6d} ({n / n_src:5.1%})   "
              f"largest cluster {big}")
    csize = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"clusters: {len(csize)}; largest: {csize[:8]}")
    if args.stats_only:
        cache.save()
        return
    if out.exists():
        shutil.rmtree(out)

    # Write.
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, it in enumerate(kept):
        if it.get("force_val"):
            split = "val"
        elif it.get("force_train"):
            split = "train"
        else:
            split = "val" if find(it["k"]) in val_keys else "train"
        stem = f"{it['src'].replace('/', '_')[:40]}_{i:06d}"   # negatives/<source>: no subfolder
        jobs.append((str(it["path"]),
                     str(out / "images" / split / f"{stem}.jpg"),
                     str(out / "labels" / split / f"{stem}.txt"),
                     it["boxes"]))
        key = "drone_images" if it["boxes"] else "background_images"
        stats[f"{split}_{key}"] += 1
        per_src[it["src"]][key] += 1
        per_src[it["src"]]["boxes"] += len(it["boxes"])
    print(f"writing {len(jobs)} images...", flush=True)
    with Pool() as pool:
        pool.map(write_one, jobs, chunksize=64)

    stats["boxes"] = len(allpx)
    stats["boxes_under_16px_at_640"] = int((allpx < 16).sum())
    stats["boxes_16_to_32px_at_640"] = int(((allpx >= 16) & (allpx < 32)).sum())
    stats["boxes_32px_and_up_at_640"] = int((allpx >= 32).sum())

    # ⚠️ No "path:" line. Ultralytics resolves a relative `path` against the
    # CURRENT directory (checked with 8.4.68: "path: ." made Colab look for
    # /content/images/train); without it, the root is this file's folder.
    (out / "data.yaml").write_text(
        "# drone detection - merged by build_dataset.py\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: drone\n", encoding="utf-8")
    lines = ["# Sources\n",
             "Merged into one class `drone` by the class maps in "
             "fetch_datasets.py; close-ups, images without boxes and images "
             "of unclean classes excluded (see build_dataset.py). "
             "Attribution as required by each licence.\n",
             "| Source | Version | Licence | Images used | Note |",
             "|---|---|---|---|---|"]
    for key, slug, version, licence, note in used_sources:
        n = size_table.get(key, {}).get("images", 0)
        url = (f"https://universe.roboflow.com/{slug}"
               if "/" in slug else "")
        lines.append(f"| {f'[{slug}]({url})' if url else slug} | {version} "
                     f"| {licence} | {n} | {note} |")
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"totals": dict(stats),
              "per_source": {k: dict(v) for k, v in per_src.items()},
              "box_long_side_at_640_per_source": {k: dict(v) for k, v in
                                                  size_table.items()}}
    (out / "stats.json").write_text(json.dumps(report, indent=1),
                                    encoding="utf-8")
    print(json.dumps(report, indent=1))
    if args.zip:
        import zipfile
        zpath = work / "drone_dataset.zip"
        tmp = zpath.with_suffix(".tmp")
        # JPEGs do not compress: store them, deflate only the text files.
        with zipfile.ZipFile(tmp, "w") as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out).as_posix(),
                            compress_type=zipfile.ZIP_STORED
                            if f.suffix.lower() == ".jpg"
                            else zipfile.ZIP_DEFLATED)
        tmp.replace(zpath)
        print(f"packed {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
leak_check.py — is any TEST image the same photo, or a frame of the same scene,
as an image of a TRAIN set? Same rule as round 7 (eval_v7/leak/leakcheck.py):

  md5    byte-identical file
  dup    64-bit dHash (build_dataset.py's) within 4 bits, direct or mirrored,
         AND 64x64 grey thumbnails within 3 grey levels on average
  scene  dHash within 6 bits AND thumbnails within 6 levels (a neighbouring
         frame of the same video, the same photo re-cropped)

    python leak_check.py OUT.json --train NAME=PATH [...] --test NAME=PATH [...]
           [--pattern '*_pub.jpg'] [--cache DIR]

PATH is a folder (searched recursively for images) or a dataset zip
(images/<split>/... ; NAME=ZIP::train picks the split). Features of every
set are cached in --cache, keyed by the set's file list.
"""
import argparse
import fnmatch
import hashlib
import io
import json
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def bits(a):
    d = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in d), 2)


def from_bytes(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    th = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.uint8)
    return hashlib.md5(data).hexdigest(), bits(g), bits(g[:, ::-1]), th.tobytes()


def job(args):
    kind, src, names = args
    out = []
    z = zipfile.ZipFile(src) if kind == "zip" else None
    for n in names:
        try:
            data = z.read(n) if z else Path(n).read_bytes()
            out.append((n, *from_bytes(data)))
        except Exception as exc:
            out.append((n, None, str(exc), None, None))
    return out


def list_set(spec, pattern):
    path, _, split = spec.partition("::")
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            pre = f"images/{split}/" if split else "images/"
            names = sorted(n for n in z.namelist() if n.startswith(pre) and Path(n).suffix.lower() in IMG
                           and fnmatch.fnmatch(Path(n).name, pattern))
        return "zip", str(p), names
    names = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG and fnmatch.fnmatch(f.name, pattern))
    return "files", str(p), names


def features(pool, name, spec, pattern, cache):
    kind, src, names = list_set(spec, pattern)
    key = hashlib.md5(("\n".join(names) + pattern).encode()).hexdigest()[:12]
    cf = cache / f"{name}_{key}.npz"
    if cf.exists():
        c = np.load(cf, allow_pickle=False)
        return {"names": [str(n) for n in c["names"]], "md5": [str(m) for m in c["md5"]],
                "h": c["h"], "hm": c["hm"], "th": c["th"]}
    rows = []
    step = 200
    for part in pool.map(job, [(kind, src, names[i:i + step]) for i in range(0, len(names), step)]):
        rows += part
    bad = [r for r in rows if r[1] is None]
    for r in bad:
        print(f"   UNREADABLE {r[0]}: {r[2]}")
    ok = [r for r in rows if r[1] is not None]
    f = {"names": [r[0] for r in ok], "md5": [r[1] for r in ok],
         "h": np.array([r[2] for r in ok], dtype=np.uint64), "hm": np.array([r[3] for r in ok], dtype=np.uint64),
         "th": np.frombuffer(b"".join(r[4] for r in ok), dtype=np.uint8).reshape(len(ok), 64, 64)}
    np.savez(cf, names=np.array(f["names"]), md5=np.array(f["md5"]), h=f["h"], hm=f["hm"], th=f["th"])
    return f


def match(tr, md5set, h, hm, th):
    if h is None:
        return None
    best = None
    for hh, thx in ((h, th), (hm, th[:, ::-1])):
        d = np.bitwise_count(tr["h"] ^ np.uint64(hh))
        cand = np.flatnonzero(d <= 6)
        if not len(cand):
            continue
        mads = np.abs(tr["th"][cand].astype(np.int16) - thx).mean(axis=(1, 2))
        for i, mad in zip(cand, mads):
            mad = float(mad)
            k = "dup" if d[i] <= 4 and mad <= 3.0 else "scene" if mad <= 6.0 else None
            if k and (best is None or (k == "dup", -mad) > best[0]):
                best = ((k == "dup", -mad), k, tr["names"][i], int(d[i]), round(mad, 2))
    return None if best is None else list(best[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", action="append", default=[])
    ap.add_argument("--train-npz", action="append", default=[],
                    help="NAME=PATH: a train set's features cached by eval_v7/leak/leakcheck.py "
                         "(names, md5, h, th - the same dHash and 64x64 thumbnails as here)")
    ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--cache", default=r"C:\Github\drone_training\v8\audit\leak_cache")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    report = {}
    with ProcessPoolExecutor() as pool:
        trains = {}
        for t in a.train:
            n, _, spec = t.partition("=")
            trains[n] = features(pool, n, spec, "*", cache)
            print(f"train {n}: {len(trains[n]['names'])} images", flush=True)
        for t in a.train_npz:
            n, _, p = t.partition("=")
            c = np.load(p, allow_pickle=False)
            trains[n] = {"names": [str(x) for x in c["names"]], "md5": [str(x) for x in c["md5"]],
                         "h": c["h"].astype(np.uint64), "th": c["th"]}
            print(f"train {n}: {len(trains[n]['names'])} images (round-7 cache)", flush=True)
        for t in a.test:
            n, _, spec = t.partition("=")
            te = features(pool, n, spec, a.pattern, cache)
            res = {}
            for i, name in enumerate(te["names"]):
                th = te["th"][i].astype(np.int16)
                r = {}
                for tn, tr in trains.items():
                    md5s = set(tr["md5"])
                    if te["md5"][i] in md5s:
                        r[tn] = ["md5", tr["names"][tr["md5"].index(te["md5"][i])], 0, 0.0]
                    else:
                        r[tn] = match(tr, md5s, int(te["h"][i]), int(te["hm"][i]), th)
                res[name] = r
            summ = {tn: Counter(v[tn][0] for v in res.values() if v[tn]) for tn in trains}
            clean = sum(1 for v in res.values() if not any(v.values()))
            print(f"test {n}: {len(res)} images, clean vs all train sets {clean}; " +
                  "; ".join(f"{tn}: {dict(c)}" for tn, c in summ.items()), flush=True)
            report[n] = res
    Path(a.out).write_text(json.dumps(report, indent=0))


if __name__ == "__main__":
    main()

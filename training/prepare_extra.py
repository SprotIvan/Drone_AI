#!/usr/bin/env python3
"""
prepare_extra.py — the data added for the third fine-tune (ft_v3):
background images WITHOUT drones, and the owner's own labelled drone photos.

    python prepare_extra.py negatives <work_dir>
    python prepare_extra.py user <work_dir> <cleaned_dir>

Both write a fixed train/val split that build_dataset.py keeps as is:

  <work>/negatives/{train,val}/<source>/*.jpg     no label = no drone
  <work>/user_photos/{train,val}/*.jpg + *.txt    YOLO boxes

WHY
  The live test showed the ft_v2 model firing on facades, windows and
  trees: only 64 of its 17 876 training images had no drone. Backgrounds
  teach "this is nothing"; they lower false positives, NOT the share of
  drones found. Birds and aircraft are included on purpose (hard
  negatives) — at 10 px a bird and a drone look alike, so they may cost a
  little recall on the smallest drones; the val split measures it.

NEGATIVE SOURCES (Kaggle; licence checked through the Kaggle API)
  cloudsky   sadhliroomyprime/cloudsky-segmentation  CC BY 4.0  sky, clouds, sun
  weather    jehanbhathena/weather-dataset           CC0        fog, rain, snow,
             lightning, rainbow, sandstorm (close-up classes dew/frost/glaze/
             rime/hail left out: macro shots, not a view the station has)
  landscape  arnaud58/landscape-pictures             CC0
  planes     maryamlsgumel/aeroplanebirdvsdrone      CC BY-SA 4.0  Aeroplanes only
  birds      adriibezboix/birds                      CC0        small birds in the
             sky, one fixed webcam
  seagulls   kapturovalexander/seagulls-images       CC0        one fixed camera
  Left out: maryamlsgumel/drone-detection-dataset (CC BY-NC-SA, non-commercial),
  harshwalia/birds-vs-drone-dataset (licence unknown).
"""
import glob
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

MAX_SIDE = 1280
MIN_SIDE = 256            # smaller thumbnails are not a view the station has
VAL_SHARE = 0.10

# (name, glob under raw_neg/, how many, sequence?) — a sequence comes from
# one fixed camera: frames are taken evenly spread in time, and val is the
# LAST stretch of it, so no val frame has a near-copy in train.
SOURCES = [
    ("cloudsky", "sadhliroomyprime_cloudsky-segmentation/**/images/*.jpg", 521, True),
    ("weather", "jehanbhathena_weather-dataset/dataset/{fogsmog,lightning,rain,"
                "rainbow,sandstorm,snow}/*", 720, False),
    ("landscape", "arnaud58_landscape-pictures/*", 700, False),
    ("planes", "maryamlsgumel_aeroplanebirdvsdrone/*/Aeroplanes/*", 194, False),
    ("birds", "adriibezboix_birds/birds/*.jpg", 400, True),
    ("seagulls", "kapturovalexander_seagulls-images/**/*.jpg", 150, True),
]
LICENCES = {"cloudsky": "CC BY 4.0", "weather": "CC0", "landscape": "CC0",
            "planes": "CC BY-SA 4.0", "birds": "CC0", "seagulls": "CC0",
            "user_phone": "own data"}
KAGGLE = {"cloudsky": "sadhliroomyprime/cloudsky-segmentation",
          "weather": "jehanbhathena/weather-dataset",
          "landscape": "arnaud58/landscape-pictures",
          "planes": "maryamlsgumel/aeroplanebirdvsdrone",
          "birds": "adriibezboix/birds",
          "seagulls": "kapturovalexander/seagulls-images"}


def expand(pattern):
    if "{" in pattern:
        head, rest = pattern.split("{", 1)
        alts, tail = rest.split("}", 1)
        return [p for a in alts.split(",") for p in expand(head + a + tail)]
    return glob.glob(pattern, recursive=True)


def save_jpg(src, dst):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > MAX_SIDE:
            s = MAX_SIDE / max(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        im.save(dst, quality=95)


def negatives(work: Path):
    raw = work / "raw_neg"
    out = work / "negatives"
    shutil.rmtree(out, ignore_errors=True)
    manifest = {}
    rng = random.Random(0)
    for name, pattern, quota, sequence in SOURCES:
        files, seen = [], set()
        for f in sorted(expand(str(raw / pattern))):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            digest = hashlib.md5(open(f, "rb").read()).hexdigest()
            if digest in seen:                  # byte-identical copy
                continue
            try:
                with Image.open(f) as im:
                    if max(im.size) < MIN_SIDE:
                        continue
            except Exception:
                continue
            seen.add(digest)
            files.append(f)
        if sequence:                            # evenly spread in time
            step = max(1, len(files) / quota)
            files = [files[int(i * step)] for i in range(min(quota, len(files)))]
            n_val = max(1, round(len(files) * VAL_SHARE))
            split = {"train": files[:-n_val], "val": files[-n_val:]}
        else:
            rng.shuffle(files)
            files = files[:quota]
            n_val = max(1, round(len(files) * VAL_SHARE))
            split = {"train": files[n_val:], "val": files[:n_val]}
        for part, fs in split.items():
            d = out / part / name
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(fs):
                save_jpg(f, d / f"{name}_{i:05d}.jpg")
        manifest[name] = {"train": len(split["train"]), "val": len(split["val"]),
                          "kaggle": KAGGLE[name], "licence": LICENCES[name]}
        print(f"{name:10s} train {len(split['train']):4d}  val {len(split['val']):3d}"
              f"  ({len(files)} of the usable files)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("total:", sum(v["train"] + v["val"] for v in manifest.values()))


# The owner's photos (Desktop/drone, Desktop/phone_for_drone), after
# the Paint rectangles were found and painted out (desk_clean.py). Held out
# for val: different scenes spread over the session.
USER_VAL = {"1", "13", "19", "29", "36", "39", "54"}
USER_EXCLUDE = {"35": "the drone is ~15 px and the painted-out rectangle "
                      "left a green smear around it"}


def user(work: Path, cleaned: Path):
    info = json.loads((cleaned / "boxes.json").read_text())
    out = work / "user_photos"
    shutil.rmtree(out, ignore_errors=True)
    for part in ("train", "val"):
        (out / part).mkdir(parents=True, exist_ok=True)
    counts = {"train": [0, 0], "val": [0, 0]}
    neg = work / "negatives" / "train" / "user_phone"
    neg.mkdir(parents=True, exist_ok=True)
    for key, v in sorted(info.items()):
        num = key.split("photo_")[1].split(".")[0]
        if num in USER_EXCLUDE:
            print(f"excluded photo {num}: {USER_EXCLUDE[num]}")
            continue
        im = Image.open(cleaned / key).convert("RGB")
        w, h = im.size
        if not v["labels"]:                   # a background from the phone
            im.save(neg / f"user_phone_{num}.jpg", quality=95)
            continue
        part = "val" if num in USER_VAL else "train"
        stem = f"user_{num}"
        im.save(out / part / f"{stem}.jpg", quality=95)
        with open(out / part / f"{stem}.txt", "w") as f:
            for lab in v["labels"]:
                x0, y0, x1, y1 = lab["box"]
                f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                        f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
        counts[part][0] += 1
        counts[part][1] += len(v["labels"])
    print("user drone photos:", {k: f"{a} images / {b} boxes" for k, (a, b) in counts.items()},
          "| phone backgrounds:", len(list(neg.glob("*.jpg"))))


if __name__ == "__main__":
    if sys.argv[1] == "negatives":
        negatives(Path(sys.argv[2]))
    elif sys.argv[1] == "user":
        user(Path(sys.argv[2]), Path(sys.argv[3]))

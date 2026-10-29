#!/usr/bin/env python3
"""
select_coco.py — real-photo NEGATIVES (no drone) for ft_v8 from COCO val2017:
people (heads, hands, clothes, shoes), a phone in a hand, cars / trucks /
buses / motorcycles / bicycles and their parts, street furniture (traffic
lights, signs, parking meters, hydrants, benches). COCO has no drone class;
images COCO tags with airplane or kite are left out (a drone could be among
them), and every kept image is screened by ft_v5 / ft_v7 and looked at
(review sheets) before use.

    python select_coco.py LABELS_ZIP IMAGES_ZIP OUT_DIR

Split by image into train / val / test (disjoint), per category, written as
OUT_DIR/<split>/images/*.jpg + empty labels/*.txt, with OUT_DIR/coco_split.json.
"""
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

PERSON, BICYCLE, CAR, MOTO, AIRPLANE, BUS, TRUCK = 0, 1, 2, 3, 4, 5, 7
STREET = {9, 10, 11, 12, 13}         # traffic light, hydrant, stop sign, parking meter, bench
PHONE, KITE = 67, 33
VEHICLE = {BICYCLE, CAR, MOTO, BUS, TRUCK}
PLAN = {"phone_in_hand": (150, 20, 60), "vehicle": (450, 50, 150), "person": (400, 50, 140),
        "street": (100, 15, 50)}     # train, val, test per category


def main():
    lz, iz, out = zipfile.ZipFile(sys.argv[1]), zipfile.ZipFile(sys.argv[2]), Path(sys.argv[3])
    cats = {}
    for n in lz.namelist():
        if "labels/val2017/" not in n or not n.endswith(".txt"):
            continue
        cls = {int(l.split()[0]) for l in lz.read(n).decode().splitlines() if l.strip()}
        if AIRPLANE in cls or KITE in cls:
            continue
        stem = Path(n).stem
        if PHONE in cls and PERSON in cls:
            c = "phone_in_hand"
        elif cls & VEHICLE:
            c = "vehicle"
        elif cls & STREET:
            c = "street"
        elif PERSON in cls:
            c = "person"
        else:
            continue
        cats[stem] = c
    rng = random.Random(8)
    split, stats = {}, Counter()
    for c, (ntr, nva, nte) in PLAN.items():
        pool = sorted(s for s, k in cats.items() if k == c)
        rng.shuffle(pool)
        stats[f"{c}: available"] = len(pool)
        for part, k0, k1 in (("test", 0, nte), ("val", nte, nte + nva), ("train", nte + nva, nte + nva + ntr)):
            for s in pool[k0:k1]:
                split[s] = (part, c)
                stats[f"{c}: {part}"] += 1
    for s, (part, c) in split.items():
        (out / part / "images").mkdir(parents=True, exist_ok=True)
        (out / part / "labels").mkdir(parents=True, exist_ok=True)
        (out / part / "images" / f"coco_{c}_{s}.jpg").write_bytes(iz.read(f"val2017/{s}.jpg"))
        (out / part / "labels" / f"coco_{c}_{s}.txt").write_text("")
    (out / "coco_split.json").write_text(json.dumps({"stats": dict(stats), "split": split}, indent=0))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
label_v9.py — eye labels for the station field tests (14.10 + 15.10.2026), object by object.

    python label_v9.py group            candidates of the 4 models (cand_v9.py) -> one object per place per frame
                                        (boxes of different models merged), sorted by session and time
                                        -> v9/label/objects.json
    python label_v9.py sheets           one 200 px tile per object (crop of the native 1320x990 frame, box in
                                        red), 64 per sheet, in time order -> v9/label/sheets_obj/obj_NNN.jpg
    python label_v9.py near REVIEW.json second pass: v5/v7/v8 boxes 0.05-0.15 in frames within 8 s of a labelled
                                        drone (where a drone the models barely see would hide)
                                        -> v9/label/objects_near.json, sheets_near/
    python label_v9.py apply            review files -> v9/label/frame_labels.json

A candidate is model output, not a label. A label is what I see on the tile:
  D = drone not in a hand (flying, hovering, on the ground)   H = drone held / carried by a person
  U = cannot tell (frames with a U object are never used)      everything else on a reviewed sheet = not a drone
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

V9 = Path(r"C:\Github\drone_training\v9")
RAW = V9 / "raw_15_10"
CAND = V9 / "cands"
LAB = V9 / "label"
MODELS = ("ft_v2", "ft_v5", "ft_v7", "ft_v8")
THR = {"ft_v2": 0.35, "ft_v5": 0.15, "ft_v7": 0.15, "ft_v8": 0.15}   # ft_v2 boxes far more noise


def t_of(pub):
    p = pub.split("_")
    return int(p[1][:2]) * 3600 + int(p[1][2:4]) * 60 + int(p[1][4:6]) + int(p[2]) / 1000.0


def overlap(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    sa, sb = a[2] * a[3], b[2] * b[3]
    return inter / max(sa + sb - inter, 1e-9), inter / max(min(sa, sb), 1e-9)


def merge_frame(boxes):
    """boxes: [(score, model, [x,y,w,h])] -> objects; the highest-scoring box of an object is its box."""
    boxes = sorted(boxes, key=lambda x: -x[0])
    objs = []
    for s, m, b in boxes:
        for o in objs:
            iou, ios = overlap(b, o["box"])
            if iou > 0.3 or ios > 0.6:
                o["scores"][m] = max(o["scores"].get(m, 0.0), s)
                break
        else:
            objs.append({"box": [float(v) for v in b], "scores": {m: s}})
    return objs


def load_cands():
    return {m: json.loads((CAND / f"{m}.json").read_text()) for m in MODELS}


def group():
    c = load_cands()
    objects = []
    for key in sorted(c["ft_v8"], key=lambda k: (k.split("/")[0], t_of(k.split("/")[1]))):
        objs = merge_frame([(b[4], m, b[:4]) for m in MODELS for b in c[m][key] if b[4] >= THR[m]])
        objs.sort(key=lambda o: o["box"][0])
        for o in objs:
            o.update({"key": key, "session": key.split("/")[0], "t": t_of(key.split("/")[1])})
            objects.append(o)
    for i, o in enumerate(objects):
        o["id"] = i
    LAB.mkdir(parents=True, exist_ok=True)
    (LAB / "objects.json").write_text(json.dumps(objects))
    only_v2 = sum(1 for o in objects if set(o["scores"]) == {"ft_v2"})
    print(f"{len(objects)} objects in {len(set(o['key'] for o in objects))} frames "
          f"(of {len(c['ft_v8'])}); seen only by ft_v2: {only_v2}")


def tile(o, size=200):
    sess, pub = o["key"].split("/")
    nat = cv2.imread(str(RAW / sess / pub.replace("_pub.jpg", "_nat.jpg")))
    k = nat.shape[1] / 640.0
    x, y, w, h = (v * k for v in o["box"])
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * 1.7 + 45
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
    z = nat[y0:y1, x0:x1].copy()
    th = max(1, int(round(max(z.shape[:2]) / size * 1.5)))
    cv2.rectangle(z, (int(x - x0) - th, int(y - y0) - th), (int(x + w - x0) + th, int(y + h - y0) + th),
                  (0, 0, 255), th)
    return cv2.resize(z, (size, size), interpolation=cv2.INTER_AREA)


def write_sheets(objects, out, prefix):
    out.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((200, 200, 3), np.uint8)
    tiles, index = [], []
    for o in objects:
        z = tile(o)
        sess, pub = o["key"].split("/")
        cv2.putText(z, f"{o['id']}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(z, f"{sess[-6:]} {pub[9:15]}", (88, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
        sc = " ".join(f"{m[-1]}{s:.2f}" for m, s in sorted(o["scores"].items()))
        cv2.putText(z, sc, (2, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
        tiles.append(z)
        index.append(o["id"])
    pages = {}
    for s in range(0, len(tiles), 64):
        tt = tiles[s:s + 64]
        while len(tt) % 8:
            tt.append(blank)
        name = f"{prefix}_{s // 64:03d}.jpg"
        cv2.imwrite(str(out / name), np.vstack([np.hstack(tt[q:q + 8]) for q in range(0, len(tt), 8)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        pages[name] = index[s:s + 64]
    (out / "pages.json").write_text(json.dumps(pages))
    print(f"{len(tiles)} tiles -> {len(pages)} sheets in {out}")


def sheets():
    write_sheets(json.loads((LAB / "objects.json").read_text()), LAB / "sheets_obj", "obj")


def read_reviews(names):
    """-> ({(objects_file, id): 'D'|'H'|'U'}, {(objects_file, id) with a bad box}, reviewed pages per file)."""
    lab, bad_box, pages = {}, set(), defaultdict(set)
    for n in names:
        r = json.loads((LAB / n).read_text(encoding="utf-8"))
        for page, v in r["pages"].items():
            if "skipped" in v:
                continue
            pages[r["objects"]].add(page)
            for k in ("D", "H", "U"):
                for i in v.get(k, []):
                    lab[(r["objects"], i)] = k
            for i in v.get("X", []):
                bad_box.add((r["objects"], i))
    return lab, bad_box, pages


def near(review):
    objects = json.loads((LAB / "objects.json").read_text())
    lab = read_reviews([review])[0]
    drones = defaultdict(list)                   # session -> times of labelled drones
    for o in objects:
        if lab.get(("objects.json", o["id"])) in ("D", "H"):
            drones[o["session"]].append(o["t"])
    c = load_cands()
    by_key = defaultdict(list)
    for o in objects:
        by_key[o["key"]].append(o)
    extra = []
    for key in sorted(c["ft_v8"], key=lambda k: (k.split("/")[0], t_of(k.split("/")[1]))):
        sess, pub = key.split("/")
        t = t_of(pub)
        if not any(abs(t - u) <= 8.0 for u in drones.get(sess, [])):
            continue
        low = [(b[4], m, b[:4]) for m in ("ft_v5", "ft_v7", "ft_v8") for b in c[m][key] if 0.05 <= b[4] < THR[m]]
        for o in merge_frame(low):
            if any(max(overlap(o["box"], p["box"])) > 0.3 for p in by_key[key]):
                continue                          # already an object of the first pass
            o.update({"key": key, "session": sess, "t": t})
            extra.append(o)
    for i, o in enumerate(extra):
        o["id"] = i
    (LAB / "objects_near.json").write_text(json.dumps(extra))
    write_sheets(extra, LAB / "sheets_near", "near")


def apply():
    """Per frame: every reviewed object with its label ('F' = reviewed, not a drone) and whether its box is bad."""
    files = {"objects.json": json.loads((LAB / "objects.json").read_text())}
    if (LAB / "objects_near.json").exists():
        files["objects_near.json"] = json.loads((LAB / "objects_near.json").read_text())
    lab, bad_box, pages = read_reviews([p.name for p in sorted(LAB.glob("review_*.json"))])
    sheet_dirs = {"objects.json": LAB / "sheets_obj", "objects_near.json": LAB / "sheets_near"}
    frames = defaultdict(list)
    cnt = defaultdict(int)
    for fname, objs in files.items():
        page_of = {}
        pj = sheet_dirs[fname] / "pages.json"
        if pj.exists():
            for page, ids in json.loads(pj.read_text()).items():
                for i in ids:
                    page_of[i] = page
        for o in objs:
            if page_of.get(o["id"]) not in pages[fname] and (fname, o["id"]) not in lab:
                cnt["not reviewed"] += 1
                continue
            v = lab.get((fname, o["id"]), "F")
            cnt[v] += 1
            frames[o["key"]].append({"box": o["box"], "lab": v, "bad_box": (fname, o["id"]) in bad_box,
                                     "scores": o["scores"], "src": f"{fname}#{o['id']}"})
    (LAB / "frame_labels.json").write_text(json.dumps(frames, indent=0))
    print(dict(cnt), "in", len(frames), "frames;", sum(1 for k in bad_box), "bad boxes")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "group":
        group()
    elif cmd == "sheets":
        sheets()
    elif cmd == "near":
        near(sys.argv[2])
    elif cmd == "apply":
        apply()

#!/usr/bin/env python3
"""
frames_v9.py — whole-frame review of the field-test frames that are actually used (a model candidate review
only sees what some model boxed; a drone no model boxed is found only by looking at the whole frame).

    python frames_v9.py select                 -> v9/label/frame_sets.json (which frames go where, rules below)
    python frames_v9.py sheets SET             -> v9/label/sheets_frames/SET_NNN.jpg + SET_index.json
    python frames_v9.py zoom OUT KEY [KEY..]   -> native-frame view with a 40 px grid in PUBLISHED px, to read
                                                  the box of a drone no model boxed

Sets (sessions as in CHANGELOG 16.10):
  test_periodic  every readable periodic frame of the test sessions (the primary test metric)
  train_pos      train frames with >= 1 D/H object and no U object
  train_neg      train frames with only not-a-drone objects whose best false box (ft_v5/v7/v8) >= 0.25,
                 at most as many as train_pos (the hardest first)
  val            val frames with >= 1 D/H object and no U, and val hard negatives (best false box >= 0.25)
Tiles: the native frame shrunk to 400x300; labelled D green, H cyan, U yellow, bad box magenta; ticks every
80 published px on the edges.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

V9 = Path(r"C:\Github\drone_training\v9")
RAW = V9 / "raw_15_10"
LAB = V9 / "label"
TEST = {"001018", "001621", "021834", "022253", "024422", "024514"}
VAL = {"000531", "003430"}
TRAIN = {"234718", "002313", "002909", "020505", "234331", "234410", "222303", "223657"}
COL = {"D": (0, 220, 0), "H": (255, 255, 0), "U": (0, 230, 255)}


def rows_of(s):
    out = []
    for line in (RAW / s / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def select():
    sess = json.loads((RAW / "SESSIONS.json").read_text())
    fl = json.loads((LAB / "frame_labels.json").read_text())
    sets = {"test_periodic": [], "train_pos": [], "train_neg": [], "val": []}
    neg_pool, val_neg = [], []
    for s, info in sess.items():
        sid = s[-6:]
        bad = set(info["unreadable"])
        for r in rows_of(s):
            pub = r["files"]["pub"]
            if pub in bad:
                continue
            k = f"{s}/{pub}"
            objs = fl.get(k, [])
            labs = {o["lab"] for o in objs}
            hard = max([max(v for m, v in o["scores"].items() if m != "ft_v2") if any(m != "ft_v2" for m in o["scores"])
                        else 0.0 for o in objs if o["lab"] == "F"], default=0.0)
            if sid in TEST and r["reason"] == "periodic":
                sets["test_periodic"].append(k)
            elif sid in TRAIN:
                if "U" in labs:
                    continue
                if labs & {"D", "H"}:
                    sets["train_pos"].append(k)
                elif hard >= 0.25:
                    neg_pool.append((hard, k))
            elif sid in VAL:
                if "U" in labs:
                    continue
                if labs & {"D", "H"}:
                    sets["val"].append(k)
                elif hard >= 0.25:
                    val_neg.append(k)
    neg_pool.sort(key=lambda x: -x[0])
    sets["train_neg"] = sorted(k for _, k in neg_pool[:len(sets["train_pos"])])
    sets["val"] = sorted(sets["val"] + val_neg)
    (LAB / "frame_sets.json").write_text(json.dumps(sets, indent=0))
    print({k: len(v) for k, v in sets.items()}, "| train hard-negative pool", len(neg_pool))


def subsample(n_pos, n_neg):
    """Owner's choice 17.10 (limits): whole-frame review only for a subset - n_pos train_pos frames spread evenly in
    time over every train session, n_neg train_neg frames = the ones already reviewed + the hardest of the rest.
    The rest of train_pos / train_neg is not used. Writes train_pos_s / train_neg_s into frame_sets.json."""
    sets = json.loads((LAB / "frame_sets.json").read_text())
    rv = json.loads((LAB / "frame_review.json").read_text(encoding="utf-8"))
    pos = sets["train_pos"]
    step = len(pos) / float(n_pos)
    sets["train_pos_s"] = [pos[int(i * step)] for i in range(n_pos)]
    old_idx = {int(k): v for k, v in json.loads((LAB / "sheets_frames" / "train_neg_index.json").read_text()).items()}
    done = [old_idx[i] for s in rv.get("train_neg", {}) for i in range(int(s) * 16, int(s) * 16 + 16) if i in old_idx]
    rest = [k for k in sets["train_neg"] if k not in set(done)]            # train_neg is hardest-first by score
    fl = json.loads((LAB / "frame_labels.json").read_text())

    def hard(k):
        return max([max([v for m, v in o["scores"].items() if m != "ft_v2"] or [0.0]) for o in fl.get(k, [])
                    if o["lab"] == "F"] or [0.0])
    rest.sort(key=lambda k: -hard(k))
    sets["train_neg_s"] = done + rest[:max(0, n_neg - len(done))]
    (LAB / "frame_sets.json").write_text(json.dumps(sets, indent=0))
    print("train_pos_s", len(sets["train_pos_s"]), "| train_neg_s", len(sets["train_neg_s"]), "(already reviewed", len(done), ")")


def tile(k, objs, w=400, h=300):
    s, pub = k.split("/")
    nat = cv2.imread(str(RAW / s / pub.replace("_pub.jpg", "_nat.jpg")))
    im = cv2.resize(nat, (w, h), interpolation=cv2.INTER_AREA)
    f = w / 640.0
    for o in objs:
        if o["lab"] == "F":
            continue
        x, y, bw, bh = (v * f for v in o["box"])
        col = (255, 0, 255) if o.get("bad_box") else COL[o["lab"]]
        cv2.rectangle(im, (int(x) - 2, int(y) - 2), (int(x + bw) + 2, int(y + bh) + 2), col, 1)
    for v in range(80, 640, 80):           # ticks in published px
        p = int(v * f)
        cv2.line(im, (p, 0), (p, 5), (255, 255, 255), 1)
        cv2.line(im, (p, h - 6), (p, h - 1), (255, 255, 255), 1)
    for v in range(80, 480, 80):
        p = int(v * f)
        cv2.line(im, (0, p), (5, p), (255, 255, 255), 1)
        cv2.line(im, (w - 6, p), (w - 1, p), (255, 255, 255), 1)
    return im


def sheets(name):
    sets = json.loads((LAB / "frame_sets.json").read_text())
    fl = json.loads((LAB / "frame_labels.json").read_text())
    keys = sets[name]
    out = LAB / "sheets_frames"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{name}_*.jpg"):
        old.unlink()
    index = {}
    for s0 in range(0, len(keys), 16):
        tiles = []
        for i, k in enumerate(keys[s0:s0 + 16]):
            t = tile(k, fl.get(k, []))
            n = s0 + i
            sess, pub = k.split("/")
            cv2.putText(t, f"{n} {sess[-6:]} {pub[9:15]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(t, f"{n} {sess[-6:]} {pub[9:15]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(t)
            index[n] = k
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[q:q + 4]) for q in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / f"{name}_{s0 // 16:03d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / f"{name}_index.json").write_text(json.dumps(index, indent=0))
    print(name, len(keys), "frames ->", (len(keys) + 15) // 16, "sheets")


def zoom(out, keys):
    rows = []
    for k in keys:
        s, pub = k.split("/")
        nat = cv2.imread(str(RAW / s / pub.replace("_pub.jpg", "_nat.jpg")))
        im = cv2.resize(nat, (1280, 960), interpolation=cv2.INTER_AREA)     # 2 px per published px
        for v in range(0, 640, 40):
            cv2.line(im, (2 * v, 0), (2 * v, 959), (255, 255, 255) if v % 80 == 0 else (128, 128, 128), 1)
            cv2.putText(im, str(v), (2 * v + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for v in range(0, 480, 40):
            cv2.line(im, (0, 2 * v), (1279, 2 * v), (255, 255, 255) if v % 80 == 0 else (128, 128, 128), 1)
            cv2.putText(im, str(v), (2, 2 * v + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(im, k[-40:], (300, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(im)
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def final_labels():
    """frame_labels.json (object review) + frame_review.json (whole-frame review: added boxes, fixed boxes)
    -> v9/label/final_labels.json {key: [[x, y, w, h, lab], ...]} for every frame of frame_sets.json, lab in
    D/H/U (published px). Objects with a bad box (X) are dropped unless a 'fix' replaces them; a train/val
    frame flagged 'missing'/'unsure' or with an unfixed bad box is marked excluded."""
    sets = json.loads((LAB / "frame_sets.json").read_text())
    fl = json.loads((LAB / "frame_labels.json").read_text())
    rv = json.loads((LAB / "frame_review.json").read_text(encoding="utf-8"))
    out, excluded, stats = {}, set(), {}
    # train_pos / train_neg are the full pools; only their reviewed subsets train_pos_s / train_neg_s are used
    sets = {k: v for k, v in sets.items() if k not in ("train_pos", "train_neg")}
    for name, keys in sets.items():
        idx = {}
        p = LAB / "sheets_frames" / f"{name}_index.json"
        if p.exists():
            idx = {int(k): v for k, v in json.loads(p.read_text()).items()}
        reviewed = rv.get(name, {})
        flagged = set()
        for sheet, v in reviewed.items():
            flagged |= set(v.get("missing", [])) | set(v.get("unsure", []))
        add_ix = rv.get("add_by_index", {}) if name == "test_periodic" else {}
        n_rev = {idx[i] for sheet in reviewed for i in range(int(sheet) * 16, int(sheet) * 16 + 16) if i in idx}
        for k in keys:
            boxes, bad_unfixed = [], False
            fixed = rv.get("fix", {}).get(k)
            for o in fl.get(k, []):
                if o["lab"] == "F":
                    continue
                if o["bad_box"]:
                    bad_unfixed |= fixed is None
                    continue
                boxes.append([*o["box"], o["lab"]])
            boxes += [list(b) for b in (fixed or [])]
            boxes += [list(b) for b in rv.get("add", {}).get(k, [])]
            if name == "test_periodic":
                boxes += [list(b) for b in rv.get("add_after_eval", {}).get(k, [])]
            for i, kk in idx.items():
                if kk == k and str(i) in add_ix:
                    boxes += [list(b) for b in add_ix[str(i)]]
            out[k] = boxes
            tile = next((i for i, kk in idx.items() if kk == k), None)
            if name != "test_periodic" and (k not in n_rev or tile in flagged or bad_unfixed
                                            or any(b[4] == "U" for b in boxes)):
                excluded.add(k)
            if name == "test_periodic" and bad_unfixed:
                # an unfixed bad box in the test: the drone is there, its box is not trusted -> U region
                for o in fl.get(k, []):
                    if o["bad_box"] and fixed is None:
                        out[k].append([*o["box"], "U"])
        stats[name] = {"frames": len(keys), "reviewed": len(n_rev), "excluded": len([k for k in keys if k in excluded]),
                       "D": sum(1 for k in keys for b in out[k] if b[4] == "D"),
                       "H": sum(1 for k in keys for b in out[k] if b[4] == "H"),
                       "U": sum(1 for k in keys for b in out[k] if b[4] == "U")}
    (LAB / "final_labels.json").write_text(json.dumps({"labels": out, "excluded": sorted(excluded)}, indent=0))
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    if sys.argv[1] == "final":
        final_labels()
    elif sys.argv[1] == "subsample":
        subsample(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "select":
        select()
    elif sys.argv[1] == "sheets":
        sheets(sys.argv[2])
    elif sys.argv[1] == "zoom":
        zoom(sys.argv[2], sys.argv[3:])

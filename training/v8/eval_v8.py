#!/usr/bin/env python3
"""
eval_v8.py — one evaluation protocol for every model (ft_v5, ft_v7, ft_v8 ...)
on the independent test sets of build_testsets.py (+ test_inet, + the v8 val
for choosing a threshold).

    python eval_v8.py predict MODEL.pt NAME [--sets a,b]   -> v8/eval/preds/NAME.json
    python eval_v8.py score NAME [NAME ...] [--val-thr]    -> v8/eval/report_*.json/.md
    python eval_v8.py new NAME [NAME ...]                  -> station boxes near no labelled
                                                              object, as sheets to label by eye

Input exactly as the station feeds the Hailo model: every image letterboxed
into 640x640 (grey 114, centred, no upscaling past 640 on the long side is
avoided: like HailoInference, the long side is scaled to 640). Station frames:
the published 640x480 frame + the 6 native 640x640 windows
(smart_crop.tile_origins), window boxes mapped with smart_crop.boxes_from_crop
and merged with smart_crop.merge_detections(iou 0.45) — the station's code.

Matching
  image sets   a prediction is a TP when IoU >= 0.5 with an unmatched label
               (greedy, highest score first) -> P, R, mAP50, mAP50-95 (101-point);
               a label is MISSED when no prediction >= thr overlaps it (IoU >= 0.1);
               a FALSE box overlaps no label (IoU < 0.1).
  station      round-7 rule: a box is ON an object when its centre lies inside the
               object's extent + 12 px. On a D object -> hit; on U -> ignored; on F
               or on nothing -> false box. Boxes on nothing are listed by `new`.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
from smart_crop import boxes_from_crop, merge_detections, tile_origins  # noqa: E402

V8 = Path(r"C:\Github\drone_training\v8")
TS = V8 / "testsets"
EV = V8 / "eval"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP = 640
THRS = (0.30, 0.40, 0.55)
IMAGE_SETS = ["test_inet", "drone_close", "drone_photos_eye", "hand_white", "hand_xdrone", "owner7",
              "bg_birds_cam", "bg_birds_dvb", "bg_seagulls", "bg_landscape", "bg_weather",
              "station_trees", "station_bg", "coco_neg"]
LEVEL_ONLY = {"drone_photos"}            # drone photos without boxes: image-level found / not found


# ───────────────────────── input ─────────────────────────
def letterbox(img):
    h, w = img.shape[:2]
    r = min(CROP / h, CROP / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    left, top = int(round((CROP - nw) / 2 - 0.1)), int(round((CROP - nh) / 2 - 0.1))
    can = np.full((CROP, CROP, 3), 114, np.uint8)
    can[top:top + nh, left:left + nw] = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh),
                                                                                  interpolation=cv2.INTER_LINEAR)
    return can, r, left, top


def run(model, imgs, conf):
    """imgs: list of BGR 640x640 arrays -> list of [[x1,y1,x2,y2,s], ...] in canvas px."""
    out = []
    for s in range(0, len(imgs), 32):
        for r in model.predict(imgs[s:s + 32], imgsz=CROP, conf=conf, verbose=False):
            b = r.boxes
            out.append([[*map(float, xy), float(c)] for xy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())])
    return out


def predict(a):
    from ultralytics import YOLO
    model = YOLO(a.model)
    sets = a.sets.split(",") if a.sets else IMAGE_SETS + ["val", "station"]
    pf = EV / "preds" / f"{a.name}.json"
    res = {"model": a.model, "sets": {}}
    if a.sets and pf.exists():                  # add / replace these sets, keep the others
        res = json.loads(pf.read_text())
        assert res["model"] == a.model, f"{pf} holds {res['model']}, not {a.model}"
    for name in sets:
        if name == "station":
            gt = json.loads((TS / "station_gt.json").read_text())
            pub, til, keys = [], [], []
            for fr in gt:
                d = TS / "station" / fr["session"]
                p = cv2.imread(str(d / fr["pub"]))
                n = cv2.imread(str(d / fr["nat"]))
                can, r, l, t = letterbox(p)
                pub.append((can, r, l, t))
                origins = tile_origins((n.shape[1], n.shape[0]), CROP)
                til.append([n[oy:oy + CROP, ox:ox + CROP] for ox, oy in origins])
                keys.append((f"{fr['session']}/{fr['pub']}", origins, (n.shape[1], n.shape[0])))
            pp = run(model, [x[0] for x in pub], a.conf)
            tp = run(model, [w for ws in til for w in ws], a.conf)
            out, k = {}, 0
            for (key, origins, nsize), (can, r, l, t), boxes in zip(keys, pub, pp):
                full = [[(b[0] - l) / r, (b[1] - t) / r, (b[2] - l) / r, (b[3] - t) / r, b[4]] for b in boxes]
                wins = []
                for _ in origins:
                    wins.append(tp[k])
                    k += 1
                out[key] = {"full": full, "wins": wins, "origins": origins, "native": nsize}
            res["sets"]["station"] = out
            print("station", len(out), "frames", flush=True)
            continue
        idir = (V8 / "dataset_v8" / "images" / "val") if name == "val" else (TS / name / "images")
        paths = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG)
        out = {}
        for s in range(0, len(paths), 256):
            chunk = paths[s:s + 256]
            ims = [letterbox(cv2.imread(str(p))) for p in chunk]
            for p, (can, r, l, t), boxes in zip(chunk, ims, run(model, [x[0] for x in ims], a.conf)):
                out[p.name] = [[(b[0] - l) / r, (b[1] - t) / r, (b[2] - l) / r, (b[3] - t) / r, b[4]] for b in boxes]
        res["sets"][name] = out
        print(name, len(out), "images", flush=True)
    (EV / "preds").mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(res))


# ───────────────────────── scoring ─────────────────────────
def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, thr=0.45):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(iou(b, k) <= thr for k in keep):
            keep.append(b)
    return keep


def labels_of(name, img_name, w, h):
    d = (V8 / "dataset_v8" / "labels" / "val") if name == "val" else (TS / name / "labels")
    lp = d / (Path(img_name).stem + ".txt")
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


_size_cache = {}


def img_size(name, img_name):
    k = (name, img_name)
    if k not in _size_cache:
        d = (V8 / "dataset_v8" / "images" / "val") if name == "val" else (TS / name / "images")
        im = cv2.imread(str(d / img_name), cv2.IMREAD_UNCHANGED)
        _size_cache[k] = (im.shape[1], im.shape[0])
    return _size_cache[k]


def ap_101(scores_tp, n_gt):
    if n_gt == 0:
        return float("nan")
    if not scores_tp:
        return 0.0
    s = sorted(scores_tp, key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in s])
    fp = np.cumsum([1 - x[1] for x in s])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    out = 0.0
    for t in np.linspace(0, 1, 101):
        i = np.searchsorted(rec, t, side="left")
        out += prec[i] if i < len(prec) else 0.0
    return out / 101


def size_bucket(g, w, h):
    side = max(g[2] - g[0], g[3] - g[1]) * 640 / max(w, h)
    return "<16px" if side < 16 else "16-32px" if side < 32 else "32-96px" if side < 96 else \
        "96-320px" if side < 320 else ">=320px"


def score_images(name, preds, thrs):
    """-> dict of metrics for one image set."""
    items = []
    for img_name, boxes in preds.items():
        w, h = img_size(name, img_name)
        items.append((img_name, labels_of(name, img_name, w, h), nms(boxes), w, h))
    n_gt = sum(len(g) for _, g, _, _, _ in items)
    out = {"images": len(items), "labels": n_gt,
           "images_with_drone": sum(1 for _, g, _, _, _ in items if g)}
    if name in LEVEL_ONLY:
        out["level_only"] = True
        for t in thrs:
            out[f"@{t:.2f}"] = {"images_found": sum(1 for _, _, p, _, _ in items if any(b[4] >= t for b in p))}
        return out
    if n_gt:
        aps = []
        for it in np.arange(0.5, 0.96, 0.05):
            st = []
            for _, g, p, _, _ in items:
                used = set()
                for b in p:
                    best, bi = 0.0, -1
                    for j, gg in enumerate(g):
                        if j not in used:
                            v = iou(b, gg)
                            if v > best:
                                best, bi = v, j
                    if best >= it:
                        used.add(bi)
                        st.append((b[4], 1))
                    else:
                        st.append((b[4], 0))
            aps.append(ap_101(st, n_gt))
        out["mAP50"] = round(aps[0], 4)
        out["mAP50-95"] = round(float(np.mean(aps)), 4)
    for t in thrs:
        tp = fp_iou = missed = false_boxes = fp_imgs = 0
        by_size = defaultdict(lambda: [0, 0])
        for _, g, p, w, h in items:
            p = [b for b in p if b[4] >= t]
            used = set()
            for b in p:
                best, bi = 0.0, -1
                for j, gg in enumerate(g):
                    if j not in used:
                        v = iou(b, gg)
                        if v > best:
                            best, bi = v, j
                if best >= 0.5:
                    used.add(bi)
                    tp += 1
                else:
                    fp_iou += 1
            nf = sum(1 for b in p if all(iou(b, gg) < 0.1 for gg in g))
            false_boxes += nf
            fp_imgs += nf > 0
            for gg in g:
                hit = any(iou(b, gg) >= 0.1 for b in p)
                missed += not hit
                sb = by_size[size_bucket(gg, w, h)]
                sb[0] += 1
                sb[1] += hit
        r = {"false_boxes": false_boxes, "images_with_false_box": fp_imgs}
        if n_gt:
            r.update({"P": round(tp / max(tp + fp_iou, 1), 4), "R": round(tp / n_gt, 4), "missed": missed,
                      "found_by_size": {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_size.items())}})
        out[f"@{t:.2f}"] = r
    return out


def station_boxes(fr, thr):
    """The station's merge of the full frame + windows at a threshold (published px, xywh)."""
    groups = [([np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]]) for b in fr["full"] if b[4] >= thr],
               [b[4] for b in fr["full"] if b[4] >= thr])]
    nat = fr["native"]
    sx = nat[0] / 640.0
    for origin, wb in zip(fr["origins"], fr["wins"]):
        wb = [b for b in nms(wb) if b[4] >= thr]
        bb, ss = boxes_from_crop([np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]]) for b in wb],
                                 [b[4] for b in wb], tuple(origin), CROP, tuple(nat), (sx, sx))
        groups.append((bb, ss))
    groups[0] = (groups[0][0], groups[0][1])
    b, s = merge_detections(groups, 0.45)
    return [(*map(float, x[:4]), float(sc)) for x, sc in zip(b, s)]


def on(box, obj, margin=12):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x, y, w, h = obj["box"]
    return x - margin <= cx <= x + w + margin and y - margin <= cy <= y + h + margin


def load_station_gt():
    gt = json.loads((TS / "station_gt.json").read_text())
    newlab = TS / "station_new_labels.json"
    extra = json.loads(newlab.read_text()) if newlab.exists() else {}
    for fr in gt:
        k = f"{fr['session']}/{fr['pub']}"
        for o in extra.get(k, []):
            fr["objects"].append(o)
    return gt


def score_station(preds, thrs):
    gt = load_station_gt()
    out = {}
    for t in thrs:
        per = defaultdict(lambda: defaultdict(int))
        for fr in gt:
            k = f"{fr['session']}/{fr['pub']}"
            boxes = station_boxes(preds[k], t)
            s = per[fr["session"]]
            s["frames"] += 1
            d_objs = [o for o in fr["objects"] if o["lab"] == "D"]
            s["drone_objects"] += len(d_objs)
            s["drone_found"] += sum(1 for o in d_objs if any(on(b, o) for b in boxes))
            for o in d_objs:                  # long side in published px
                side = max(o["box"][2], o["box"][3])
                bk = "<16px" if side < 16 else "16-32px" if side < 32 else "32-96px" if side < 96 else ">=96px"
                s[f"n_{bk}"] += 1
                s[f"found_{bk}"] += any(on(b, o) for b in boxes)
            nfalse = 0
            for b in boxes:
                labs = {o["lab"] for o in fr["objects"] if on(b, o)}
                if "D" in labs or "U" in labs:
                    continue
                nfalse += 1
            s["false_boxes"] += nfalse
            s["frames_with_false_box"] += nfalse > 0
            if fr.get("reason") == "periodic":    # saved every 10 s regardless of detections:
                s["periodic_frames"] += 1         # the unbiased sample for a false-alarm rate
                s["periodic_false_boxes"] += nfalse
                s["periodic_frames_with_false_box"] += nfalse > 0
                s["periodic_frames_with_drone"] += bool(d_objs)
                s["periodic_drone_objects"] += len(d_objs)
                s["periodic_drone_found"] += sum(1 for o in d_objs if any(on(b, o) for b in boxes))
        out[f"@{t:.2f}"] = {k: dict(v) for k, v in sorted(per.items())}
    return out


def score(a):
    names = a.names
    preds = {n: json.loads((EV / "preds" / f"{n}.json").read_text())["sets"] for n in names}
    thr_val = {}
    for n in names:
        if "val" in preds[n]:
            best = (0, None)
            for t in np.arange(0.20, 0.71, 0.01):
                m = score_images("val", preds[n]["val"], [round(float(t), 2)])[f"@{t:.2f}"]
                f1 = 2 * m["P"] * m["R"] / max(m["P"] + m["R"], 1e-9)
                if f1 > best[0]:
                    best = (f1, round(float(t), 2))
            thr_val[n] = best[1]
    rep = {"val_threshold_maxF1": thr_val, "models": {}}
    for n in names:
        thrs = sorted(set(THRS) | ({thr_val[n]} if thr_val.get(n) else set()))
        r = {}
        for s, p in preds[n].items():
            r[s] = score_station(p, thrs) if s == "station" else score_images(s, p, thrs)
        rep["models"][n] = r
        print("scored", n, flush=True)
    tag = "_".join(names)
    (EV / f"report_{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(thr_val))


def new(a):
    """Station boxes (>= 0.30, any listed model) near no labelled object -> sheets."""
    gt = load_station_gt()
    preds = {n: json.loads((EV / "preds" / f"{n}.json").read_text())["sets"]["station"] for n in a.names}
    found = []
    for fr in gt:
        k = f"{fr['session']}/{fr['pub']}"
        allb = []
        for n in a.names:
            for b in station_boxes(preds[n][k], 0.30):
                if not any(on(b, o) for o in fr["objects"]):
                    allb.append((n, b))
        groups = []
        for n, b in allb:
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            for g in groups:
                if abs(cx - g["cx"]) < 12 + b[2] / 2 and abs(cy - g["cy"]) < 12 + b[3] / 2:
                    g["boxes"].append((n, b))
                    break
            else:
                groups.append({"cx": cx, "cy": cy, "boxes": [(n, b)]})
        for g in groups:
            xs = [b[0] for _, b in g["boxes"]]
            ys = [b[1] for _, b in g["boxes"]]
            x1 = [b[0] + b[2] for _, b in g["boxes"]]
            y1 = [b[1] + b[3] for _, b in g["boxes"]]
            found.append({"id": len(found), "key": k, "box": [min(xs), min(ys), max(x1) - min(xs), max(y1) - min(ys)],
                          "best": {n: max([b[4] for m, b in g["boxes"] if m == n], default=0) for n in a.names}})
    out = EV / "new_objects"
    out.mkdir(parents=True, exist_ok=True)
    (out / "new_objects.json").write_text(json.dumps(found, indent=0))
    tiles = []
    for o in found:
        sess, pub = o["key"].split("/")
        d = TS / "station" / sess
        nat = cv2.imread(str(d / pub.replace("_pub.jpg", "_nat.jpg")))
        k = nat.shape[1] / 640
        x, y, w, h = o["box"]
        cx, cy = (x + w / 2) * k, (y + h / 2) * k
        half = max(w, h) * k * 1.5 + 40
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
        z = nat[y0:y1, x0:x1].copy()
        cv2.rectangle(z, (int(x * k - x0), int(y * k - y0)), (int((x + w) * k - x0), int((y + h) * k - y0)),
                      (0, 0, 255), 1)
        z = cv2.resize(z, (300, 300), interpolation=cv2.INTER_CUBIC)
        lab = " ".join(f"{n[-2:]}{v:.2f}" for n, v in o["best"].items() if v)
        cv2.putText(z, f"{o['id']} {sess[16:]} {lab}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(z)
    cols, per = 6, 36
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        while len(t) % cols:
            t.append(np.zeros((300, 300, 3), np.uint8))
        cv2.imwrite(str(out / f"new_{s // per:02d}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(len(found), "new objects ->", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("model")
    p.add_argument("name")
    p.add_argument("--sets", default="")
    p.add_argument("--conf", type=float, default=0.01)
    s = sub.add_parser("score")
    s.add_argument("names", nargs="+")
    n = sub.add_parser("new")
    n.add_argument("names", nargs="+")
    a = ap.parse_args()
    {"predict": predict, "score": score, "new": new}[a.cmd](a)


if __name__ == "__main__":
    main()

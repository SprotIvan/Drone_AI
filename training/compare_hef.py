#!/usr/bin/env python3
"""
compare_hef.py — run HEFs on the Pi through the station's OWN inference
class on the same labelled images, and score them the same way.

    ~/mk/venv/bin/python compare_hef.py --station ~/mk/acoustic_radar \\
        --images DIR [--labels DIR] --hef OLD.hef --hef NEW.hef \\
        [--order rgb|bgr|both] [--limit N] --out results.json

--images   *.jpg / *.png; a same-stem *.txt in --labels (default: the images' own
           folder, or ../labels/<split> for an images/<split> layout) holds
           YOLO boxes; no .txt, or an empty one, = a frame with no drone
--order    what the chip is fed. `rgb` = the order the model was trained in.
           `bgr` = the channels swapped, which is what the station feeds by
           default (VisualConfig.swap_detector_channels = False hands the
           detector picamera2's "RGB888" array, whose bytes are B,G,R).

WHY THROUGH THE STATION'S CLASS
  HailoInference (TWO_CAMERAS_FIXED.py) does the letterbox, the UINT8
  input, the decode of the six raw head outputs, the sigmoid detection and
  the NMS. A separate decoder would score a pipeline the station does not
  run. Each HEF is first warmed up at the station's own threshold (0.4) so
  its sigmoid and LTRB auto-detection lock exactly as they would in
  service; only then is the threshold lowered to 0.05 to collect the whole
  score range. NMS keeps a box only if no higher-scoring box overlaps it,
  so the boxes >= 0.4 found at 0.05 are exactly those found at 0.4.

SCORES (IoU >= 0.5 unless stated)
  AP50                        area under precision/recall, all thresholds
  recall / precision / FP     at 0.40 (detector) and 0.55 (confirmation)
  recall by drone size        long side in px after letterboxing to 640
  recall_loose                IoU >= 0.1 — "is the drone marked at all";
                              for a 10-px target a 2-px box shift already
                              costs IoU 0.5
  FP per empty frame          frames with no drone: boxes at 0.40 / 0.55
  infer ms                    the chip call alone (HailoInference.stage_ms)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THRESHOLDS = (0.40, 0.55)
SIZE_BINS = [("<16", 0, 16), ("16-32", 16, 32), ("32-64", 32, 64),
             (">=64", 64, 1e9)]
MAX_WARMUP_FRAMES = 300


def load_labels(img_path, labels_dir, w, h):
    lab = labels_dir / (img_path.stem + ".txt")
    boxes = []
    if lab.exists():
        for line in lab.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              bw * w, bh * h])
    return boxes


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr_iou):
    """Greedy by score, the usual VOC/COCO way. Returns, per prediction,
    whether it is a true positive, and, per ground truth, the score of the
    prediction that found it (None = missed)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    taken = [None] * len(gts)
    tp = [False] * len(preds)
    for i in order:
        box, score = preds[i]
        best, best_j = thr_iou, -1
        for j, g in enumerate(gts):
            if taken[j] is None:
                v = iou(box, g)
                if v >= best:
                    best, best_j = v, j
        if best_j >= 0:
            taken[best_j] = score
            tp[i] = True
    return tp, taken


def ap(scored_tp, n_gt):
    """All-point interpolated average precision."""
    if n_gt == 0 or not scored_tp:
        return None
    scored_tp.sort(key=lambda x: -x[0])
    tps = np.cumsum([t for _, t in scored_tp])
    fps = np.cumsum([not t for _, t in scored_tp])
    rec = tps / n_gt
    prec = tps / np.maximum(tps + fps, 1)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def size_bin(g, w, h):
    px = max(g[2], g[3]) * 640.0 / max(w, h)
    for name, lo, hi in SIZE_BINS:
        if lo <= px < hi:
            return name
    return SIZE_BINS[-1][0]


def run_hef(tc, hef, items, order):
    """Predictions of one HEF on every item: [(boxes, scores)], infer ms."""
    det = tc.HailoInference(hef, conf_thresh=0.4, iou_thresh=0.45)
    try:
        def feed(it):
            img = cv2.imread(it["path"])            # BGR, as cv2 reads it
            x = img if order == "bgr" else img[:, :, ::-1]
            return det.predict_with_scores(np.ascontiguousarray(x))

        # Lock the auto-detection at the station's threshold, as in service.
        for it in items[:MAX_WARMUP_FRAMES]:
            feed(it)
            if det._ltrb_mode is not None:
                break
        ltrb = det._ltrb_mode
        if ltrb is None:
            # Not enough boxes >= 0.4 to vote: freeze the decoder's own
            # default, so boxes at 0.05 cannot cast the vote instead.
            det._ltrb_mode = "stride"
            ltrb = "stride (decoder default, vote did not lock)"
        locked = {"sigmoid_applied_in_decode": det._cls_needs_sigmoid,
                  "ltrb_mode": ltrb}
        det.conf_thresh = 0.05
        out, infer_ms = [], []
        for it in items:
            boxes, scores = feed(it)
            infer_ms.append(det.stage_ms["infer"])
            out.append(([[float(v) for v in b] for b in boxes],
                        [float(s) for s in scores]))
        return out, infer_ms, locked
    finally:
        det.release()


def score(items, preds):
    res = {}
    n_gt = sum(len(it["gts"]) for it in items)
    for thr_iou, key in ((0.5, ""), (0.1, "_loose")):
        scored_tp, per_gt = [], []
        for it, (boxes, scores) in zip(items, preds):
            tp, taken = match(list(zip(boxes, scores)), it["gts"], thr_iou)
            scored_tp += list(zip(scores, tp))
            per_gt += [(size_bin(g, it["w"], it["h"]), s)
                       for g, s in zip(it["gts"], taken)]
        res["AP50" if not key else "AP10"] = ap(list(scored_tp), n_gt)
        for t in THRESHOLDS:
            found = sum(s is not None and s >= t for _, s in per_gt)
            tp_t = sum(1 for s, ok in scored_tp if ok and s >= t)
            fp_t = sum(1 for s, ok in scored_tp if not ok and s >= t)
            res[f"recall{key}@{t:.2f}"] = found / n_gt if n_gt else None
            if not key:
                res[f"precision@{t:.2f}"] = (tp_t / (tp_t + fp_t)
                                             if tp_t + fp_t else None)
                res[f"false_boxes@{t:.2f}"] = fp_t
                bins = {}
                for name, *_ in SIZE_BINS:
                    sel = [s for b, s in per_gt if b == name]
                    if sel:
                        bins[name] = [sum(s is not None and s >= t
                                          for s in sel), len(sel)]
                res[f"recall_by_size@{t:.2f}"] = bins
    empty = [p for it, p in zip(items, preds) if not it["gts"]]
    for t in THRESHOLDS:
        res[f"empty_frames_with_box@{t:.2f}"] = [
            sum(any(s >= t for s in sc) for _, sc in empty), len(empty)]
    return res


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--station", required=True,
                     help="folder with TWO_CAMERAS_FIXED.py")
    ap_.add_argument("--images", required=True)
    ap_.add_argument("--labels")
    ap_.add_argument("--hef", action="append", required=True)
    ap_.add_argument("--order", choices=("rgb", "bgr", "both"), default="rgb")
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()

    sys.path.insert(0, str(Path(args.station).expanduser().resolve()))
    import TWO_CAMERAS_FIXED as tc

    img_dir = Path(args.images).expanduser()
    if args.labels:
        lab_dir = Path(args.labels).expanduser()
    elif img_dir.parent.name == "images":
        lab_dir = img_dir.parent.parent / "labels" / img_dir.name
    else:
        lab_dir = img_dir
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        files = files[:args.limit]
    t0 = time.time()
    items = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        items.append({"name": f.name, "path": str(f), "w": w, "h": h,
                      "gts": load_labels(f, lab_dir, w, h)})
    n_gt = sum(len(it["gts"]) for it in items)
    print(f"{len(items)} images, {n_gt} drones, "
          f"{sum(not it['gts'] for it in items)} empty "
          f"(loaded in {time.time() - t0:.0f} s)", flush=True)

    orders = ("rgb", "bgr") if args.order == "both" else (args.order,)
    report = {"images": len(items), "drones": n_gt, "runs": []}
    for hef in args.hef:
        for order in orders:
            t0 = time.time()
            preds, ms, locked = run_hef(tc, hef, items, order)
            r = {"hef": Path(hef).name, "order": order, **locked,
                 "infer_ms_median": float(np.median(ms)),
                 "infer_ms_p95": float(np.percentile(ms, 95)),
                 "seconds": round(time.time() - t0, 1)}
            r.update(score(items, preds))
            report["runs"].append(r)
            print(json.dumps(r), flush=True)
            out = Path(args.out)
            (out.parent / f"{out.stem}.{Path(hef).stem}.{order}.preds.json"
             ).write_text(json.dumps(
                 {it["name"]: p for it, p in zip(items, preds)}))
            out.write_text(json.dumps(report, indent=1))   # after each run
    print("wrote", args.out)


if __name__ == "__main__":
    main()

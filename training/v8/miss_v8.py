#!/usr/bin/env python3
"""
miss_v8.py — which independent test drones a model misses, and why (same matching as eval_v8.py).

    python miss_v8.py HEF_NAME PT_NAME THR OUT_DIR

For every labelled drone of test_inet_clean, drone_close, drone_photos_eye, owner7 that HEF_NAME misses
at THR (no box >= THR with IoU >= 0.1), records: its long side at 640, the best HEF score on it (any box
>= 0.05 with IoU >= 0.1; the Pi collects >= 0.05) and the best .pt score on it (PT_NAME, >= 0.01).
Kinds:  below_thr   the HEF box is there, score in [0.05, THR)       -> threshold / quantisation
        pt_only     no HEF box >= 0.05, but the .pt sees it >= THR    -> lost in quantisation
        unseen      neither HEF nor .pt has a box >= THR on it        -> the model does not see it
Writes OUT_DIR/misses.json and contact sheets OUT_DIR/miss_<set>_<n>.jpg (crop, label box, scores).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import eval_v8 as ev

SETS = ["test_inet_clean", "drone_close", "drone_photos_eye", "owner7"]


def best_on(boxes, g, lo):
    return max([b[4] for b in boxes if b[4] >= lo and ev.iou(b, g) >= 0.1], default=0.0)


def main():
    hef, pt, thr, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    ph = json.loads((ev.EV / "preds" / f"{hef}.json").read_text())["sets"]
    pp = json.loads((ev.EV / "preds" / f"{pt}.json").read_text())["sets"]
    misses, tot = [], Counter()
    for s in SETS:
        for img, boxes in sorted(ph[s].items()):
            w, h = ev.img_size(s, img)
            bh, bp = ev.nms(boxes), ev.nms(pp[s][img])
            for j, g in enumerate(ev.labels_of(s, img, w, h)):
                tot[s] += 1
                sh, sp = best_on(bh, g, 0.05), best_on(bp, g, 0.01)
                if sh >= thr:
                    continue
                kind = "below_thr" if sh >= 0.05 else "pt_only" if sp >= thr else "unseen"
                side = max(g[2] - g[0], g[3] - g[1]) * 640 / max(w, h)
                misses.append({"set": s, "img": img, "j": j, "box": g, "side640": round(side, 1),
                               "hef": round(sh, 3), "pt": round(sp, 3), "kind": kind,
                               "frac_of_image": round((g[2] - g[0]) * (g[3] - g[1]) / (w * h), 3)})
    (out / "misses.json").write_text(json.dumps(misses, indent=0))
    print(f"{hef} @ {thr:.2f}: missed {len(misses)} of {sum(tot.values())}")
    for s in SETS:
        m = [x for x in misses if x["set"] == s]
        k = Counter(x["kind"] for x in m)
        sz = Counter(ev.size_bucket([0, 0, x["side640"], x["side640"]], 640, 640) for x in m)
        print(f"  {s:17s} missed {len(m):3d}/{tot[s]:3d} | below_thr {k['below_thr']:3d}  pt_only {k['pt_only']:3d}"
              f"  unseen {k['unseen']:3d} | .pt finds it >= thr: {sum(x['pt'] >= thr for x in m):3d} | sizes {dict(sz)}")
    # sheets
    by = defaultdict(list)
    for x in misses:
        by[x["set"]].append(x)
    for s, m in by.items():
        tiles = []
        for i, x in enumerate(m):
            im = cv2.imread(str(ev.TS / s / "images" / x["img"]))
            H, W = im.shape[:2]
            x1, y1, x2, y2 = x["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1) * 1.3 + 0.04 * max(W, H)
            a, b = int(max(0, cx - half)), int(max(0, cy - half))
            c, d = int(min(W, cx + half)), int(min(H, cy + half))
            z = im[b:d, a:c].copy()
            t = max(1, int(round((c - a) / 200)))
            cv2.rectangle(z, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 0, 255), t)
            zh, zw = z.shape[:2]
            r = 220 / max(zh, zw)
            z = cv2.resize(z, (max(1, int(zw * r)), max(1, int(zh * r))), interpolation=cv2.INTER_AREA)
            can = np.zeros((240, 220, 3), np.uint8)
            can[:z.shape[0], :z.shape[1]] = z
            cv2.putText(can, f"#{i} {x['side640']:.0f}px h{x['hef']:.2f} p{x['pt']:.2f} {x['kind'][:5]}", (2, 236),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(can)
        for k in range(0, len(tiles), 36):
            t = tiles[k:k + 36]
            while len(t) % 6:
                t.append(np.zeros((240, 220, 3), np.uint8))
            cv2.imwrite(str(out / f"miss_{s}_{k // 36}.jpg"),
                        np.vstack([np.hstack(t[q:q + 6]) for q in range(0, len(t), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

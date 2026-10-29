#!/usr/bin/env python3
"""
extra_v8.py — extra checks on the eval_v8 preds (same matching rules as eval_v8.py).

    python extra_v8.py paired A@THR B@THR      drone by drone and photo by photo: what only A
                                               finds / only B finds (exact McNemar p) and an
                                               image-level bootstrap 95 % CI of B - A
    python extra_v8.py episodes NAME@THR [...] station recordings: drone appearances grouped into
                                               flights (frames <= 20 s apart): flights with >= 1
                                               found frame, and frames found per flight
    python extra_v8.py quant PT HEF            .pt vs the same model's HEF on the Pi: recall,
                                               false boxes, mAP (the .pt boxes cut at 0.05, as the
                                               Pi collects >= 0.05, so mAP is compared like for like)

Groups (as rules_v81.py): small = test_inet_clean < 32 px + drone_photos_eye < 32 px + owner7;
large = drone_close + drone_photos_eye >= 96 px; all = test_inet_clean + drone_close +
drone_photos_eye + owner7; bg = the 8 drone-free photo sets.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import eval_v8 as ev

DRONE_SETS = ["test_inet_clean", "drone_close", "drone_photos_eye", "owner7"]
BG = ["bg_birds_cam", "bg_birds_dvb", "bg_seagulls", "bg_landscape", "bg_weather", "coco_neg",
      "station_bg", "station_trees"]


def load(name):
    return json.loads((ev.EV / "preds" / f"{name}.json").read_text())["sets"]


def per_label(sname, preds, t):
    """-> {(img, j): (size_bucket, found)}"""
    out = {}
    for img, boxes in preds.items():
        w, h = ev.img_size(sname, img)
        g = ev.labels_of(sname, img, w, h)
        p = [b for b in ev.nms(boxes) if b[4] >= t]
        for j, gg in enumerate(g):
            out[(img, j)] = (ev.size_bucket(gg, w, h), any(ev.iou(b, gg) >= 0.1 for b in p))
    return out


def per_image_fp(sname, preds, t):
    out = {}
    for img, boxes in preds.items():
        w, h = ev.img_size(sname, img)
        g = ev.labels_of(sname, img, w, h)
        p = [b for b in ev.nms(boxes) if b[4] >= t]
        out[img] = any(all(ev.iou(b, gg) < 0.1 for gg in g) for b in p)
    return out


def group_of(sname, size):
    small = sname == "owner7" or (sname in ("test_inet_clean", "drone_photos_eye") and size in ("<16px", "16-32px"))
    large = sname == "drone_close" or (sname == "drone_photos_eye" and size in ("96-320px", ">=320px"))
    return ["all"] + (["small"] if small else []) + (["large"] if large else [])


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(rows, B=5000, seed=1):
    """rows: per image (n, a, b) -> 95 % CI of (sum b - sum a) / sum n."""
    x = np.array(rows, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    n, a, b = (x[:, k][idx].sum(axis=1) for k in range(3))
    d = (b - a) / np.maximum(n, 1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def paired(specA, specB):
    (A, ta), (B_, tb) = [(s.split("@")[0], float(s.split("@")[1])) for s in (specA, specB)]
    pa, pb = load(A), load(B_)
    print(f"A = {A} @ {ta:.2f}    B = {B_} @ {tb:.2f}")
    res = {}
    for grp in ("all", "small", "large"):
        only_a = only_b = both = none = 0
        per_img = defaultdict(lambda: [0, 0, 0])
        for s in DRONE_SETS:
            la, lb = per_label(s, pa[s], ta), per_label(s, pb[s], tb)
            for k, (size, fa) in la.items():
                if grp not in group_of(s, size):
                    continue
                fb = lb[k][1]
                only_a += fa and not fb
                only_b += fb and not fa
                both += fa and fb
                none += not fa and not fb
                r = per_img[(s, k[0])]
                r[0] += 1
                r[1] += fa
                r[2] += fb
        n = only_a + only_b + both + none
        lo, hi, p_le0 = boot(list(per_img.values()))
        res[grp] = {"labels": n, "A_found": only_a + both, "B_found": only_b + both, "only_A": only_a,
                    "only_B": only_b, "mcnemar_p": mcnemar(only_a, only_b), "ci95_B_minus_A": [lo, hi]}
        print(f"drones {grp:5s}: {n:4d} | A {only_a + both:4d} ({100 * (only_a + both) / n:5.1f} %)  "
              f"B {only_b + both:4d} ({100 * (only_b + both) / n:5.1f} %) | only A {only_a:3d}  only B {only_b:3d} "
              f"| McNemar p = {mcnemar(only_a, only_b):.2g} | B-A 95 % CI [{100 * lo:+.1f}, {100 * hi:+.1f}] pp")
    only_a = only_b = both = 0
    rows = []
    for s in BG:
        fa, fb = per_image_fp(s, pa[s], ta), per_image_fp(s, pb[s], tb)
        for k in fa:
            only_a += fa[k] and not fb[k]
            only_b += fb[k] and not fa[k]
            both += fa[k] and fb[k]
            rows.append((1, fa[k], fb[k]))
    n = len(rows)
    lo, hi, _ = boot(rows)
    res["bg"] = {"photos": n, "A_fp": only_a + both, "B_fp": only_b + both, "only_A": only_a, "only_B": only_b,
                 "mcnemar_p": mcnemar(only_a, only_b), "ci95_B_minus_A": [lo, hi]}
    print(f"drone-free photos with a false box: {n} | A {only_a + both} ({100 * (only_a + both) / n:.1f} %)  "
          f"B {only_b + both} ({100 * (only_b + both) / n:.1f} %) | only A {only_a}  only B {only_b} "
          f"| McNemar p = {mcnemar(only_a, only_b):.2g} | B-A 95 % CI [{100 * lo:+.1f}, {100 * hi:+.1f}] pp")
    # station drones (biased: frames saved when the live ft_v5 / ft_v2 fired)
    gt = ev.load_station_gt()
    sa, sb = pa["station"], pb["station"]
    st = defaultdict(lambda: [0, 0, 0])
    for fr in gt:
        k = f"{fr['session']}/{fr['pub']}"
        ba, bb = ev.station_boxes(sa[k], ta), ev.station_boxes(sb[k], tb)
        for o in fr["objects"]:
            if o["lab"] != "D":
                continue
            fa, fb = any(ev.on(b, o) for b in ba), any(ev.on(b, o) for b in bb)
            r = st[fr["session"]]
            r[0] += fa and not fb
            r[1] += fb and not fa
            r[2] += fa and fb
    print("station drones (BIASED sample) by recording: only A / only B / both")
    for s, (x, y, z) in sorted(st.items()):
        print(f"   {s}: {x} / {y} / {z}")
    res["station"] = {s: {"only_A": x, "only_B": y, "both": z} for s, (x, y, z) in st.items()}
    out = ev.EV / f"paired_{A}@{ta:.2f}_vs_{B_}@{tb:.2f}.json"
    out.write_text(json.dumps(res, indent=1))


def frame_time(pub):
    p = pub.split("_")
    t = p[1]
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + int(p[2]) / 1000


def episodes(specs):
    gt = ev.load_station_gt()
    models = [(s.split("@")[0], float(s.split("@")[1])) for s in specs]
    preds = {m: load(m)["station"] for m, _ in models}
    by_sess = defaultdict(list)
    for fr in gt:
        d = [o for o in fr["objects"] if o["lab"] == "D"]
        if d:
            by_sess[fr["session"]].append((frame_time(fr["pub"]), fr, d))
    rows = []
    for s, frs in sorted(by_sess.items()):
        frs.sort(key=lambda x: x[0])
        ep = [[frs[0]]]
        for x in frs[1:]:
            if x[0] - ep[-1][-1][0] > 20:
                ep.append([x])
            else:
                ep[-1].append(x)
        for e in ep:
            r = {"session": s, "start": e[0][1]["pub"][9:17], "seconds": round(e[-1][0] - e[0][0], 1),
                 "frames": len(e)}
            rs = defaultdict(int)
            for _, fr, _ in e:
                rs[fr.get("reason", "?")] += 1
            r["reasons"] = dict(rs)
            for m, t in models:
                n = 0
                for _, fr, d in e:
                    k = f"{fr['session']}/{fr['pub']}"
                    bx = ev.station_boxes(preds[m][k], t)
                    n += any(ev.on(b, o) for o in d for b in bx)
                r[f"{m}@{t:.2f}"] = n
            rows.append(r)
    names = [f"{m}@{t:.2f}" for m, t in models]
    print(f"{'recording':18s} {'start':9s} {'sec':>6s} {'frames':>6s}  " + "  ".join(f"{n:>14s}" for n in names) + "  reasons")
    tot = defaultdict(int)
    for r in rows:
        print(f"{r['session'][7:]:18s} {r['start']:9s} {r['seconds']:6.0f} {r['frames']:6d}  " +
              "  ".join(f"{r[n]:>14d}" for n in names) + f"  {r['reasons']}")
        for n in names:
            tot[n] += r[n] > 0
    print(f"flights: {len(rows)}; flights with >= 1 found frame: " + ", ".join(f"{n} {tot[n]}" for n in names))
    (ev.EV / f"episodes_{'_'.join(names)}.json").write_text(json.dumps(rows, indent=1))


def quant(pt, hef):
    pp, ph = load(pt), load(hef)
    cut = {s: {k: [b for b in v if b[4] >= 0.05] for k, v in pp[s].items()} for s in pp if s != "station"}
    for s in ["test_inet_clean", "drone_close", "drone_photos_eye", "val"]:
        a = ev.score_images(s, cut[s], ev.THRS)
        b = ev.score_images(s, ph[s], ev.THRS)
        cells = [f"mAP50 {100 * a['mAP50']:.1f} -> {100 * b['mAP50']:.1f}"]
        for t in ("@0.30", "@0.40", "@0.55"):
            n = a["labels"]
            cells.append(f"{t[1:]}: found {100 * (n - a[t]['missed']) / n:.1f} -> {100 * (n - b[t]['missed']) / n:.1f} %")
        print(f"{s:18s} " + " | ".join(cells))
    for t in ("@0.30", "@0.40", "@0.55"):
        fa = sum(ev.score_images(s, cut[s], [float(t[1:])])[t]["images_with_false_box"] for s in BG)
        fb = sum(ev.score_images(s, ph[s], [float(t[1:])])[t]["images_with_false_box"] for s in BG)
        print(f"drone-free photos with a false box {t[1:]}: .pt {fa} -> HEF {fb} (of 1686)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "paired":
        paired(sys.argv[2], sys.argv[3])
    elif cmd == "episodes":
        episodes(sys.argv[2:])
    elif cmd == "quant":
        quant(sys.argv[2], sys.argv[3])

#!/usr/bin/env python3
"""
robust_v8.py — the same independent test images under synthetic bad conditions,
to see which model holds up (run on the Pi through the station's HailoInference,
scored here by eval_v8.score_images - the same matching as every other set).

    python robust_v8.py make                 -> v8/testsets_robust/<cond>_<set>/images [+ labels]
    python robust_v8.py score NAME [NAME..]  -> v8/eval/robust_<names>.json (+ printed table)

Every image is first brought to <= 640 px on the long side (INTER_AREA, as a
camera bins its sensor; the model sees 640 anyway), then:

  clean   nothing (the control every condition is compared with)
  dark    brightness x 0.35 (dusk / under-exposure, ~1.5 stops)
  fog     contrast 45 %, blended with light grey 200 (haze)
  blur    horizontal motion blur, 9 px
  noise   gaussian noise sigma 15 grey levels (high ISO), fixed seed per image
  jpeg20  JPEG quality 20 (strong stream compression)
  half    content shrunk x0.5, centred on a canvas of the same size filled with
          the image's mean colour: every drone becomes 2x smaller ("2x farther");
          drone sets only

Saved as JPEG q95 (jpeg20: q20). Labels (YOLO, normalised) are copied, or moved
for "half". Sets: test_inet_clean (inet), drone_close (close), and the drone-free
bg = bg_birds_dvb + coco_neg + bg_landscape (900 photos, prefixed dvb_/coco_/land_).
"""
import json
import shutil
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

V8 = Path(r"C:\Github\drone_training\v8")
TS = V8 / "testsets"
RB = V8 / "testsets_robust"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SRC = {"inet": [("test_inet_clean", "")], "close": [("drone_close", "")],
       "bg": [("bg_birds_dvb", "dvb_"), ("coco_neg", "coco_"), ("bg_landscape", "land_")]}
CONDS = ["clean", "dark", "fog", "blur", "noise", "jpeg20", "half"]


def to640(im):
    h, w = im.shape[:2]
    r = 640 / max(h, w)
    if r >= 1:
        return im
    return cv2.resize(im, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)


def distort(cond, im, name):
    f = im.astype(np.float32)
    if cond == "clean":
        return im, None
    if cond == "dark":
        return np.clip(np.round(f * 0.35), 0, 255).astype(np.uint8), None
    if cond == "fog":
        return np.clip(np.round(f * 0.45 + 0.55 * 200), 0, 255).astype(np.uint8), None
    if cond == "blur":
        k = np.full((1, 9), 1 / 9, np.float32)
        return cv2.filter2D(im, -1, k, borderType=cv2.BORDER_REFLECT), None
    if cond == "noise":
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        return np.clip(np.round(f + rng.normal(0, 15, im.shape)), 0, 255).astype(np.uint8), None
    if cond == "jpeg20":
        return im, None
    if cond == "half":
        h, w = im.shape[:2]
        nw, nh = max(1, int(round(w * 0.5))), max(1, int(round(h * 0.5)))
        can = np.empty_like(im)
        can[:] = np.round(im.reshape(-1, im.shape[2]).mean(axis=0)).astype(np.uint8)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        can[y0:y0 + nh, x0:x0 + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        return can, (x0 / w, y0 / h, nw / w, nh / h)
    raise ValueError(cond)


def make():
    if RB.exists():
        sys.exit(f"{RB} exists")
    stats = {}
    for cond in CONDS:
        for grp, srcs in SRC.items():
            if cond == "half" and grp == "bg":
                continue
            d = RB / f"{cond}_{grp}"
            (d / "images").mkdir(parents=True)
            if grp != "bg":
                (d / "labels").mkdir()
            n = nlab = 0
            for sname, pre in srcs:
                for p in sorted((TS / sname / "images").iterdir()):
                    if p.suffix.lower() not in IMG:
                        continue
                    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    out, tr = distort(cond, to640(im), p.name)
                    stem = pre + p.stem
                    q = 20 if cond == "jpeg20" else 95
                    cv2.imwrite(str(d / "images" / (stem + ".jpg")), out, [cv2.IMWRITE_JPEG_QUALITY, q])
                    n += 1
                    lp = TS / sname / "labels" / (p.stem + ".txt")
                    if grp != "bg" and lp.exists():
                        lines = []
                        for line in lp.read_text().splitlines():
                            v = line.split()
                            if len(v) != 5:
                                continue
                            c, cx, cy, bw, bh = v[0], *map(float, v[1:])
                            if tr:
                                ox, oy, sx, sy = tr
                                cx, cy, bw, bh = ox + cx * sx, oy + cy * sy, bw * sx, bh * sy
                            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                            nlab += 1
                        (d / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[d.name] = {"images": n, "labels": nlab}
            print(d.name, n, "images", nlab, "labels", flush=True)
    (RB / "MANIFEST.json").write_text(json.dumps({"doc": __doc__, "sets": stats}, indent=1))


def score(names):
    import eval_v8
    eval_v8.TS = RB                      # labels_of / img_size read <TS>/<set>/{images,labels}
    rep = {}
    sets = sorted(json.loads((RB / "MANIFEST.json").read_text())["sets"])
    for n in names:
        pr = json.loads((V8 / "eval" / "preds" / f"{n}.json").read_text())["sets"]
        rep[n] = {}
        for s in sets:
            key = "rb_" + s
            if key not in pr:
                print(f"{n}: {key} missing")
                continue
            rep[n][s] = eval_v8.score_images(s, pr[key], eval_v8.THRS)
        print("scored", n, flush=True)
    out = V8 / "eval" / f"robust_{'_'.join(names)}.json"
    out.write_text(json.dumps(rep, indent=1))

    def found(n, s, t):
        x = rep[n].get(s)
        return (x["labels"] - x[t]["missed"], x["labels"]) if x else None

    def fp(n, s, t):
        x = rep[n].get(s)
        return (x[t]["images_with_false_box"], x["images"]) if x else None

    md = []
    worse = ["dark", "fog", "blur", "noise", "jpeg20"]
    for t in ("@0.30", "@0.40", "@0.55"):
        md += [f"### Поріг {t[1:]}: знайдено дронів — інтернет-кадри (555) / зблизька (309); фото без дрона з хибною "
               f"рамкою (900)", "", "| умова | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for c in CONDS:
            cells = []
            for n in names:
                a, b, f = found(n, f"{c}_inet", t), found(n, f"{c}_close", t), fp(n, f"{c}_bg", t)
                cells.append(f"{100 * a[0] / a[1]:.1f} / {100 * b[0] / b[1]:.1f} %" +
                             (f"; {100 * f[0] / f[1]:.1f} %" if f else ""))
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        cells = []
        for n in names:
            a = [found(n, f"{c}_inet", t) for c in worse]
            b = [found(n, f"{c}_close", t) for c in worse]
            f = [fp(n, f"{c}_bg", t) for c in worse]
            cells.append(f"**{100 * sum(x[0] for x in a) / sum(x[1] for x in a):.1f} / "
                         f"{100 * sum(x[0] for x in b) / sum(x[1] for x in b):.1f} %; "
                         f"{100 * sum(x[0] for x in f) / sum(x[1] for x in f):.1f} %**")
        md.append("| **середнє dark–jpeg20** | " + " | ".join(cells) + " |")
        md.append("")
    (V8 / "eval" / f"robust_{'_'.join(names)}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for t in ("@0.30", "@0.40", "@0.55"):
        print(f"\nthreshold {t[1:]}: drones found % (inet | close) / bg photos with a false box %")
        print("cond     " + "".join(f"{n:>30s}" for n in names))
        for c in CONDS:
            row = []
            for n in names:
                r = rep[n]
                cell = []
                for g in ("inet", "close"):
                    x = r.get(f"{c}_{g}")
                    cell.append(f"{100 * (x['labels'] - x[t]['missed']) / x['labels']:5.1f}" if x else "  -  ")
                b = r.get(f"{c}_bg")
                cell.append(f"{100 * b[t]['images_with_false_box'] / b['images']:4.1f}" if b else "  - ")
                row.append(f"{cell[0]} | {cell[1]} / {cell[2]}")
            print(f"{c:8s} " + "".join(f"{x:>30s}" for x in row))
    print("->", out)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        score(sys.argv[2:])

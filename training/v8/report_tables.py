#!/usr/bin/env python3
"""
report_tables.py — markdown tables from eval_v8.py score output.

    python report_tables.py REPORT.json [OUT.md]

Groups: drone sets (recall, P, mAP), background sets (false boxes), station
recordings (drone objects found, false boxes, frames with a false box).
Every number is copied from REPORT.json; nothing is recomputed here except
sums over sets.
"""
import json
import sys
from pathlib import Path

DRONE = [("test_inet_clean", "інтернет-кадри, яких не бачила жодна модель (632)"),
         ("test_inet", "інтернет-кадри, усі 826 (194 бачила ft_v2)"),
         ("drone_close", "великі дрони зблизька"),
         ("drone_photos_eye", "фото дронів (Kaggle, рамки оком)"),
         ("hand_white", "білий дрон у руці (станція)"),
         ("hand_xdrone", "X-дрон у руці (станція)"),
         ("owner7", "7 фото власника (малі дрони)")]
BG = [("bg_birds_cam", "птахи (камера)"), ("bg_birds_dvb", "птахи (drone-vs-bird)"),
      ("bg_seagulls", "чайки"), ("bg_landscape", "пейзажі"), ("bg_weather", "погода"),
      ("coco_neg", "люди / авто / телефони / вулиця (COCO)"),
      ("station_bg", "фони станції"), ("station_trees", "дерева станції (запис 180118)")]
FLIGHT = ["frames_20260919_042047", "frames_20260920_025044", "frames_20260920_101214"]


def main():
    rep = json.loads(Path(sys.argv[1]).read_text())
    models = list(rep["models"])
    thr_val = rep.get("val_threshold_maxF1", {})
    thrs = ["@0.30", "@0.40", "@0.55"]
    L = []
    L.append(f"Поріг з val (max F1): " + ", ".join(f"{m} {thr_val.get(m)}" for m in models))
    L.append("")
    for t in thrs + ["val"]:
        L.append(f"### Дрони — {'поріг з val' if t == 'val' else 'поріг ' + t[1:]}")
        L.append("")
        L.append("| набір | " + " | ".join(f"{m} R / P / знайдено" for m in models) + " |")
        L.append("|---|" + "---|" * len(models))
        for s, name in DRONE:
            row = []
            for m in models:
                r = rep["models"][m].get(s)
                key = f"@{thr_val[m]:.2f}" if t == "val" else t
                if not r or key not in r:
                    row.append("—")
                    continue
                x = r[key]
                n = r["labels"]
                row.append(f"{x['R']:.3f} / {x['P']:.3f} / {n - x['missed']}/{n}")
            L.append(f"| {name} | " + " | ".join(row) + " |")
        L.append("")
    L.append("### mAP (від порогу не залежить)")
    L.append("")
    L.append("| набір | " + " | ".join(f"{m} mAP50 / mAP50-95" for m in models) + " |")
    L.append("|---|" + "---|" * len(models))
    for s, name in DRONE + [("val", "v8 val (вибір моделі)")]:
        row = []
        for m in models:
            r = rep["models"][m].get(s)
            row.append(f"{r['mAP50']:.3f} / {r['mAP50-95']:.3f}" if r and "mAP50" in r else "—")
        L.append(f"| {name} | " + " | ".join(row) + " |")
    L.append("")
    for t in thrs + ["val"]:
        L.append(f"### Фото без дрона: фото з хибною рамкою — {'поріг з val' if t == 'val' else 'поріг ' + t[1:]}")
        L.append("")
        L.append("| набір | фото | " + " | ".join(models) + " |")
        L.append("|---|---|" + "---|" * len(models))
        tot = {m: 0 for m in models}
        nimg = 0
        for s, name in BG:
            r0 = rep["models"][models[0]].get(s)
            if not r0:
                continue
            nimg += r0["images"]
            row = []
            for m in models:
                key = f"@{thr_val[m]:.2f}" if t == "val" else t
                x = rep["models"][m][s][key]
                tot[m] += x["images_with_false_box"]
                row.append(f"{x['images_with_false_box']} ({x['false_boxes']} рамок)")
            L.append(f"| {name} | {r0['images']} | " + " | ".join(row) + " |")
        L.append(f"| **разом** | {nimg} | " + " | ".join(f"**{tot[m]}**" for m in models) + " |")
        L.append("")
    for t in thrs + ["val"]:
        L.append(f"### Записи станції (кадр + 6 вікон, злиття як на станції) — {'поріг з val' if t == 'val' else 'поріг ' + t[1:]}")
        L.append("")
        L.append("| запис | кадрів | дронів | " + " | ".join(f"{m} знайдено / хибних рамок / кадрів з хибною" for m in models) + " |")
        L.append("|---|---|---|" + "---|" * len(models))
        st0 = rep["models"][models[0]]["station"]
        key0 = f"@{thr_val[models[0]]:.2f}" if t == "val" else t
        sess = sorted(st0[key0])
        tot = {m: [0, 0, 0] for m in models}
        nfr = nd = 0
        for s in sess:
            b0 = st0[key0][s]
            nfr += b0["frames"]
            nd += b0["drone_objects"]
            row = []
            for m in models:
                key = f"@{thr_val[m]:.2f}" if t == "val" else t
                b = rep["models"][m]["station"][key][s]
                for k, v in enumerate((b["drone_found"], b["false_boxes"], b["frames_with_false_box"])):
                    tot[m][k] += v
                row.append(f"{b['drone_found']}/{b['drone_objects']} / {b['false_boxes']} / {b['frames_with_false_box']}")
            L.append(f"| {s[7:]} | {b0['frames']} | {b0['drone_objects']} | " + " | ".join(row) + " |")
        L.append(f"| **разом** | {nfr} | {nd} | " + " | ".join(
            f"**{tot[m][0]}/{nd} / {tot[m][1]} / {tot[m][2]}**" for m in models) + " |")
        L.append("")
    # station drones by size, and the periodic (unbiased) false-alarm sample
    L.append("### Записи станції: знайдені дрони за розміром (довша сторона в px опублікованого кадру 640×480)")
    L.append("")
    L.append("| поріг | " + " | ".join(f"{m} <16 / 16–32 / 32–96 / ≥96" for m in models) + " |")
    L.append("|---|" + "---|" * len(models))
    for t in thrs + ["val"]:
        row = []
        for m in models:
            key = f"@{thr_val[m]:.2f}" if t == "val" else t
            tot = {}
            for b in rep["models"][m]["station"][key].values():
                for k, v in b.items():
                    if k.startswith(("n_", "found_")):
                        tot[k] = tot.get(k, 0) + v
            row.append(" / ".join(f"{tot.get('found_' + bk, 0)}/{tot.get('n_' + bk, 0)}"
                                  for bk in ("<16px", "16-32px", "32-96px", ">=96px")))
        L.append(f"| {'з val' if t == 'val' else t[1:]} | " + " | ".join(row) + " |")
    L.append("")
    L.append("### Записи станції: хибні рамки на кадрах «periodic» (зберігались кожні 10 с незалежно від детекцій)")
    L.append("")
    L.append("| поріг | кадрів periodic | " + " | ".join(f"{m} хибних рамок / кадрів з хибною" for m in models) + " |")
    L.append("|---|---|" + "---|" * len(models))
    for t in thrs + ["val"]:
        row, nper = [], 0
        for m in models:
            key = f"@{thr_val[m]:.2f}" if t == "val" else t
            fb = ff = nper = 0
            for b in rep["models"][m]["station"][key].values():
                fb += b.get("periodic_false_boxes", 0)
                ff += b.get("periodic_frames_with_false_box", 0)
                nper += b.get("periodic_frames", 0)
            row.append(f"{fb} / {ff}")
        L.append(f"| {'з val' if t == 'val' else t[1:]} | {nper} | " + " | ".join(row) + " |")
    L.append("")
    text = "\n".join(L) + "\n"
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

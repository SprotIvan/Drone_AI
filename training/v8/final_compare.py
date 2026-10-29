#!/usr/bin/env python3
"""
final_compare.py — percentages for the final ft_v5 / ft_v7 / ft_v8 comparison,
straight from an eval_v8.py score report (no metric is recomputed here).

    python final_compare.py REPORT.json OUT.md --models hef_v5,hef_v7,hef_v8 [--names ft_v5,ft_v7,ft_v8]

"found %"      drones with a box of the model on them (IoU >= 0.1), of all labelled drones
"precision %"  boxes that hit a drone (IoU >= 0.5), of all boxes (sets with drones)
"false-alarm %" drone-free photos with at least one false box, of all drone-free photos
station        899 frames of 17 recordings no model trained on (frame + 6 crop windows, merged as
               the station merges); drones only on frames the live model saved -> biased to ft_v5
"""
import argparse
import json
from pathlib import Path

DRONE = [("test_inet_clean", "інтернет-кадри, яких не бачила жодна модель"),
         ("drone_close", "великі дрони зблизька"),
         ("drone_photos_eye", "фото дронів (рамки оком)"),
         ("owner7", "7 фото власника (малі дрони)")]
HAND = ["hand_white", "hand_xdrone"]
BG = [("bg_birds_cam", "птахи (камера)"), ("bg_birds_dvb", "птахи (drone-vs-bird)"), ("bg_seagulls", "чайки"),
      ("bg_landscape", "пейзажі"), ("bg_weather", "погода"), ("coco_neg", "люди / авто / телефони (COCO)"),
      ("station_bg", "фони станції"), ("station_trees", "дерева станції")]


def pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "—"


def found(r, key):
    return r["labels"] - r[key]["missed"], r["labels"]


def size_found(r, key, keys):
    fs = r[key]["found_by_size"]
    return (sum(int(fs[k].split("/")[0]) for k in keys if k in fs),
            sum(int(fs[k].split("/")[1]) for k in keys if k in fs))


def col(rep, m, key):
    r = rep["models"][m]
    out = {}
    for s, _ in DRONE:
        f, n = found(r[s], key)
        out[f"found|{s}"] = (f, n)
        out[f"prec|{s}"] = r[s][key].get("P")
    out["small"] = size_found(r["test_inet_clean"], key, ("<16px", "16-32px"))
    pe = size_found(r["drone_photos_eye"], key, ("<16px", "16-32px"))
    out["small"] = (out["small"][0] + pe[0], out["small"][1] + pe[1])
    out["large"] = (found(r["drone_close"], key)[0] + size_found(r["drone_photos_eye"], key, ("96-320px", ">=320px"))[0],
                    found(r["drone_close"], key)[1] + size_found(r["drone_photos_eye"], key, ("96-320px", ">=320px"))[1])
    h = [found(r[s], key) for s in HAND]
    out["hand"] = (sum(x[0] for x in h), sum(x[1] for x in h))
    out["bg"] = (sum(r[s][key]["images_with_false_box"] for s, _ in BG), sum(r[s]["images"] for s, _ in BG))
    for s, _ in BG:
        out[f"bg|{s}"] = (r[s][key]["images_with_false_box"], r[s]["images"])
    st = r["station"][key]
    out["st_found"] = (sum(v["drone_found"] for v in st.values()), sum(v["drone_objects"] for v in st.values()))
    out["st_fp"] = (sum(v["false_boxes"] for v in st.values()), sum(v["frames"] for v in st.values()))
    out["st_fpf"] = (sum(v["frames_with_false_box"] for v in st.values()), sum(v["frames"] for v in st.values()))
    out["per_fpf"] = (sum(v.get("periodic_frames_with_false_box", 0) for v in st.values()),
                      sum(v.get("periodic_frames", 0) for v in st.values()))
    out["per_found"] = (sum(v.get("periodic_drone_found", 0) for v in st.values()),
                        sum(v.get("periodic_drone_objects", 0) for v in st.values()))
    return out


def rows(cols):
    R = []
    for s, name in DRONE:
        R.append((f"знайдено дронів — {name}", [f"**{pct(*c[f'found|{s}'])}** ({c[f'found|{s}'][0]}/{c[f'found|{s}'][1]})" for c in cols]))
    R.append(("— з них малі < 32 px (інтернет + фото)", [f"{pct(*c['small'])} ({c['small'][0]}/{c['small'][1]})" for c in cols]))
    R.append(("— з них великі ≥ 96 px (зблизька + фото)", [f"{pct(*c['large'])} ({c['large'][0]}/{c['large'][1]})" for c in cols]))
    R.append(("точність рамок — інтернет-кадри", [f"{100 * c['prec|test_inet_clean']:.1f} %" if c['prec|test_inet_clean'] is not None else "—" for c in cols]))
    R.append(("**фото БЕЗ дрона з хибною рамкою**", [f"**{pct(*c['bg'])}** ({c['bg'][0]}/{c['bg'][1]})" for c in cols]))
    for s, name in BG:
        R.append((f"— {name}", [f"{pct(*c[f'bg|{s}'])} ({c[f'bg|{s}'][0]}/{c[f'bg|{s}'][1]})" for c in cols]))
    R.append(("станція: кадрів з хибною рамкою (899)", [f"{pct(*c['st_fpf'])} ({c['st_fpf'][0]})" for c in cols]))
    R.append(("станція: хибних рамок на 100 кадрів", [f"{100 * c['st_fp'][0] / c['st_fp'][1]:.1f}" for c in cols]))
    R.append(("станція «periodic» (без упередження): кадрів з хибною", [f"{pct(*c['per_fpf'])} ({c['per_fpf'][0]}/{c['per_fpf'][1]})" for c in cols]))
    R.append(("⚠️ станція: дронів знайдено (упереджено на користь ft_v5)", [f"{pct(*c['st_found'])} ({c['st_found'][0]}/{c['st_found'][1]})" for c in cols]))
    R.append(("⚠️ дрон у руці на станції (упереджено)", [f"{pct(*c['hand'])} ({c['hand'][0]}/{c['hand'][1]})" for c in cols]))
    return R


def table(title, heads, R):
    L = [f"### {title}", "", "| | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
    L += [f"| {name} | " + " | ".join(vals) + " |" for name, vals in R]
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("out")
    ap.add_argument("--models", required=True)
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    models = a.models.split(",")
    names = a.names.split(",") if a.names else models
    tv = rep.get("val_threshold_maxF1", {})
    L = [f"Поріг з val (max F1 на v8-val, тест на вибір не впливає): " +
         ", ".join(f"{n} {tv.get(m)}" for m, n in zip(models, names)), ""]
    for t in ("@0.30", "@0.40", "@0.55"):
        L += table(f"Поріг {t[1:]}", names, rows([col(rep, m, t) for m in models]))
    # each model at its own val threshold
    L += table("Кожна модель на своєму порозі з val", [f"{n} ({tv.get(m)})" for m, n in zip(models, names)],
               rows([col(rep, m, f"@{tv[m]:.2f}") for m in models]))
    # mAP
    L += ["### mAP (не залежить від порогу)", "", "| набір | " + " | ".join(f"{n} mAP50 / mAP50-95" for n in names) + " |",
          "|---|" + "---|" * len(names)]
    for s, name in DRONE[:3] + [("val", "v8 val")]:
        L.append(f"| {name} | " + " | ".join(
            f"{rep['models'][m][s]['mAP50'] * 100:.1f} / {rep['models'][m][s]['mAP50-95'] * 100:.1f}" for m in models) + " |")
    L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

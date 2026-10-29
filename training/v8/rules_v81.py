#!/usr/bin/env python3
"""
rules_v81.py — the v8.1 decision rules (CHANGELOG, written 14.10.2026 BEFORE
any v8.1 result), evaluated mechanically from an eval_v8.py score report.

    python rules_v81.py REPORT.json CANDIDATE [--thr 0.40]

small far   test_inet < 32 px + drone_photos_eye < 32 px + owner7 + drones on
            the station's "periodic" frames: found >= ft_v5
large close drone_close + drone_photos_eye >= 96 px: found > ft_v5 AND > ft_v7
false       drone-free photos (all background sets) with a false box <= ft_v5;
            false boxes on "periodic" station frames <= ft_v5
(birds per set and the biased station "detection" frames are printed, not judged)
"""
import argparse
import json

BG = ["bg_birds_cam", "bg_birds_dvb", "bg_seagulls", "bg_landscape", "bg_weather", "coco_neg",
      "station_bg", "station_trees"]


def found_size(r, s, t, keys):
    fs = r[s][t]["found_by_size"]
    f = sum(int(fs[k].split("/")[0]) for k in keys if k in fs)
    n = sum(int(fs[k].split("/")[1]) for k in keys if k in fs)
    return f, n


def metrics(rep, m, t, inet="test_inet"):
    r = rep["models"][m]
    out = {}
    a = found_size(r, inet, t, ("<16px", "16-32px"))
    b = found_size(r, "drone_photos_eye", t, ("<16px", "16-32px"))
    o = (r["owner7"]["labels"] - r["owner7"][t]["missed"], r["owner7"]["labels"])
    st = r["station"][t]
    p = (sum(v.get("periodic_drone_found", 0) for v in st.values()),
         sum(v.get("periodic_drone_objects", 0) for v in st.values()))
    out["small"] = (a[0] + b[0] + o[0] + p[0], a[1] + b[1] + o[1] + p[1])
    out["small_parts"] = {inet + "<32": a, "photos<32": b, "owner7": o, "station_periodic": p}
    c = (r["drone_close"]["labels"] - r["drone_close"][t]["missed"], r["drone_close"]["labels"])
    d = found_size(r, "drone_photos_eye", t, ("96-320px", ">=320px"))
    out["large"] = (c[0] + d[0], c[1] + d[1])
    out["large_parts"] = {"drone_close": c, "photos>=96": d}
    out["bg_imgs_fp"] = sum(r[s][t]["images_with_false_box"] for s in BG)
    out["bg_imgs"] = sum(r[s]["images"] for s in BG)
    out["bg_parts"] = {s: r[s][t]["images_with_false_box"] for s in BG}
    out["periodic_fp"] = sum(v.get("periodic_false_boxes", 0) for v in st.values())
    out["station_detection_found"] = (sum(v["drone_found"] for v in st.values()),
                                      sum(v["drone_objects"] for v in st.values()))
    out["station_fp"] = sum(v["false_boxes"] for v in st.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("candidate")
    ap.add_argument("--thr", default="0.40")
    ap.add_argument("--v5", default="ft_v5", help="report name of the ft_v5 model (e.g. hef_v5)")
    ap.add_argument("--v7", default="ft_v7", help="report name of the ft_v7 model (e.g. hef_v7)")
    ap.add_argument("--inet", default="test_inet", help="internet test set for the small-drone rule")
    a = ap.parse_args()
    rep = json.loads(open(a.report, encoding="utf-8").read())
    t = f"@{float(a.thr):.2f}"
    M = {m: metrics(rep, m, t, a.inet) for m in rep["models"]}
    for m, v in M.items():
        print(f"{m:7s} small {v['small'][0]}/{v['small'][1]} {v['small_parts']} | large {v['large'][0]}/{v['large'][1]} "
              f"{v['large_parts']} | bg photos with FP {v['bg_imgs_fp']}/{v['bg_imgs']} | periodic FP {v['periodic_fp']} "
              f"| station(detection, biased) found {v['station_detection_found']} FP {v['station_fp']}")
        print(f"        bg by set {v['bg_parts']}")
    c, v5, v7 = M[a.candidate], M[a.v5], M[a.v7]
    rules = {
        "small far >= ft_v5": c["small"][0] >= v5["small"][0],
        "large close > ft_v5 and > ft_v7": c["large"][0] > v5["large"][0] and c["large"][0] > v7["large"][0],
        "drone-free photos with FP <= ft_v5": c["bg_imgs_fp"] <= v5["bg_imgs_fp"],
        "periodic station FP <= ft_v5": c["periodic_fp"] <= v5["periodic_fp"],
    }
    print(f"\nrules for {a.candidate} at {t}:")
    for k, ok in rules.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print("VERDICT:", "ALL PASS" if all(rules.values()) else "NOT PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_testsets.py — the ft_v8 INDEPENDENT test sets, gathered in one place
(v8/testsets/), never used for training or model selection.

    python build_testsets.py

image sets   testsets/<name>/images + labels   (copied; YOLO txt, empty = no drone)
station      testsets/station/<session>/  (published + native frames, copied)
             testsets/station_gt.json: per frame the by-eye objects
             (D drone, U unclear, F not a drone) in published px:
               * frames_unseen sessions — eval_v7/sess_scene (round 7 by eye)
               * live flight 025044      — eval_v7/pi_v6 live_clusters + labels_live
               * indoor 101214/104632/110828, extra 082902/140442/142959 — none yet
             An object is the extent of the model boxes that formed it.
"""
import json
import shutil
import sys
from pathlib import Path

DT = Path(r"C:\Github\drone_training")
EV7 = DT / "eval_v7"
OUT = DT / "v8" / "testsets"
SCR_SESS = Path(r"C:\Users\Nazar\AppData\Local\Temp\claude\c--Github-THIS-IS-CAMERA-ANDAU"
                r"\1af7577f-dcd9-409f-9d28-467db2513d82\scratchpad\pi_testsets\sessions")
sys.path.insert(0, str(EV7 / "pi_v6"))
from labels_live import D as LIVE_D, U as LIVE_U  # noqa: E402

TRAIN_SESSIONS = {"frames_20260919_140445", "frames_20260919_141424", "frames_20260919_160325",
                  "frames_20260919_160513", "frames_20260919_180118"}
IMAGE_SETS = {  # name: (source dir with images/ + labels/ or plain images, note)
    "drone_close": (DT / "eval_v5" / "drone_close", "large close drones, internet photos"),
    "drone_photos": (DT / "eval_v5" / "drone_photos", "drone photos WITHOUT boxes: image-level only"),
    "hand_white": (DT / "eval_v5" / "hand_white", "white drone in a hand, station 033903/042047"),
    "hand_xdrone": (DT / "eval_v5" / "hand_xdrone", "X-drone in a hand, station 042047"),
    "owner7": (DT / "user_photos" / "val", "the owner's 7 photos of small drones"),
    "bg_birds_cam": (DT / "eval_v5" / "bg_birds_cam", "birds, no drone"),
    "bg_birds_dvb": (DT / "eval_v5" / "bg_birds_dvb", "birds (drone-vs-bird), no drone"),
    "bg_seagulls": (DT / "eval_v5" / "bg_seagulls", "seagulls, no drone"),
    "bg_landscape": (DT / "eval_v5" / "bg_landscape", "landscapes, no drone"),
    "bg_weather": (DT / "eval_v5" / "bg_weather", "fog/rain/snow/sand, no drone"),
    "station_trees": (DT / "eval_v5" / "station_trees", "station trees - from TRAIN recording 180118"),
    "station_bg": (EV7 / "pi_testsets" / "bg", "station backgrounds (pi_testsets/bg), no drone"),
    "coco_neg": (DT / "v8" / "parts" / "coco_neg" / "test", "COCO people/vehicles/phones/street, no drone"),
}
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_image_set(name, src):
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    idir = src / "images" if (src / "images").is_dir() else src
    n = nb = 0
    for p in sorted(idir.iterdir()):
        if p.suffix.lower() not in IMG:
            continue
        shutil.copy2(p, dst / "images" / p.name)
        lp = src / "labels" / (p.stem + ".txt")
        if not lp.exists():
            lp = p.with_suffix(".txt")            # labels next to the image (user_photos)
        txt = lp.read_text() if lp.exists() else ""
        (dst / "labels" / (p.stem + ".txt")).write_text(txt)
        n += 1
        nb += sum(1 for l in txt.splitlines() if l.strip())
    return n, nb


def extent(boxes):
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x1 = [b[0] + b[2] for b in boxes]
    y1 = [b[1] + b[3] for b in boxes]
    return [min(xs), min(ys), max(x1) - min(xs), max(y1) - min(ys)]


def session_dir(s):
    for root in (DT / "station_raw", DT / "station_raw_extra", SCR_SESS):
        if (root / s).is_dir():
            return root / s
    raise FileNotFoundError(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (src, note) in IMAGE_SETS.items():
        n, nb = copy_image_set(name, src)
        report[name] = {"images": n, "boxes": nb, "note": note}
        print(f"{name:14s} {n:5d} images {nb:5d} boxes  {note}")
    frames = []
    # round-7 by-eye scene labels (frames_unseen sessions)
    fu = json.loads((EV7 / "pi_v57" / "frames_unseen.json").read_text())
    so = json.loads((EV7 / "sess_scene" / "sess_objects.json").read_text())
    sl = json.loads((EV7 / "sess_scene" / "sess_labels.json").read_text())
    byf = {}
    for o in so:
        byf.setdefault(o["frame"], []).append(o)
    for i, fr in enumerate(fu):
        if fr["session"] in TRAIN_SESSIONS:
            continue
        objs = [{"src": f"sess{o['id']}", "lab": sl[str(o["id"])], "box": extent([b[1][:4] for b in o["boxes"]])}
                for o in byf.get(i, [])]
        frames.append({"session": fr["session"], "pub": fr["pub"], "nat": fr["nat"], "reason": fr["reason"],
                       "t": fr["t"], "objects": objs, "gt_from": "round7 sess_scene"})
    # live flight 025044
    li = json.loads((EV7 / "pi_testsets" / "live_index.json").read_text())
    lc = json.loads((EV7 / "pi_v6" / "live_clusters.json").read_text())
    byl = {}
    for c in lc:
        byl.setdefault(c["frame"], []).append(c)
    for i, fr in enumerate(li):
        objs = [{"src": f"live{c['id']}", "lab": "D" if c["id"] in LIVE_D else "U" if c["id"] in LIVE_U else "F",
                 "box": extent([b[1][:4] for b in c["boxes"]])} for c in byl.get(i, [])]
        frames.append({"session": "frames_20260920_025044", "pub": fr["pub"], "nat": fr["nat"],
                       "reason": fr["reason"], "t": fr["t"], "objects": objs, "gt_from": "round6 live_clusters"})
    # indoor 'today' + extra sessions: frames only, labels to be added by eye
    ti = json.loads((EV7 / "pi_testsets" / "today_index.json").read_text())
    for fr in ti:
        frames.append({"session": fr["session"], "pub": fr["pub"], "nat": fr["nat"], "reason": fr["reason"],
                       "t": fr["t"], "objects": [], "gt_from": "none yet"})
    for s in ("frames_20260920_082902", "frames_20260920_140442", "frames_20260920_142959"):
        d = DT / "station_raw_extra" / s
        jl = {}
        for line in (d / "frames.jsonl").read_text().splitlines():
            try:
                e = json.loads(line)
                jl[Path(e.get("pub", e.get("file", ""))).name] = e
            except json.JSONDecodeError:
                pass
        for p in sorted(d.glob("*_pub.jpg")):
            frames.append({"session": s, "pub": p.name, "nat": p.name.replace("_pub.jpg", "_nat.jpg"),
                           "reason": jl.get(p.name, {}).get("reason", "?"), "t": jl.get(p.name, {}).get("t"),
                           "objects": [], "gt_from": "none yet"})
    # copy frames
    for fr in frames:
        d = OUT / "station" / fr["session"]
        d.mkdir(parents=True, exist_ok=True)
        sd = session_dir(fr["session"])
        for k in ("pub", "nat"):
            if not (d / fr[k]).exists():
                shutil.copy2(sd / fr[k], d / fr[k])
    (OUT / "station_gt.json").write_text(json.dumps(frames, indent=0))
    by = {}
    for fr in frames:
        b = by.setdefault(fr["session"], {"frames": 0, "D": 0, "U": 0, "F": 0, "frames_with_D": 0})
        b["frames"] += 1
        labs = [o["lab"] for o in fr["objects"]]
        for x in labs:
            b[x] += 1
        b["frames_with_D"] += "D" in labs
    report["station"] = by
    for s, b in sorted(by.items()):
        print(s, b)
    (OUT / "testsets.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

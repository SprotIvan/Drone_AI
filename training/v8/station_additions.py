#!/usr/bin/env python3
"""
station_additions.py — frames of the station's TRAINING recordings (the ones
ft_v5 already trained on: 140445, 141424, 160325, 160513, 180118) that no
training set contains yet, as additions for ft_v8. Built only from by-eye
decisions; a model box is a proposal until a sheet has been checked.

    python station_additions.py review OUT       sheets to check
    python station_additions.py export OUT DECISIONS.json DATASET_PART_DIR

review   D frames: the published frame with the proposed drone box (the best
         scoring model box of every object labelled D) + a x2 zoom of it;
         F frames (only non-drone objects): the published frame with those
         objects marked, to confirm there is NO drone anywhere in it.
export   per decided frame: the published 640x480 frame + the 6 native
         640x640 crop-mode windows (smart_crop.tile_origins, as the station
         and station_frames.py cut them); a window that cuts a drone box is
         left out, as station_frames.py does. DECISIONS.json:
         {"drone": {frame_key: [[x, y, w, h], ...] in published px},
          "background": [frame_key, ...]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
from smart_crop import tile_origins  # noqa: E402

SCR = Path(r"C:\Users\Nazar\AppData\Local\Temp\claude\c--Github-THIS-IS-CAMERA-ANDAU\1af7577f-dcd9-409f-9d28-467db2513d82\scratchpad")
RAW = Path(r"C:\Github\drone_training\station_raw")
SES = SCR / "pi_testsets" / "sessions"
TRAIN = ("frames_20260919_140445", "frames_20260919_141424", "frames_20260919_160325",
         "frames_20260919_160513", "frames_20260919_180118")
CROP = 640


def sess_dir(s):
    return RAW / s if (RAW / s).is_dir() else SES / s


def load():
    frames = json.loads((SCR / "pi_v57" / "frames_unseen.json").read_text())
    objs = json.loads((SCR / "sess_scene" / "sess_objects.json").read_text())
    lab = json.loads((SCR / "sess_scene" / "sess_labels.json").read_text())
    byf = {}
    for o in objs:
        o["lab"] = lab[str(o["id"])]
        byf.setdefault(o["frame"], []).append(o)
    return frames, byf


def key(fr):
    return f"{fr['session']}/{fr['pub']}"


def best_box(o):
    m, b = max(o["boxes"], key=lambda mb: mb[1][4])
    return [round(v, 1) for v in b[:4]], round(b[4], 3), m


def review(out):
    frames, byf = load()
    out.mkdir(parents=True, exist_ok=True)
    dfr, ffr, prop = [], [], {}
    for i, fr in enumerate(frames):
        if fr["session"] not in TRAIN:
            continue
        ob = byf.get(i, [])
        labs = {o["lab"] for o in ob}
        if "U" in labs:
            continue
        if "D" in labs:
            dfr.append(i)
            prop[key(fr)] = [best_box(o) for o in ob if o["lab"] == "D"]
        elif "F" in labs:
            ffr.append(i)
    (out / "proposals.json").write_text(json.dumps({"drone": prop, "background_candidates":
                                                    [key(frames[i]) for i in ffr]}, indent=0))

    def tile(i, mode):
        fr = frames[i]
        pub = cv2.imread(str(sess_dir(fr["session"]) / fr["pub"]))
        canvas = cv2.resize(pub, (480, 360))
        k = 480 / 640
        panels = [canvas]
        for o in byf.get(i, []):
            if mode == "D" and o["lab"] != "D":
                continue
            (x, y, w, h), sc, m = best_box(o)
            col = (0, 255, 0) if o["lab"] == "D" else (0, 0, 255)
            cv2.rectangle(canvas, (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k)), col, 2)
            if mode == "D":
                cx, cy = x + w / 2, y + h / 2
                half = max(w, h) * 0.8 + 12
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                x1, y1 = int(min(640, cx + half)), int(min(480, cy + half))
                z = pub[y0:y1, x0:x1].copy()
                cv2.rectangle(z, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 255, 0), 1)
                panels.append(cv2.resize(z, (360, 360), interpolation=cv2.INTER_NEAREST))
        cv2.putText(canvas, f"{i} {fr['session'][16:]}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if mode == "D":
            while len(panels) < 2:
                panels.append(np.zeros((360, 360, 3), np.uint8))
            return np.hstack(panels[:2])
        return canvas

    for name, ids, mode, cols in (("drone", dfr, "D", 3), ("bg", ffr, "F", 4)):
        per = cols * 4
        for s in range(0, len(ids), per):
            tiles = [tile(i, mode) for i in ids[s:s + per]]
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(out / f"{name}_{s // per:02d}.jpg"),
                        np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(dfr)} drone frames, {len(ffr)} background candidates -> {out}")


def yolo(b, w, h):
    x, y, bw, bh = b
    return f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def export(out, decisions, dst):
    frames = {key(fr): fr for fr in json.loads((SCR / "pi_v57" / "frames_unseen.json").read_text())}
    dec = json.loads(Path(decisions).read_text())
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = {"drone_frames": 0, "bg_frames": 0, "files": 0, "windows_left_out": 0, "boxes": 0}
    items = [(k, v) for k, v in dec.get("drone", {}).items()] + [(k, []) for k in dec.get("background", [])]
    for k, boxes in items:
        fr = frames[k]
        d = sess_dir(fr["session"])
        stem = f"st8_{fr['session'][7:]}_{fr['pub'][:-len('_pub.jpg')]}"
        pub = cv2.imread(str(d / fr["pub"]))
        nat = cv2.imread(str(d / fr["nat"]))
        ph, pw = pub.shape[:2]
        n["drone_frames" if boxes else "bg_frames"] += 1
        cv2.imwrite(str(dst / "images" / f"{stem}_pub.jpg"), pub, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dst / "labels" / f"{stem}_pub.txt").write_text("".join(yolo(b, pw, ph) + "\n" for b in boxes))
        n["files"] += 1
        n["boxes"] += len(boxes)
        kk = nat.shape[1] / pw
        nb = [[v * kk for v in b] for b in boxes]
        for i, (ox, oy) in enumerate(tile_origins((nat.shape[1], nat.shape[0]), CROP)):
            inside, touched = [], False
            for x, y, w, h in nb:
                if x >= ox and y >= oy and x + w <= ox + CROP and y + h <= oy + CROP:
                    inside.append([x - ox, y - oy, w, h])
                elif x < ox + CROP and x + w > ox and y < oy + CROP and y + h > oy:
                    touched = True
            if touched:
                n["windows_left_out"] += 1
                continue
            cv2.imwrite(str(dst / "images" / f"{stem}_tile{i}.jpg"), nat[oy:oy + CROP, ox:ox + CROP],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (dst / "labels" / f"{stem}_tile{i}.txt").write_text("".join(yolo(b, CROP, CROP) + "\n" for b in inside))
            n["files"] += 1
            n["boxes"] += len(inside)
    print(json.dumps(n))
    (dst / "export_stats.json").write_text(json.dumps(n, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "review":
        review(Path(sys.argv[2]))
    else:
        export(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))

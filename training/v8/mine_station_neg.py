#!/usr/bin/env python3
"""
mine_station_neg.py — hard negatives from a station recording WITHOUT a drone
(160513): run a model the station's way (published frame + 6 native windows,
smart_crop merge) on every frame, keep the frames where it puts a box, and
spread the pick over time. Every pick is drawn on review sheets — a frame is
only used after checking by eye that there is no drone in it.

    python mine_station_neg.py MODEL.pt SESSION_DIR OUT_DIR [--thr 0.25] [--n 150] [--gap 20]

Excluded: frames already in a training set (v5 station set, v8 station_add,
by published-frame stem), frames with an unclear (U) object in the round-7
scene review. --gap: at most one frame per GAP seconds of the recording,
best-scoring first, then fill with frames without a box (also gap-spaced).
Writes OUT_DIR/picked.json (frame, boxes) and OUT_DIR/sheet_NN.jpg.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\acoustic_radar")
import eval_v8 as E  # noqa: E402
from smart_crop import tile_origins  # noqa: E402

DT = Path(r"C:\Github\drone_training")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--gap", type=float, default=20.0)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    sess = Path(a.session)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = sess.name
    used = {p.name.replace(f"{name}_", "", 1) for p in (DT / "station").glob(f"{name}_*_pub.jpg")}
    used |= {p.name.replace(f"st8_{name[7:]}_", "", 1) for p in
             (DT / "v8" / "parts" / "station_train" / "images").glob(f"st8_{name[7:]}_*_pub.jpg")}
    fu = json.loads((DT / "eval_v7" / "pi_v57" / "frames_unseen.json").read_text())
    objs = json.loads((DT / "eval_v7" / "sess_scene" / "sess_objects.json").read_text())
    lab = json.loads((DT / "eval_v7" / "sess_scene" / "sess_labels.json").read_text())
    unclear = {fu[o["frame"]]["pub"] for o in objs if o["session"] == name and lab[str(o["id"])] == "U"}
    jl = {}
    for line in (sess / "frames.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
            jl[e["files"]["pub"]] = e
        except (json.JSONDecodeError, KeyError):
            pass
    pubs = sorted(p.name for p in sess.glob("*_pub.jpg"))
    stats = {"frames": len(pubs), "already_in_training": 0, "unclear_U": 0, "unreadable": 0}
    cand = []
    for s in range(0, len(pubs), 16):
        chunk, ims, wins, meta = [], [], [], []
        for pub in pubs[s:s + 16]:
            if pub in used:
                stats["already_in_training"] += 1
                continue
            if pub in unclear:
                stats["unclear_U"] += 1
                continue
            p = cv2.imread(str(sess / pub))
            n = cv2.imread(str(sess / pub.replace("_pub.jpg", "_nat.jpg")))
            if p is None or n is None:
                stats["unreadable"] += 1
                continue
            can, r, l, t = E.letterbox(p)
            origins = tile_origins((n.shape[1], n.shape[0]), 640)
            chunk.append(pub)
            ims.append((can, r, l, t))
            wins.append([n[oy:oy + 640, ox:ox + 640] for ox, oy in origins])
            meta.append((origins, (n.shape[1], n.shape[0])))
        if not chunk:
            continue
        pp = E.run(model, [x[0] for x in ims], 0.05)
        tp = E.run(model, [w for ws in wins for w in ws], 0.05)
        k = 0
        for pub, (can, r, l, t), boxes, (origins, nsize) in zip(chunk, ims, pp, meta):
            full = [[(b[0] - l) / r, (b[1] - t) / r, (b[2] - l) / r, (b[3] - t) / r, b[4]] for b in boxes]
            ws = []
            for _ in origins:
                ws.append(tp[k])
                k += 1
            fb = E.station_boxes({"full": full, "wins": ws, "origins": origins, "native": nsize}, a.thr)
            cand.append({"pub": pub, "t": jl.get(pub, {}).get("t"), "reason": jl.get(pub, {}).get("reason"),
                         "boxes": [[round(v, 1) for v in b[:4]] + [round(b[4], 3)] for b in fb]})
    stats["candidates"] = len(cand)
    stats["with_box"] = sum(1 for c in cand if c["boxes"])
    # spread over time: boxed frames first (best score first), then frames without a box
    order = sorted(cand, key=lambda c: (-max([b[4] for b in c["boxes"]], default=0)))
    picked = []
    for c in order:
        if len(picked) >= a.n:
            break
        if c["t"] is not None and any(p["t"] is not None and abs(p["t"] - c["t"]) < a.gap for p in picked):
            continue
        picked.append(c)
    picked.sort(key=lambda c: c["pub"])
    stats["picked"] = len(picked)
    stats["picked_with_box"] = sum(1 for c in picked if c["boxes"])
    (out / "picked.json").write_text(json.dumps({"stats": stats, "picked": picked}, indent=0))
    tiles = []
    for i, c in enumerate(picked):
        im = cv2.imread(str(sess / c["pub"]))
        for x, y, w, h, s in c["boxes"]:
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(im, f"{s:.2f}", (int(x), max(12, int(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{i} {c['pub'][9:15]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(im)
    cols, per = 6, 36
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        while len(t) % cols:
            t.append(np.zeros((240, 320, 3), np.uint8))
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

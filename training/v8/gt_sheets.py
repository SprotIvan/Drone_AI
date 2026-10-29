#!/usr/bin/env python3
"""
gt_sheets.py — every frame of the station test recordings on numbered sheets,
to complete the ground truth by eye (a drone no model boxed must still count).

    python gt_sheets.py OUT_DIR [--sessions 042047,025044] [--tile 480] [--cols 4] [--rows 4]

Overlays (published px): labelled D objects green, U yellow, F thin red, and the
eval_v8 'new' objects (boxes near no labelled object) magenta with their id.
Tile title: frame index in station_gt.json, session, time.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TS = Path(r"C:\Github\drone_training\v8\testsets")
NEW = Path(r"C:\Github\drone_training\v8\eval\new_objects\new_objects.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads((TS / "station_gt.json").read_text())
    new = json.loads(NEW.read_text()) if NEW.exists() else []
    newk = {}
    for o in new:
        newk.setdefault(o["key"], []).append(o)
    want = set(s for s in a.sessions.split(",") if s)
    tw, th = a.tile, a.tile * 3 // 4
    k = tw / 640
    per = a.cols * a.rows
    for sess in sorted({fr["session"] for fr in gt}):
        if want and sess[16:] not in want:
            continue
        idx = [i for i, fr in enumerate(gt) if fr["session"] == sess]
        for s in range(0, len(idx), per):
            tiles = []
            for i in idx[s:s + per]:
                fr = gt[i]
                im = cv2.resize(cv2.imread(str(TS / "station" / sess / fr["pub"])), (tw, th),
                                interpolation=cv2.INTER_AREA)
                for o in fr["objects"]:
                    x, y, w, h = o["box"]
                    col, t = {"D": ((0, 255, 0), 2), "U": ((0, 255, 255), 1), "F": ((0, 0, 255), 1)}[o["lab"]]
                    cv2.rectangle(im, (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k)), col, t)
                for o in newk.get(f"{sess}/{fr['pub']}", []):
                    x, y, w, h = o["box"]
                    cv2.rectangle(im, (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k)), (255, 0, 255), 1)
                    cv2.putText(im, str(o["id"]), (int(x * k), max(10, int(y * k) - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 0, 255), 1)
                cv2.rectangle(im, (0, 0), (tw - 1, 15), (0, 0, 0), -1)
                cv2.putText(im, f"{i} {sess[16:]} {fr['pub'][9:15]} {fr['reason'][:3]}", (3, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                tiles.append(im)
            while len(tiles) % a.cols:
                tiles.append(np.zeros((th, tw, 3), np.uint8))
            cv2.imwrite(str(out / f"{sess[16:]}_{s // per:02d}.jpg"),
                        np.vstack([np.hstack(tiles[j:j + a.cols]) for j in range(0, len(tiles), a.cols)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
    print("sheets ->", out)


if __name__ == "__main__":
    main()

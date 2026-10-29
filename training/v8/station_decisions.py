#!/usr/bin/env python3
"""
station_decisions.py — the by-eye decisions on the station training-session
frames (review sheets of station_additions.py review + native-resolution
zooms), written out as DECISIONS.json for station_additions.py export, and a
check sheet of every final box on the native frame.

    python station_decisions.py OUT_DIR

Every number below was decided by looking at the frame; nothing here is a
model output taken on trust. Frame ids are indices into frames_unseen.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import station_additions as SA  # noqa: E402

K = 640 / 1320                       # native 1320x990 -> published 640x480

# 140445: many frames have a second box on a white object at the hip that can
# not be told apart (phone or the white drone) -> no positive of this session.
SKIP_POSITIVE_SESSIONS = {"frames_20260919_140445"}

# 141424, rejected: box only on part of the drone (564, 573), loose (579, 586),
# box on the face + hand rather than on the drone (594 ... 640).
REJECT = {564, 573, 579, 586, 594, 606, 611, 627, 628, 629, 634, 636, 640}

# 180118: the station's crop windows split a near drone into a full box and a
# part box (the part starts at the window edge, pub x ~330); keep the full one.
LARGEST = {2472, 2474, 2495, 2497, 2499, 2508, 2516, 2517, 2518, 2519}

# Manual boxes, NATIVE px x1, y1, x2, y2, read off grid zooms and checked drawn
# back on the frame (review/station/manual_check*.jpg, manual_far3.jpg).
MANUAL = {
    # drone clearly visible, NO model boxed it (white Mavic-type against trees)
    2461: [(323, 29, 470, 67)], 2485: [(302, 353, 435, 409)], 2487: [(518, 372, 696, 441)],
    2488: [(627, 325, 785, 377)], 2489: [(474, 235, 630, 285)], 2490: [(477, 8, 629, 56)],
    2492: [(568, 468, 760, 523)],
    # small far drone in the trees, found while checking background candidates
    2357: [(219, 654, 300, 681)], 2368: [(487, 686, 569, 709)], 2372: [(420, 717, 501, 741)],
    2452: [(572, 398, 633, 430)], 2453: [(556, 470, 624, 491)],
    # model box loose (reached well below the drone) or shifted / cut
    2449: [(515, 333, 582, 353)], 2450: [(534, 340, 600, 360)], 2451: [(580, 366, 648, 393)],
    2491: [(559, 335, 727, 376)], 2493: [(563, 375, 788, 434)], 2501: [(582, 392, 861, 499)],
    2502: [(598, 297, 922, 408)], 2352: [(673, 215, 812, 260)], 2402: [(372, 300, 438, 325)],
    2444: [(630, 436, 698, 459)], 2482: [(192, 303, 245, 326)], 2486: [(522, 411, 679, 453)],
    # drone split into two part boxes -> one box over the whole drone
    2470: [(506, 197, 840, 322)], 2503: [(471, 264, 867, 391)],
}

# Backgrounds. 160513 is a recording without a drone (v5 has 547 files of it,
# none with a box; every object of it was labelled not-a-drone in the round-7
# scene review): its frames where a model put a false box are hard negatives
# (leaves, car wheels/lights, bollards, clothes, people, a phone in hand).
# 140445 / 141424 candidates are NOT used: the person in them may be holding
# the drone. 180118 candidates are NOT used except the plain wall frames: a
# far drone was found unboxed in several of them (2357, 2368, 2372, 2452, 2453
# moved to MANUAL) and white blobs in 2479, 2521, 2523 and a dot in 2456 can
# not be identified.
BG_SESSIONS = {"frames_20260919_160513"}
BG_EXTRA = {690, 2328, 2329, 2331, 2332, 2333, 2334}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    frames, byf = SA.load()
    drone, bg, log = {}, [], []
    for i, fr in enumerate(frames):
        if fr["session"] not in SA.TRAIN:
            continue
        ob = byf.get(i, [])
        labs = {o["lab"] for o in ob}
        k = SA.key(fr)
        if i in MANUAL:
            drone[k] = [[x1 * K, y1 * K, (x2 - x1) * K, (y2 - y1) * K] for x1, y1, x2, y2 in MANUAL[i]]
            log.append((i, "manual", len(MANUAL[i])))
            continue
        if "U" in labs:
            continue
        if "D" in labs:
            if fr["session"] in SKIP_POSITIVE_SESSIONS or i in REJECT:
                log.append((i, "rejected", 0))
                continue
            boxes = [SA.best_box(o)[0] for o in ob if o["lab"] == "D"]
            if i in LARGEST:
                boxes = [max(boxes, key=lambda b: b[2] * b[3])]
            drone[k] = boxes
            log.append((i, "largest" if i in LARGEST else "accepted", len(boxes)))
        elif "F" in labs and (fr["session"] in BG_SESSIONS or i in BG_EXTRA):
            bg.append(k)
            log.append((i, "background", 0))
    dec = {"drone": drone, "background": bg}
    (out / "DECISIONS.json").write_text(json.dumps(dec, indent=1))
    by = {}
    for i, what, _ in log:
        s = frames[i]["session"][16:]
        by.setdefault(s, {}).setdefault(what, 0)
        by[s][what] += 1
    print(json.dumps(by, indent=1))
    print(f"drone frames {len(drone)}, boxes {sum(len(v) for v in drone.values())}, background frames {len(bg)}")

    # check sheet: every final drone box on the native frame
    idx = {SA.key(fr): i for i, fr in enumerate(frames)}
    tiles = []
    for k, boxes in drone.items():
        fr = frames[idx[k]]
        nat = cv2.imread(str(SA.sess_dir(fr["session"]) / fr["nat"]))
        s = nat.shape[1] / 640
        for x, y, w, h in boxes:
            cx, cy = (x + w / 2) * s, (y + h / 2) * s
            half = max(w, h) * s * 0.75 + 16
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(nat.shape[1], cx + half)), int(min(nat.shape[0], cy + half))
            z = nat[y0:y1, x0:x1].copy()
            cv2.rectangle(z, (int(x * s - x0), int(y * s - y0)), (int((x + w) * s - x0), int((y + h) * s - y0)),
                          (0, 255, 0), 1)
            z = cv2.resize(z, (240, 240), interpolation=cv2.INTER_CUBIC)
            cv2.putText(z, f"{idx[k]} {w:.0f}x{h:.0f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(z)
    cols, per = 8, 40
    for s0 in range(0, len(tiles), per):
        t = tiles[s0:s0 + per]
        while len(t) % cols:
            t.append(np.zeros((240, 240, 3), np.uint8))
        cv2.imwrite(str(out / f"final_boxes_{s0 // per}.jpg"),
                    np.vstack([np.hstack(t[j:j + cols]) for j in range(0, len(t), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()

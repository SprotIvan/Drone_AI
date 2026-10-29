#!/usr/bin/env python3
"""
station_gt_add.py — by-eye additions to the station test ground truth
(testsets/station_gt.json comes from rounds 6/7 and only knows objects some
model boxed). Written to testsets/station_new_labels.json, which
eval_v8.load_station_gt() appends to the frames.

    python station_gt_add.py

D  a drone I can see (also when no model boxed it)  -> counts for recall
U  something in a hand / a sky dot I can not identify -> ignored both ways
Boxes in PUBLISHED px (x1, y1, x2, y2), read off grid zooms (gt_assist.py,
review/gt_zoom/*.jpg) and drawn back to check. D/U are keyed by the frame's position in
station_gt.json (guarded by ORDER_MD5); NEW_D/NEW_U by frame key.
"""
import hashlib
import json
from pathlib import Path

TS = Path(r"C:\Github\drone_training\v8\testsets")
ORDER_MD5 = "ec086d5b849d1c681726d59a1adcb5e0"   # md5 of the 899 frame keys in station_gt.json order

# station_gt.json frame index -> boxes
D = {
    # 042047: X-drone carried at the hip / held up (review/gt_zoom/c01.jpg)
    78: [(182, 199, 204, 218)], 79: [(157, 210, 180, 229)], 81: [(181, 203, 197, 226)],
    90: [(159, 201, 187, 224)], 91: [(164, 211, 178, 228)], 92: [(158, 211, 182, 226)],
    93: [(159, 210, 184, 227)], 105: [(163, 195, 185, 228)], 106: [(168, 204, 184, 228)],
    # 042047: white drone held up high (review/gt_zoom/z04.jpg, c01.jpg)
    109: [(255, 86, 270, 99)], 111: [(252, 82, 267, 92)], 112: [(250, 81, 265, 92)],
    113: [(250, 81, 265, 92)], 114: [(254, 83, 268, 94)], 115: [(255, 85, 270, 95)],
    116: [(256, 85, 270, 95)],
    # 025044 (live flight): white drone in flight / on the ground, no label yet (gt_zoom/c02.jpg)
    408: [(270, 79, 290, 92)], 562: [(226, 261, 261, 276)], 608: [(313, 223, 341, 239)],
    609: [(177, 175, 206, 194)], 610: [(151, 132, 179, 148)],
    # 025044: X-drone carried at the hip / held up in the fist (gt_zoom/l04, l05, c02.jpg)
    463: [(357, 118, 400, 158)], 489: [(269, 88, 293, 103)], 490: [(262, 96, 284, 110)],
    500: [(262, 93, 285, 110)], 504: [(325, 168, 346, 186)], 513: [(295, 159, 323, 179)],
    # 025044: white drone in front of the bus-shelter cage - found only by v8B (new_objects #101,
    # eval/new_objects/ctx_v8_b.jpg); my first pass over this frame had missed it
    573: [(78, 189, 112, 208)],
}
U = {
    # 042047: dark / white object in the hand, drone not recognisable
    76: [(201, 247, 215, 273)], 80: [(172, 204, 180, 228)], 94: [(154, 141, 160, 168)],
    104: [(179, 182, 192, 224)], 107: [(174, 211, 182, 230)], 108: [(174, 213, 182, 228)],
    117: [(230, 125, 257, 155)], 121: [(208, 169, 222, 180)], 184: [(300, 68, 318, 83)],
    # 025044: something dark / white at the raised fist, drone not recognisable (gt_zoom/l05.jpg)
    488: [(370, 112, 390, 132)], 506: [(305, 285, 334, 313)], 509: [(320, 210, 346, 242)],
    511: [(343, 218, 368, 242)],
    # 025044, objects only v8 boxed, checked at native resolution (ctx_v8_a/b.jpg) - can not tell:
    # white glint in dark foliage (401), small white/dark object behind the railing (418), blur at
    # the frame edge (436), 14x4 px object on the paving (629), dark dots in the sky (641-648)
    401: [(37, 198, 81, 223)], 418: [(429, 374, 453, 386)], 436: [(4, 305, 25, 328)],
    629: [(128, 164, 142, 168)], 641: [(378, 172, 383, 177)], 643: [(62, 57, 69, 63)],
    644: [(283, 46, 288, 50)], 647: [(309, 118, 313, 121)], 648: [(105, 12, 110, 16)],
}
# Objects first listed by `eval_v8.py new ft_v5 ft_v7` (ids 19, 21, 22, 85, 86 of
# eval/new_objects_v5v7/new_objects.json), checked by eye. Written out as explicit
# key + box: `eval_v8.py new` regenerates new_objects.json with NEW ids, and an id
# reference here once silently pointed at other objects (14.10.2026, caught and fixed).
NEW_D = {
    "frames_20260919_042047/20260919_042215_080_TELE-IMX477P_pub.jpg": [175.9, 80.0, 25.4, 29.0],   # X-drone hanging from the hand
    "frames_20260919_042047/20260919_042224_739_TELE-IMX477P_pub.jpg": [132.1, 117.2, 37.8, 34.2],  # X-drone held up
    "frames_20260919_042047/20260919_042326_952_TELE-IMX477P_pub.jpg": [226.9, 197.8, 18.7, 22.6],  # X-drone at the side
    "frames_20260920_101214/20260920_101343_969_TELE-IMX477P_pub.jpg": [7.1, 205.6, 375.4, 202.6],  # black X-drone, close
    "frames_20260920_101214/20260920_101345_032_TELE-IMX477P_pub.jpg": [43.4, 272.0, 375.0, 208.4], # black X-drone, close
}
NEW_U = {
    # 025044 03:07:55: 5x4 px dark dot in the sky (boxed by v8.1, 14.10 review, eval/new_objects/ctx_v81.jpg),
    # same kind as the U dots of frames 641-648 - can not tell bird from far drone
    "frames_20260920_025044/20260920_030755_690_TELE-IMX477P_pub.jpg": [121.3, 89.0, 4.7, 4.1],
}
# Checked by eye with NO drone: 033903 (frames 0-46), 033948, 034047 (gt sheets); the
# indoor / no-flight recordings 073211, 075158, 082430, 104632, 110828, 114101, 115750,
# 135037, 082902, 140442, 142959 (gt_nodrone sheets + frame-vs-median outliers: 747 head,
# 830 head edge, 073211/082430 camera moved - plant, wall, ceiling).


def main():
    gt = json.loads((TS / "station_gt.json").read_text())
    order = hashlib.md5("\n".join(f"{f['session']}/{f['pub']}" for f in gt).encode()).hexdigest()
    assert order == ORDER_MD5, "station_gt.json frame order changed - the frame indices below no longer apply"
    out = {}

    def add(key, lab, box, src):
        out.setdefault(key, []).append({"src": src, "lab": lab, "box": [round(float(v), 1) for v in box]})

    for lab, table in (("D", D), ("U", U)):
        for i, boxes in table.items():
            fr = gt[i]
            for x1, y1, x2, y2 in boxes:
                add(f"{fr['session']}/{fr['pub']}", lab, (x1, y1, x2 - x1, y2 - y1), "v8eye")
    keys = {f"{fr['session']}/{fr['pub']}" for fr in gt}
    for lab, table in (("D", NEW_D), ("U", NEW_U)):
        for k, box in table.items():
            assert k in keys, k
            add(k, lab, box, "v5v7new")
    (TS / "station_new_labels.json").write_text(json.dumps(out, indent=0))
    n = {"D": 0, "U": 0}
    for v in out.values():
        for o in v:
            n[o["lab"]] += 1
    print(len(out), "frames,", n)


if __name__ == "__main__":
    main()

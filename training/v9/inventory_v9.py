#!/usr/bin/env python3
"""
inventory_v9.py — the owner's field tests of 15.10.2026 (station ~/mk_crops, Pi clock names), copied to
v9/raw_15_10/: which model + threshold produced each session (from station.log: "HEF loaded" followed by
"frame recorder -> ..."), how many frames, which are readable.

    python inventory_v9.py      -> v9/raw_15_10/SESSIONS.json
"""
import json
from pathlib import Path

import cv2

RAW = Path(r"C:\Github\drone_training\v9\raw_15_10")
HEF = {"bestty_yolo26_1002_1255_zoo_fast.hef": "ft_v2", "bestty_yolo26_1006_2128_zoo_fast.hef": "ft_v5",
       "bestty_yolo26_1013_0852_zoo_fast.hef": "ft_v7", "bestty_yolo26_1014_1007_zoo_fast.hef": "ft_v8"}
# station.log (lines 5589-9278 on the Pi, read 16.10.2026): session -> (HEF, detector threshold, note)
RUNS = {
    "frames_20260920_222303": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.30, "14.10 owner test, frames every 10 s"),
    "frames_20260920_223657": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, "14.10 owner test, frames every 10 s, killed"),
    "frames_20260920_234331": ("bestty_yolo26_1002_1255_zoo_fast.hef", 0.30, "start attempt"),
    "frames_20260920_234410": ("bestty_yolo26_1002_1255_zoo_fast.hef", 0.30, "start attempt"),
    "frames_20260920_234718": ("bestty_yolo26_1002_1255_zoo_fast.hef", 0.30, ""),
    "frames_20260921_000531": ("bestty_yolo26_1002_1255_zoo_fast.hef", 0.40, ""),
    "frames_20260921_001018": ("bestty_yolo26_1006_2128_zoo_fast.hef", 0.30, ""),
    "frames_20260921_001621": ("bestty_yolo26_1006_2128_zoo_fast.hef", 0.40, ""),
    "frames_20260921_002313": ("bestty_yolo26_1013_0852_zoo_fast.hef", 0.30, ""),
    "frames_20260921_002909": ("bestty_yolo26_1013_0852_zoo_fast.hef", 0.40, ""),
    "frames_20260921_003430": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.30, "ended by a Pi reboot"),
    "frames_20260921_020505": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, ""),
    "frames_20260921_021834": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, ""),
    "frames_20260921_022253": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, ""),
    "frames_20260921_024027": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, "my WIDE-view check, static drone"),
    "frames_20260921_024159": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, "my A/B speed check, static drone"),
    "frames_20260921_024422": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, "owner, WIDE operator view"),
    "frames_20260921_024514": ("bestty_yolo26_1014_1007_zoo_fast.hef", 0.40, "owner, WIDE operator view"),
}


def main():
    out = {}
    for s, (hef, thr, note) in RUNS.items():
        d = RAW / s
        rows, bad_lines = [], 0
        for line in (d / "frames.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad_lines += 1
        ok, bad = [], []
        for r in rows:
            p, n = d / r["files"]["pub"], d / r["files"]["nat"]
            a, b = cv2.imread(str(p)), cv2.imread(str(n))
            if a is None or b is None or a.shape[:2] != (480, 640) or b.shape[:2] != (990, 1320):
                bad.append(r["files"]["pub"])
            else:
                ok.append(r["files"]["pub"])
        reasons = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        out[s] = {"model": HEF[hef], "hef": hef, "threshold": thr, "note": note, "frames": len(rows),
                  "readable": len(ok), "unreadable": bad, "broken_index_lines": bad_lines, "reasons": reasons,
                  "first": rows[0]["files"]["pub"][:15] if rows else None,
                  "last": rows[-1]["files"]["pub"][:15] if rows else None}
        print(f"{s}  {HEF[hef]}@{thr:.2f}  frames {len(rows):4d} readable {len(ok):4d} bad {len(bad)} "
              f"broken lines {bad_lines}  {reasons}  {note}")
    (RAW / "SESSIONS.json").write_text(json.dumps(out, indent=1))
    print("total readable", sum(v["readable"] for v in out.values()))


if __name__ == "__main__":
    main()

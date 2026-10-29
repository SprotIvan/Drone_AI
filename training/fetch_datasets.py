#!/usr/bin/env python3
"""
fetch_datasets.py — download the public drone datasets used to fine-tune the
station's YOLO26n model, in YOLO (txt) format, from Roboflow Universe.

    ROBOFLOW_API_KEY=... python fetch_datasets.py <out_dir> [--include-nc]

The API key is read from the environment ONLY. It is never written to disk,
to the dataset, to the notebook or to git.

Every source, its version and its licence are listed in SOURCES below and are
copied into the final dataset's SOURCES.md by build_dataset.py — CC BY 4.0
requires attribution.

⚠️ NON-COMMERCIAL SOURCES are skipped unless --include-nc is given. Use that
flag only if the station and its model are used non-commercially.
"""
import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "https://api.roboflow.com"

# slug, version, licence, class map, note. The class map gives EVERY class
# name of the source (or "*") a role:
#   "drone"       labelled as the one class `drone`
#   "background"  box removed, the image may stay as a negative
#   "drop"        the WHOLE image is excluded (a class that is not clean)
#   "?"           not inspected yet: build_dataset.py refuses to run
# A class name missing from the map also makes build_dataset.py refuse: a
# silent default would have made all 8 132 anti-drone boxes "background"
# (its classes turned out to be named Fixed-wing/Multicopter/VTOL).
SOURCES = [
    ("drone-detection-project/drone-vs-bird-detection", 3, "CC BY 4.0",
     {"Drones": "drone", "Birds": "background"}, "drones + birds (hard negatives)"),
    ("paresh-makwana/drone-detect-suvzw", 6, "Public Domain",
     {"*": "drone"}, "Mavic, Phantom, FPV, cinewhoop, fixed wing, VTOL ..."),
    ("yogith-nams8/anti-uav-s8wri", 1, "CC BY 4.0",
     {"UAV": "drone"}, "Anti-UAV; train augmented x3 -> one copy kept"),
    ("get/uav-detect-pfiqs", 1, "CC BY 4.0",
     {"UAV": "drone"}, "train augmented x3 -> one copy kept"),
    # ⚠️ "Fixed-wing" mixes fixed-wing UAVs with an airliner and aerobatic
    # jets (checked on a contact sheet): labelled "drone" it would teach
    # that aircraft are drones. Its images are dropped whole.
    ("drone-detection-fn0bd/anti-drone", 7, "CC BY 4.0",
     {"Multicopter": "drone", "VTOL": "drone", "Fixed-wing": "drop"},
     "multicopters + VTOL UAVs; Fixed-wing images dropped (mixed with "
     "manned aircraft)"),
    # 4 = DJI Matrice/Agras, 5 = Mavic, 6 = Phantom (contact sheets).
    ("crypt2/dji-drones", 1, "CC BY 4.0",
     {"4": "drone", "5": "drone", "6": "drone"},
     "DJI Matrice/Agras, Mavic, Phantom; cut-outs only (see EXCLUDED in "
     "build_dataset.py)"),
    ("uav-detection/drone-yolov5-b4787", 2, "CC BY 4.0",
     {"drone": "drone"}, ""),
    ("passants-workspace/dut-anti-uav-detection", 2, "CC BY 4.0",
     {"drone": "drone"}, "DUT Anti-UAV subset, small drones"),
    ("zhejiang-university-china-dliq1/drones-detection-with-yolov8", 1,
     "Public Domain", {"drone": "drone"}, "train augmented x3 -> one copy kept"),
    ("drone-a7lpy/drones-yolo11-a", 17, "CC BY-NC-SA 4.0",
     {"drone": "drone", "bird": "background", "airplane": "background",
      "0": "?", "object": "drop"}, "NON-COMMERCIAL"),
]


def export_link(slug, version, key, fmt="yolov8", tries=60):
    """The signed download link of a version export, waiting while Roboflow
    generates it."""
    url = f"{API}/{slug}/{version}/{fmt}"
    for _ in range(tries):
        r = requests.get(url, params={"api_key": key}, timeout=60)
        r.raise_for_status()
        d = r.json()
        link = (d.get("export") or {}).get("link")
        if link:
            return link
        time.sleep(5)                      # export still being generated
    raise RuntimeError(f"{slug} v{version}: export not ready")


def download(link, dest, label, report_every=100e6):
    """Download to `dest`, RESUMING a partial file with an HTTP Range request.
    Returns the final size in bytes."""
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(link, stream=True, timeout=120, headers=headers) as r:
        if r.status_code == 416:                  # nothing left to fetch
            return done
        r.raise_for_status()
        if done and r.status_code != 206:         # Range ignored: restart
            done = 0
        mode = "ab" if done else "wb"
        next_report = done + report_every
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"      {label}: {done / 1e6:.0f} MB", flush=True)
                    next_report += report_every
    return done


def fetch_one(source, out, key, attempts=12):
    slug, version, licence, _cmap, note = source
    name = f"{slug.replace('/', '__')}__v{version}"
    target = out / name
    if (target / "data.yaml").exists():
        return f"have  {name}"
    zpath = out / f"{name}.zip"
    t0 = time.time()
    for attempt in range(1, attempts + 1):
        try:
            # A fresh link each attempt: signed links expire.
            size = download(export_link(slug, version, key), zpath, name)
            with zipfile.ZipFile(zpath) as z:
                bad = z.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"corrupt member {bad}")
                z.extractall(target)
            zpath.unlink()
            n = (sum(1 for _ in target.rglob("*.jpg"))
                 + sum(1 for _ in target.rglob("*.png")))
            return (f"got   {name}: {size / 1e6:.0f} MB, {n} images, "
                    f"{time.time() - t0:.0f} s  [{licence}] {note}")
        except zipfile.BadZipFile as exc:
            print(f"      {name}: bad zip ({exc}) — downloading again",
                  flush=True)
            zpath.unlink(missing_ok=True)
        except Exception as exc:
            print(f"      {name}: attempt {attempt} failed ({exc}) — "
                  f"resuming in 10 s", flush=True)
            time.sleep(10)
    return f"FAILED {name} after {attempts} attempts"


def main():
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--include-nc", action="store_true",
                    help="also download NON-COMMERCIAL sources")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY in the environment")

    out = Path(args.out_dir) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for source in SOURCES:
        if "NC" in source[2] and not args.include_nc:
            print(f"SKIP (non-commercial licence {source[2]}): {source[0]}")
        else:
            todo.append(source)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(lambda s: fetch_one(s, out, key), todo):
            print(line, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

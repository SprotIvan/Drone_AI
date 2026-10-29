#!/usr/bin/env python3
"""
station_frames.py — turn the frames the station recorded (frame_recorder.py,
logs/frames_<start>/) into training data for <work_dir>/station/:
backgrounds, and drone frames labelled from the detector's own box.

    python station_frames.py sheets <frames_dir>... --out <review_dir> [--every S]
        [--between A B] [--propose model.pt [--propose-conf 0.10]]
    python station_frames.py folders <review_dir> --out <sorted_dir>
    python station_frames.py export <review_dir> --to <work_dir>/station \
        [--folders <sorted_dir>]
    python station_frames.py thin <work_dir>/station --model model.pt \
        --out <dir for the easy backgrounds> [--conf 0.40]

sheets   --between A B  only frames A..B seconds after each folder's first
                        record: to keep part of a session out of training,
                        as a test the model has never seen
         --propose      the recorder keeps only the detector's boxes >= its
                        threshold, so a drone it scored lower has no box to
                        label. The model (a .pt, run on the published frame
                        and on the native crop-mode windows, where a small
                        drone is ~2x larger) adds up to 4 boxes per frame
                        from --propose-conf up,
                        not overlapping a recorded one, drawn in BLUE with
                        "p". They are numbered after the recorded boxes;
                        rerun sheets with the SAME options, or the numbers
                        in labels.json point at other boxes.
         <review_dir>/frames.json   every recorded frame: id, files, boxes
         <review_dir>/sheet_NNN.jpg 12 frames each: the published frame with
                                    its boxes numbered, each box zoomed from
                                    the native frame
         <review_dir>/labels.json   one entry per id, null until reviewed:
             "bg"         NO drone ANYWHERE in the frame. Not "the box is
                          wrong": a drone in someone's hand elsewhere in the
                          frame makes it a drone frame, and as a background
                          it would teach that a drone is background.
             "drone:0,2"  boxes 0 and 2 are on a drone, and there is no
                          other drone in the frame
             "skip"       anything else — a drone without a box, can't tell
         A rerun keeps the labels already given.
folders  <sorted_dir>/drone, background, skip: a copy of every published
         frame, in the folder of its label, the drone box drawn green and
         every other box red — for a person to check in a file browser.
         Moving a file to another folder changes its label (see export).
export   bg     -> the published frame + its native windows, no .txt
         drone  -> the published frame with a YOLO .txt from those boxes;
                   a window holding a drone box whole gets one too, a window
                   no drone box touches goes in as a background, a window
                   cutting a drone box is left out
         null / skip -> nothing. Files are named <frames_dir>_<stamp>_pub.jpg
         and ..._tile<i>.jpg, as the station's existing backgrounds.
         --folders: the folder a frame's file is in now overrides labels.json
         (background -> bg, skip -> skip; a frame moved INTO drone has no
         drone box to label, so it becomes skip and is reported). What an
         earlier export wrote for these frames is removed first, so a rerun
         after files were moved leaves no stale copies.
thin     moves the station backgrounds a model finds NOTHING on (no box >=
         --conf) out of the station folder: they teach the model nothing it
         does not already do, and only shift the balance towards "there is
         never a drone here" (ft_v4, 06.10.2026: 21 % backgrounds, -8 points
         of recall on val; ft_v3 with 11 %: -2). Only frames_* images this
         tool exported, never one with a .txt. Moved, not deleted.

The windows are smart_crop.tile_origins() of the native frame — the 640x640
windows the station's crop mode infers (6 on the 1320x990 stream). The
detector's box on a small drone can be off by a few pixels: a "drone" label
says the box is on the drone, so a box that is visibly off is "skip".
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "acoustic_radar"))
from smart_crop import tile_origins  # noqa: E402

CROP = 640
CELL_W, CELL_H, ZOOM = 400, 300, 150      # one frame on a sheet
COLS, ROWS = 3, 4
MAX_PROPOSALS = 4


def read_records(frames_dir):
    """frames.jsonl, minus a last line cut short when the station stopped."""
    recs = []
    for line in open(frames_dir / "frames.jsonl", encoding="utf-8"):
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs


def zoom(nat, pub_w, box):
    k = nat.shape[1] / pub_w
    x, y, w, h = (v * k for v in box[:4])
    side = int(min(max(64, 4 * max(w, h)), *nat.shape[:2]))
    x0 = int(min(max(0, x + w / 2 - side / 2), nat.shape[1] - side))
    y0 = int(min(max(0, y + h / 2 - side / 2), nat.shape[0] - side))
    z = nat[y0:y0 + side, x0:x0 + side].copy()
    cv2.rectangle(z, (int(x - x0) - 2, int(y - y0) - 2),
                  (int(x + w - x0) + 2, int(y + h - y0) + 2), (0, 0, 255), 1)
    return cv2.resize(z, (ZOOM, ZOOM), interpolation=cv2.INTER_NEAREST)


def cell(rec, d):
    pub = cv2.imread(str(d / rec["files"]["pub"]))
    nat = (cv2.imread(str(d / rec["files"]["nat"]))
           if "nat" in rec["files"] else None)
    pw, ph = rec["pub_size"]
    sx, sy = CELL_W / pw, CELL_H / ph
    view = cv2.resize(pub, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    n_rec = len(rec["boxes"]) - rec.get("proposed", 0)
    for i, (x, y, w, h, s) in enumerate(rec["boxes"]):
        colour = (0, 0, 255) if i < n_rec else (255, 160, 0)
        tag = f"{i}:{s:.2f}" if i < n_rec else f"{i}:p{s:.2f}"
        p0 = (int(x * sx) - 3, int(y * sy) - 3)
        cv2.rectangle(view, p0, (int((x + w) * sx) + 3, int((y + h) * sy) + 3),
                      colour, 1)
        cv2.putText(view, tag, (p0[0], max(10, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(view, f"#{rec['id']} {rec['reason'][0]}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    side = np.zeros((CELL_H, 2 * ZOOM, 3), np.uint8)   # 2 x 2 zoomed boxes
    src = nat if nat is not None else pub
    for i, b in enumerate(rec["boxes"][:4]):
        r, c = divmod(i, 2)
        side[r * ZOOM:(r + 1) * ZOOM, c * ZOOM:(c + 1) * ZOOM] = zoom(src, pw, b)
        cv2.putText(side, str(i), (c * ZOOM + 3, r * ZOOM + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    out = np.hstack([view, side])
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1),
                  (90, 90, 90), 1)
    return out


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / union if union > 0 else 0.0


def propose(frames, model_path, conf):
    """Append the model's own boxes the recorder did not keep (see --propose):
    from the published frame AND from the native 640x640 windows the crop
    mode infers, where a small drone is twice as large (mapped back to
    published px)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    for f in frames:
        d = Path(f["dir"])
        imgs, offsets = [cv2.imread(str(d / f["files"]["pub"]))], [(0, 0, 1.0)]
        if "nat" in f["files"]:
            nat = cv2.imread(str(d / f["files"]["nat"]))
            k = nat.shape[1] / f["pub_size"][0]
            for ox, oy in tile_origins((nat.shape[1], nat.shape[0]), CROP):
                imgs.append(nat[oy:oy + CROP, ox:ox + CROP])
                offsets.append((ox, oy, k))
        found = []
        for r, (ox, oy, k) in zip(model.predict(imgs, imgsz=640, conf=conf, verbose=False),
                                  offsets):
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                found.append([round((ox + x1) / k, 1), round((oy + y1) / k, 1),
                              round((x2 - x1) / k, 1), round((y2 - y1) / k, 1),
                              round(s, 3)])
        extra = []
        for b in sorted(found, key=lambda b: -b[4]):
            if all(iou(b, o) < 0.3 for o in f["boxes"] + extra):
                extra.append(b)
        extra = extra[:MAX_PROPOSALS]
        f["boxes"] = f["boxes"] + extra
        f["proposed"] = len(extra)


def cmd_sheets(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, unreadable = [], []
    for d in map(Path, args.frames_dirs):
        last, t0 = None, None
        for rec in read_records(d):
            t0 = rec["t"] if t0 is None else t0
            if args.between and not args.between[0] <= rec["t"] - t0 < args.between[1]:
                continue
            # A box on almost every frame makes the recorder save one per
            # min_gap_s: consecutive frames are near-copies of each other.
            if args.every and last is not None and rec["t"] - last < args.every:
                continue
            # A station stopped without a clean shutdown (power cut) leaves
            # its last frames as 0-byte files: they are not offered for review.
            if any(cv2.imread(str(d / f)) is None for f in rec["files"].values()):
                unreadable.append(f"{d.name}/{rec['files']['pub']}")
                continue
            last = rec["t"]
            frames.append({"id": len(frames), "dir": str(d.resolve()), **rec})
    if unreadable:
        print(f"{len(unreadable)} frames left out, an image unreadable: "
              + ", ".join(unreadable))
    if args.propose:
        propose(frames, args.propose, args.propose_conf)
        print(f"{sum(f['proposed'] for f in frames)} boxes proposed by "
              f"{Path(args.propose).name} from {args.propose_conf}")
    (out / "frames.json").write_text(json.dumps(frames, indent=0))
    lab_path = out / "labels.json"
    old = json.loads(lab_path.read_text()) if lab_path.exists() else {}
    labels = {str(f["id"]): old.get(str(f["id"])) for f in frames}
    lab_path.write_text(json.dumps(labels, indent=0))
    per = COLS * ROWS
    blank = np.zeros((CELL_H, CELL_W + 2 * ZOOM, 3), np.uint8)
    for k in range(0, len(frames), per):
        cells = [cell(f, Path(f["dir"])) for f in frames[k:k + per]]
        cells += [blank] * (per - len(cells))
        rows = [np.hstack(cells[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        cv2.imwrite(str(out / f"sheet_{k // per:03d}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(frames)} frames, {-(-len(frames) // per)} sheets -> {out}; "
          f"{sum(v is not None for v in labels.values())} already labelled")


FOLDERS = ("drone", "background", "skip")


def folder_of(lab):
    if lab is None or lab == "skip":
        return "skip"
    return "background" if lab == "bg" else "drone"


def drone_boxes(lab):
    return ({int(i) for i in lab[6:].split(",")}
            if lab and lab.startswith("drone:") else set())


def cmd_folders(args):
    rev, out = Path(args.review_dir), Path(args.out)
    if any(out.glob("*/*.jpg")):
        sys.exit(f"{out} already holds frames — files moved there would be "
                 "lost. Export from it, or delete it to rebuild.")
    frames = json.loads((rev / "frames.json").read_text())
    labels = json.loads((rev / "labels.json").read_text())
    for name in FOLDERS:
        (out / name).mkdir(parents=True, exist_ok=True)
    for f in frames:
        lab = labels.get(str(f["id"]))
        drone = drone_boxes(lab)
        img = cv2.imread(str(Path(f["dir"]) / f["files"]["pub"]))
        for i, (x, y, w, h, s) in enumerate(f["boxes"]):
            colour = (0, 220, 0) if i in drone else (0, 0, 255)
            cv2.rectangle(img, (int(x) - 2, int(y) - 2),
                          (int(x + w) + 2, int(y + h) + 2), colour, 1)
            cv2.putText(img, f"{s:.2f}", (int(x), max(10, int(y) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        name = f"{f['id']:04d}_{Path(f['dir']).name}_{f['files']['pub']}"
        cv2.imwrite(str(out / folder_of(lab) / name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print({n: len(list((out / n).glob("*.jpg"))) for n in FOLDERS})


def labels_from_folders(labels, sorted_dir):
    """The folder each frame's file is in now; a file deleted = skip."""
    where = {}
    for name in FOLDERS:
        for p in (Path(sorted_dir) / name).glob("*.jpg"):
            where[str(int(p.name.split("_", 1)[0]))] = name
    out, into_drone = {}, []
    for fid, lab in labels.items():
        name = where.get(fid, "skip")
        if name == "drone":
            if not drone_boxes(lab):
                into_drone.append(fid)
                name = "skip"
            out[fid] = lab if name == "drone" else "skip"
        else:
            out[fid] = "bg" if name == "background" else "skip"
    if into_drone:
        print(f"moved into drone/ but no drone box to label, left out: "
              f"#{', #'.join(into_drone)}")
    changed = sum(out[k] != labels[k] for k in labels)
    print(f"{changed} label(s) changed by the folders")
    return out


def yolo(box, w, h):
    x, y, bw, bh = box[:4]
    return f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def cmd_export(args):
    rev, to = Path(args.review_dir), Path(args.to)
    to.mkdir(parents=True, exist_ok=True)
    frames = {str(f["id"]): f for f in json.loads((rev / "frames.json").read_text())}
    labels = json.loads((rev / "labels.json").read_text())
    if args.folders:
        labels = labels_from_folders(labels, args.folders)
    stale = 0
    for d in {Path(f["dir"]).name for f in frames.values()}:
        for p in to.glob(f"{d}_*"):           # an earlier export of these frames
            p.unlink()
            stale += 1
    if stale:
        print(f"{stale} files of an earlier export of these frames removed")
    n = {"bg": 0, "drone": 0, "skip": 0, "unlabelled": 0,
         "files": 0, "windows_left_out": 0}
    for fid, lab in labels.items():
        if lab is None:
            n["unlabelled"] += 1
            continue
        if lab == "skip":
            n["skip"] += 1
            continue
        f = frames[fid]
        d = Path(f["dir"])
        if lab == "bg":
            drones = []
        elif lab.startswith("drone:"):
            drones = [f["boxes"][int(i)] for i in lab[6:].split(",")]
        else:
            sys.exit(f"#{fid}: unknown label {lab!r}")
        n["bg" if not drones else "drone"] += 1
        stem = f"{d.name}_{f['files']['pub'][:-len('_pub.jpg')]}"
        pw, ph = f["pub_size"]
        shutil.copy2(d / f["files"]["pub"], to / f"{stem}_pub.jpg")
        n["files"] += 1
        if drones:
            (to / f"{stem}_pub.txt").write_text(
                "\n".join(yolo(b, pw, ph) for b in drones) + "\n")
        if "nat" not in f["files"]:
            continue
        nat = cv2.imread(str(d / f["files"]["nat"]))
        k = nat.shape[1] / pw
        nboxes = [[v * k for v in b[:4]] for b in drones]
        for i, (ox, oy) in enumerate(tile_origins((nat.shape[1], nat.shape[0]), CROP)):
            inside, touched = [], False
            for x, y, w, h in nboxes:
                if x >= ox and y >= oy and x + w <= ox + CROP and y + h <= oy + CROP:
                    inside.append([x - ox, y - oy, w, h])
                elif x < ox + CROP and x + w > ox and y < oy + CROP and y + h > oy:
                    touched = True
            if touched:
                n["windows_left_out"] += 1
                continue
            cv2.imwrite(str(to / f"{stem}_tile{i}.jpg"),
                        nat[oy:oy + CROP, ox:ox + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95])
            n["files"] += 1
            if inside:
                (to / f"{stem}_tile{i}.txt").write_text(
                    "\n".join(yolo(b, CROP, CROP) for b in inside) + "\n")
    print(json.dumps(n))


def cmd_thin(args):
    from ultralytics import YOLO
    st, out = Path(args.station_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in sorted(st.glob("frames_*.jpg"))
            if not p.with_suffix(".txt").exists()]
    model = YOLO(args.model)
    easy = []
    for k in range(0, len(imgs), 16):
        batch = imgs[k:k + 16]
        for p, r in zip(batch, model.predict([str(p) for p in batch], imgsz=640,
                                             conf=args.conf, verbose=False)):
            if len(r.boxes) == 0:
                easy.append(p)
    for p in easy:
        shutil.move(str(p), str(out / p.name))
    print(f"{len(imgs)} station backgrounds: {len(easy)} with no box >= "
          f"{args.conf} moved to {out}, {len(imgs) - len(easy)} kept")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("frames_dirs", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--every", type=float, default=0,
                   help="keep at most one frame per this many seconds")
    s.add_argument("--between", type=float, nargs=2, metavar=("A", "B"))
    s.add_argument("--propose")
    s.add_argument("--propose-conf", type=float, default=0.10)
    th = sub.add_parser("thin")
    th.add_argument("station_dir")
    th.add_argument("--model", required=True)
    th.add_argument("--out", required=True)
    th.add_argument("--conf", type=float, default=0.40)
    fo = sub.add_parser("folders")
    fo.add_argument("review_dir")
    fo.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("review_dir")
    e.add_argument("--to", required=True)
    e.add_argument("--folders")
    args = ap.parse_args()
    {"sheets": cmd_sheets, "folders": cmd_folders,
     "export": cmd_export, "thin": cmd_thin}[args.cmd](args)


if __name__ == "__main__":
    main()

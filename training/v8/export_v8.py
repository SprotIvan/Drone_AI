#!/usr/bin/env python3
"""
export_v8.py — best.pt -> best.onnx exactly as train_colab.ipynb exports it
for the Hailo compile (opset 11, simplify, static 640), and a check that the
ONNX, decoded THE STATION'S WAY, gives the same boxes as the .pt.

    python export_v8.py BEST.pt OUT_DIR [--n 300]

1. export (Ultralytics, same arguments as make_notebook.py)
2. the six one2one head convolutions the DFC parse ends at must exist
   (/model.23/one2one_cv2.{0,1,2}/.../Conv and one2one_cv3.*)
3. that cut graph is run with onnxruntime on letterboxed RGB images and
   decoded as TWO_CAMERAS_FIXED.HailoInference does (LTRB x stride, sigmoid
   on the class logits, cv2.dnn.NMSBoxes iou 0.45), and compared with the .pt
   (Ultralytics predict, conf 0.25 + NMS 0.45): per image, boxes >= 0.30 of
   each side matched at IoU >= 0.5, score difference of the matches.
   This checks the float graph and the decode, NOT the quantization — that
   can only be measured with the HEF on the Hailo.
Writes OUT_DIR/best.onnx, best.onnx.md5, onnx_check.json.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SIZE = 640


def letterbox(img):
    sh, sw = img.shape[:2]
    r = min(SIZE / sh, SIZE / sw)
    nw, nh = int(round(sw * r)), int(round(sh * r))
    left, top = int(round((SIZE - nw) / 2.0 - 0.1)), int(round((SIZE - nh) / 2.0 - 0.1))
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img if (nw, nh) == (sw, sh) else cv2.resize(img, (nw, nh),
                                                                                     interpolation=cv2.INTER_LINEAR)
    return out, r, float(left), float(top)


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def station_decode(outs, names, r, pw, ph, w, h, conf=0.25, iou_t=0.45):
    """outs: dict name -> (1, C, H, W). Box + class conv per stride."""
    boxes, scores = [], []
    for stride, (bn, cn) in names.items():
        bx = outs[bn][0].transpose(1, 2, 0)          # H W 4
        cl = outs[cn][0].transpose(1, 2, 0)[:, :, 0]  # H W
        cl = 1.0 / (1.0 + np.exp(-np.clip(cl, -88, 88)))
        gy, gx = np.where(cl > conf)
        if not len(gy):
            continue
        reg = bx[gy, gx]
        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
        x1 = np.clip((cx - reg[:, 0] * stride - pw) / r, 0, w)
        y1 = np.clip((cy - reg[:, 1] * stride - ph) / r, 0, h)
        x2 = np.clip((cx + reg[:, 2] * stride - pw) / r, 0, w)
        y2 = np.clip((cy + reg[:, 3] * stride - ph) / r, 0, h)
        for a, b, c, d, s in zip(x1, y1, x2, y2, cl[gy, gx]):
            if c - a > 1 and d - b > 1:
                boxes.append([float(a), float(b), float(c - a), float(d - b)])
                scores.append(float(s))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, iou_t)
    return [[boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], scores[i]]
            for i in np.array(keep).flatten()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(YOLO(a.pt).export(format="onnx", imgsz=640, opset=11, simplify=True, dynamic=False,
                                     device=a.device))
    onnx_path = out / "best.onnx"
    if src.resolve() != onnx_path.resolve():
        shutil.copy2(src, onnx_path)
    md5 = hashlib.md5(onnx_path.read_bytes()).hexdigest()
    (out / "best.onnx.md5").write_text(md5 + "\n")
    m = onnx.load(str(onnx_path))
    convs = {n.name: n.output[0] for n in m.graph.node if n.op_type == "Conv" and "one2one" in n.name}
    names = {}
    for i, stride in enumerate((8, 16, 32)):
        bn = f"/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv"
        cn = f"/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv"
        assert bn in convs and cn in convs, f"head conv missing: {bn} / {cn} - the DFC parse would fail"
        names[stride] = (convs[bn], convs[cn])
    cut = out / "best_cut_heads.onnx"
    onnx.utils.extract_model(str(onnx_path), str(cut), ["images"], [t for p in names.values() for t in p])
    sess = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])
    pt = YOLO(a.pt)
    imgs = []
    for d in (Path(r"C:\Github\drone_training\v8\dataset_v8\images\val"),
              Path(r"C:\Github\drone_training\v8\testsets\drone_close\images"),
              Path(r"C:\Github\drone_training\v8\testsets\coco_neg\images")):
        imgs += sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    st = sorted(Path(r"C:\Github\drone_training\v8\testsets\station").rglob("*_pub.jpg"))
    rng = random.Random(0)
    pick = rng.sample(imgs, min(a.n, len(imgs))) + rng.sample(st, min(a.n // 3, len(st)))
    stats = {"images": 0, "pt_boxes": 0, "onnx_boxes": 0, "matched": 0, "max_score_diff": 0.0,
             "mean_score_diff": 0.0, "images_identical_set": 0}
    diffs = []
    for p in pick:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        lb, r, pw, ph = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        res = sess.run(None, {"images": inp})
        outs = dict(zip([o.name for o in sess.get_outputs()], res))
        ob = [b for b in station_decode(outs, names, r, pw, ph, w, h) if b[4] >= 0.30]
        lbb = cv2.cvtColor(lb, cv2.COLOR_RGB2BGR)
        rr = pt.predict(lbb, imgsz=640, conf=0.25, verbose=False, device=a.device)[0]
        pb = [[(x1 - pw) / r, (y1 - ph) / r, (x2 - pw) / r, (y2 - ph) / r, float(s)]
              for (x1, y1, x2, y2), s in zip(rr.boxes.xyxy.cpu().numpy(), rr.boxes.conf.cpu().numpy())
              if s >= 0.30]
        used, mt = set(), 0
        for b in pb:
            best, bi = 0.0, -1
            for j, c in enumerate(ob):
                if j not in used and iou(b, c) > best:
                    best, bi = iou(b, c), j
            if best >= 0.5:
                used.add(bi)
                mt += 1
                diffs.append(abs(b[4] - ob[bi][4]))
        stats["images"] += 1
        stats["pt_boxes"] += len(pb)
        stats["onnx_boxes"] += len(ob)
        stats["matched"] += mt
        stats["images_identical_set"] += (mt == len(pb) == len(ob))
    if diffs:
        stats["max_score_diff"] = round(float(np.max(diffs)), 4)
        stats["mean_score_diff"] = round(float(np.mean(diffs)), 4)
    stats["onnx_md5"] = md5
    stats["note"] = "float graph + station decode vs .pt; quantization NOT covered"
    (out / "onnx_check.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

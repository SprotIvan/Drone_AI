#!/usr/bin/env python3
"""
colab_export_verify.py — stages 1-3 of the ft_v8 Colab notebook (runs as its
own process, so Ultralytics' select_device("cpu") cannot leave
CUDA_VISIBLE_DEVICES="" behind for the Hailo compile that follows).

    python colab_export_verify.py --pt best.pt --expected samples_expected.json \
        --samples DIR --out DIR

1  .pt:   md5 = the one in samples_expected.json; the checkpoint loads; its
          training record (start weights, data, epochs) is printed; predictions
          on the sample images (letterboxed 640, grey 114 — as the station) match
          the ones computed on the laptop (same boxes, score within 0.02).
2  ONNX:  Ultralytics export exactly as train_colab.ipynb / export_v8.py
          (opset 11, simplify, static 640x640) -> OUT/best.onnx (+ .md5).
3  check: onnx.checker; the six one2one head convolutions the DFC parse ends
          at exist; the graph cut at them, run with onnxruntime and decoded the
          way the station decodes the HEF (LTRB x stride, sigmoid, NMS 0.45),
          gives the .pt's boxes on the samples. Writes OUT/onnx_check.json.
Exit code 1 on any failed check — the notebook stops before the compile.
This proves the float ONNX; what quantization does can only be measured with
the HEF on the Hailo.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 640
END_NODES = [f"/model.23/one2one_cv{k}.{i}/one2one_cv{k}.{i}.2/Conv" for i in range(3) for k in (2, 3)]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


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


def match(pa, pb):
    """greedy IoU >= 0.5 matching -> (matched, score diffs)"""
    used, diffs = set(), []
    for a in sorted(pa, key=lambda x: -x[4]):
        best, bi = 0.0, -1
        for j, b in enumerate(pb):
            if j not in used and iou(a, b) > best:
                best, bi = iou(a, b), j
        if best >= 0.5:
            used.add(bi)
            diffs.append(abs(a[4] - pb[bi][4]))
    return len(diffs), diffs


def station_nms(boxes, conf=0.25, iou_t=0.45):
    """the station's cv2.dnn.NMSBoxes on [x1,y1,x2,y2,s] boxes (the .pt's end-to-end head keeps
    overlapping boxes the station would merge)"""
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], [b[4] for b in boxes], conf, iou_t)
    return [boxes[i] for i in np.array(keep).flatten()]


def graph_md5(mdl):
    """md5 of the graph's weights and nodes - NOT of the file: Ultralytics stamps the export date into
    the ONNX metadata, so two exports of one model never have the same file md5"""
    from onnx import numpy_helper
    h = hashlib.md5()
    for t in sorted(mdl.graph.initializer, key=lambda t: t.name):
        h.update(t.name.encode())
        h.update(numpy_helper.to_array(t).tobytes())
    for n in mdl.graph.node:
        h.update((n.op_type + '|' + ','.join(n.input) + '|' + ','.join(n.output)).encode())
    return h.hexdigest()


def pt_boxes(model, img):
    lb, r, pw, ph = letterbox(img)
    res = model.predict(lb, imgsz=SIZE, conf=0.25, verbose=False, device="cpu")[0]
    return [[(x1 - pw) / r, (y1 - ph) / r, (x2 - pw) / r, (y2 - ph) / r, float(s)]
            for (x1, y1, x2, y2), s in zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy())]


def onnx_boxes(sess, names, img, conf=0.25):
    h, w = img.shape[:2]
    lb, r, pw, ph = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    outs = dict(zip([o.name for o in sess.get_outputs()],
                    sess.run(None, {"images": lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0})))
    boxes, scores = [], []
    for stride, (bn, cn) in names.items():
        bx = outs[bn][0].transpose(1, 2, 0)
        cl = 1.0 / (1.0 + np.exp(-np.clip(outs[cn][0][0], -88, 88)))
        gy, gx = np.where(cl > conf)
        for y, x in zip(gy, gx):
            l, t, rr, b = bx[y, x]
            cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
            x1 = float(np.clip((cx - l * stride - pw) / r, 0, w))
            y1 = float(np.clip((cy - t * stride - ph) / r, 0, h))
            x2 = float(np.clip((cx + rr * stride - pw) / r, 0, w))
            y2 = float(np.clip((cy + b * stride - ph) / r, 0, h))
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(cl[y, x]))
    if not boxes:
        return []
    keep = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)).flatten()
    return [[boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], scores[i]] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = json.loads(Path(a.expected).read_text())
    report, ok = {"stage1": {}, "stage2": {}, "stage3": {}}, True

    # ── stage 1: the .pt ──
    m5 = md5(a.pt)
    report["stage1"]["pt_md5"] = m5
    if m5 != exp["pt_md5"]:
        print(f"STAGE 1 FAIL: best.pt md5 {m5} != {exp['pt_md5']} - not the model this package was made for")
        sys.exit(1)
    ck = torch.load(a.pt, map_location="cpu", weights_only=False)
    ta = ck.get("train_args", {})
    rec = ck.get("train_results") or {}
    print(f"STAGE 1: best.pt md5 OK ({m5}); Ultralytics {ck.get('version')}, saved {str(ck.get('date'))[:19]}; "
          f"trained from {Path(str(ta.get('model'))).name} on {Path(str(ta.get('data'))).parent.name}, "
          f"{ta.get('epochs')} epochs, record ends at epoch {int(rec['epoch'][-1]) if rec.get('epoch') else '?'}")
    model = YOLO(a.pt)
    s1 = []
    for name, s in exp["samples"].items():
        want = s["boxes"]
        img = cv2.imread(str(Path(a.samples) / name))
        got = pt_boxes(model, img)
        g30 = [b for b in got if b[4] >= 0.30]
        w30 = [b for b in want if b[4] >= 0.30]
        n, diffs = match(w30, g30)
        good = n == len(w30) == len(g30) and (not diffs or max(diffs) < 0.02)
        s1.append({"image": name, "laptop": len(w30), "colab": len(g30), "matched": n,
                   "max_score_diff": round(max(diffs), 4) if diffs else 0.0, "ok": good})
        print(f"   {name}: laptop {len(w30)} box(es) >= 0.30, here {len(g30)}, matched {n}"
              f"{'' if good else '   <-- DIFFERENT'}")
    report["stage1"]["samples"] = s1
    if not all(x["ok"] for x in s1):
        print("STAGE 1 WARNING: the .pt predicts differently here than on the laptop "
              "(another torch build may move a borderline score) - see above")
        ok = sum(x["ok"] for x in s1) >= len(s1) - 1      # one borderline image tolerated

    # ── stage 2: export ──
    src = Path(YOLO(a.pt).export(format="onnx", imgsz=SIZE, opset=11, simplify=True, dynamic=False, device="cpu"))
    onnx_path = out / "best.onnx"
    if src.resolve() != onnx_path.resolve():
        shutil.move(str(src), onnx_path)        # Ultralytics writes it next to best.pt
    om5 = md5(onnx_path)
    (out / "best.onnx.md5").write_text(om5 + "\n")
    gm5 = graph_md5(onnx.load(str(onnx_path)))
    (out / "best.onnx.graph_md5").write_text(gm5 + "\n")
    same = gm5 == exp.get("laptop_onnx_graph_md5")
    report["stage2"] = {"onnx": str(onnx_path), "bytes": onnx_path.stat().st_size, "md5": om5, "graph_md5": gm5,
                        "same_graph_and_weights_as_laptop_export": same}
    print(f"STAGE 2: exported {onnx_path} ({onnx_path.stat().st_size} B); graph+weights md5 {gm5} - "
          f"identical to the export verified on the laptop: {same}")

    # ── stage 3: check the ONNX ──
    mdl = onnx.load(str(onnx_path))
    onnx.checker.check_model(mdl)
    convs = {n.name: n.output[0] for n in mdl.graph.node if n.op_type == "Conv"}
    missing = [n for n in END_NODES if n not in convs]
    if missing:
        print("STAGE 3 FAIL: head convolutions missing (the DFC parse would fail):", missing)
        sys.exit(1)
    names = {s: (convs[f"/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv"],
                 convs[f"/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv"]) for i, s in enumerate((8, 16, 32))}
    cut = out / "best_cut_heads.onnx"
    onnx.utils.extract_model(str(onnx_path), str(cut), ["images"], [t for p in names.values() for t in p])
    sess = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])
    tot_pt = tot_on = tot_m = 0
    diffs_all = []
    for name in exp["samples"]:
        img = cv2.imread(str(Path(a.samples) / name))
        pb = [b for b in station_nms(pt_boxes(model, img)) if b[4] >= 0.30]
        ob = [b for b in onnx_boxes(sess, names, img) if b[4] >= 0.30]
        n, diffs = match(pb, ob)
        tot_pt, tot_on, tot_m = tot_pt + len(pb), tot_on + len(ob), tot_m + n
        diffs_all += diffs
    report["stage3"] = {"checker": "ok", "end_nodes": "all 6 present", "pt_boxes": tot_pt, "onnx_boxes": tot_on,
                        "matched": tot_m, "max_score_diff": round(max(diffs_all), 4) if diffs_all else 0.0}
    print(f"STAGE 3: onnx.checker OK; 6 head convolutions present; .pt {tot_pt} boxes vs ONNX (station decode) "
          f"{tot_on}, matched {tot_m}, max score diff {report['stage3']['max_score_diff']}")
    if tot_pt and (tot_m < 0.95 * tot_pt or report["stage3"]["max_score_diff"] > 0.02):
        print("STAGE 3 FAIL: the ONNX does not reproduce the .pt")
        ok = False
    report["ok"] = ok
    (out / "onnx_check.json").write_text(json.dumps(report, indent=1))
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED - do not compile")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

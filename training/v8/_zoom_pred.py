import io, json, sys, zipfile
import cv2, numpy as np
from PIL import Image
z = zipfile.ZipFile(r"C:\Users\Nazar\Downloads\drone_dataset_v5.zip")
P = {m: json.loads(open(rf"C:\Github\drone_training\v8\preds\v5ds_{m}.json").read())["preds"] for m in ("ft_v5", "ft_v7")}
out = sys.argv[1]; names = sys.argv[2:]
panels = []
for n in names:
    im = cv2.cvtColor(np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB")), cv2.COLOR_RGB2BGR)
    full = im.copy()
    best = None
    for m, col in (("ft_v5", (255, 160, 0)), ("ft_v7", (255, 0, 255))):
        for x1, y1, x2, y2, s in P[m][n]:
            if s >= 0.3:
                cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(full, f"{m[3:]} {s:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                if best is None or s > best[4]:
                    best = (x1, y1, x2, y2, s)
    h, w = im.shape[:2]
    p1 = cv2.resize(full, (640, int(640 * h / w)))
    if best:
        cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        half = max(best[2] - best[0], best[3] - best[1]) * 2 + 20
        x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))
        p2 = cv2.resize(im[y0:y1, x0:x1], (p1.shape[0], p1.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        p2 = np.zeros((p1.shape[0], p1.shape[0], 3), np.uint8)
    row = np.hstack([p1, p2])
    cv2.putText(row, n.split("/")[-1], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    panels.append(row)
wmax = max(p.shape[1] for p in panels)
panels = [np.hstack([p, np.zeros((p.shape[0], wmax - p.shape[1], 3), np.uint8)]) for p in panels]
cv2.imwrite(out, np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])

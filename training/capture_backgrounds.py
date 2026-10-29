#!/usr/bin/env python3
"""
capture_backgrounds.py — frames from BOTH station cameras, configured as the
station configures them (CameraManager._open_camera: video configuration,
640x480, "RGB888", 30 fps), saved losslessly, for a false-positive test of
a HEF on the scene the station actually watches.

    ~/mk/venv/bin/python capture_backgrounds.py OUT_DIR [--n 150] [--every 1.0]

Stop the station first: a camera can be held by one process only.

OUT_DIR/cam<i>_<k>.png  the array picamera2 returned, written with
                        cv2.imwrite unchanged — so cv2.imread gives back the
                        exact array the station hands its detector
OUT_DIR/capture.json    per camera: the mean of each array channel over the
                        top quarter of the frames (normally sky). Clear sky
                        is blue: the channel with the highest mean there is
                        the blue one, which tells the array's channel order.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--every", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cams = {}
    for info in Picamera2.global_camera_info():
        i = int(info.get("Num", len(cams)))
        cam = Picamera2(camera_num=i)
        cam.configure(cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}, buffer_count=4,
            controls={"FrameDurationLimits": (33333, 33333)}))
        cam.start()
        cams[i] = (cam, info.get("Model", "?"))
    time.sleep(2.0)                      # let exposure and white balance settle
    for cam, _ in cams.values():
        for _ in range(30):
            cam.capture_array()

    top = {i: [] for i in cams}
    t_next = time.monotonic()
    for k in range(args.n):
        for i, (cam, _) in cams.items():
            a = cam.capture_array()
            cv2.imwrite(str(out / f"cam{i}_{k:04d}.png"), a)
            top[i].append(a[: a.shape[0] // 4].reshape(-1, 3).mean(0))
        t_next += args.every
        time.sleep(max(0.0, t_next - time.monotonic()))
    for cam, _ in cams.values():
        cam.stop()
        cam.close()

    report = {}
    for i, (_, model) in cams.items():
        m = np.mean(top[i], axis=0)
        report[f"cam{i}"] = {"model": model,
                             "top_quarter_mean_by_array_channel":
                                 [round(float(v), 1) for v in m],
                             "brightest_channel_in_sky": int(np.argmax(m))}
    (out / "capture.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{args.n} frames x {len(cams)} cameras -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
frame_recorder.py — save the frames the detector sees, for training data.

VisualConfig.frame_record. Off by default: this is a data-collection tool
for test sessions, not something a deployed station needs.

What is saved, to <logging.log_dir>/frames_<start>/:

  <stamp>_<cam>_pub.jpg   the published frame — exactly what the full-frame
                          inference sees (640x480);
  <stamp>_<cam>_nat.jpg   in crop mode, the native frame of the same exposure
                          — what the 640x640 windows are cut from;
  frames.jsonl            one line per saved frame: why it was saved and the
                          detector's boxes on it (published px + score), to
                          pre-fill labelling and to find false positives.

When:
  • every `every_s` seconds ("periodic") — backgrounds, and validation frames
    of a drone flight;
  • whenever the detector returned a box, at most once per `min_gap_s`
    ("detection") — this is how false positives (clouds, cars, leaves) are
    captured for hard-negative mining without anyone watching the screen.

JPEG encoding happens on a writer thread with a bounded queue, so a slow SD
card can never stall the camera loop. A full queue DROPS the frame and
counts it — a dropped training frame costs nothing, a stalled camera does.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("station.camera")


class FrameRecorder:

    def __init__(self, out_dir: Path, every_s: float = 1.0,
                 min_gap_s: float = 0.25, jpeg_quality: int = 95,
                 max_queue: int = 16):
        self.out_dir = Path(out_dir)
        self.every_s = float(every_s)
        self.min_gap_s = float(min_gap_s)
        self.jpeg_quality = int(jpeg_quality)
        self.saved = 0
        self.dropped = 0
        self._last_periodic: Optional[float] = None
        self._last_detection: Optional[float] = None
        self._queue: "queue.Queue" = queue.Queue(maxsize=int(max_queue))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._index = open(self.out_dir / "frames.jsonl", "a",
                           encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._writer, daemon=True,
                                        name="frame-recorder")
        self._thread.start()

    # ── Camera thread ──────────────────────────────────────────

    def offer(self, t: float, camera: str, published: np.ndarray,
              native: Optional[np.ndarray],
              boxes: Sequence[Sequence[float]]) -> Optional[str]:
        """
        Decide whether this frame is saved; returns the reason or None.

        `boxes` are [x, y, w, h, score] in published px. The arrays are
        COPIED before they are queued: the frame is also published to the
        UI, and the recorder must never see anything drawn onto it.
        """
        reason = None
        if len(boxes) and (self._last_detection is None
                           or t - self._last_detection >= self.min_gap_s):
            reason = "detection"
            self._last_detection = t
        elif self._last_periodic is None or t - self._last_periodic >= self.every_s:
            reason = "periodic"
        if reason is None:
            return None
        self._last_periodic = t
        item = (t, camera, reason, np.array(published, copy=True),
                None if native is None else np.array(native, copy=True),
                [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 3)]
                 for b in boxes])
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            return None
        return reason

    # ── Writer thread ──────────────────────────────────────────

    def _writer(self) -> None:
        import cv2
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while True:
            item = self._queue.get()
            if item is None:
                return
            t, camera, reason, pub, nat, boxes = item
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) \
                + f"_{int((t % 1) * 1000):03d}"
            safe_cam = "".join(c if c.isalnum() else "-" for c in camera)
            base = f"{stamp}_{safe_cam}"
            try:
                files = {"pub": f"{base}_pub.jpg"}
                cv2.imwrite(str(self.out_dir / files["pub"]), pub, params)
                if nat is not None:
                    files["nat"] = f"{base}_nat.jpg"
                    cv2.imwrite(str(self.out_dir / files["nat"]), nat, params)
                self._index.write(json.dumps(
                    {"t": round(t, 3), "cam": camera, "reason": reason,
                     "files": files, "pub_size": [pub.shape[1], pub.shape[0]],
                     "boxes": boxes}, separators=(",", ":")) + "\n")
                self.saved += 1
            except Exception as exc:        # never let a write kill the thread
                log.error("frame recorder: write failed: %s", exc)

    def close(self, timeout: float = 5.0) -> None:
        """Write what is queued, then stop."""
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        try:
            self._index.close()
        except OSError:
            pass
        log.info("frame recorder: %d frame(s) saved, %d dropped -> %s",
                 self.saved, self.dropped, self.out_dir)

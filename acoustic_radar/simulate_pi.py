#!/usr/bin/env python3
"""
simulate_pi.py — Run the whole station against a simulated Pi 5 + Hailo-8L.

    python simulate_pi.py                 # full run, prints achieved FPS
    python simulate_pi.py --seconds 20
    python simulate_pi.py --compare       # A/B the UI cost against the fix

═══════════════════════════════════════════════════════════════════
WHAT THIS DOES AND DOES NOT PROVE
═══════════════════════════════════════════════════════════════════

It installs fake `picamera2` and `hailo_platform` modules that present the
SAME API surface the real ones do, then runs the real CameraWorker, the real
HailoInference decode path, the real IMM tracker, the real fusion layer and
the real HUD against them.

PROVES:
  • the pipeline is wired correctly end to end — capture, letterbox,
    3-head decode, NMS, tracking, switching, fusion, HUD;
  • how much CPU the INTEGRATION LAYER itself costs per frame, which is the
    thing that caused the 37 -> 19 fps regression;
  • that the camera loop is not blocked by the UI or by audio.

DOES NOT PROVE:
  • real Hailo-8L throughput. The simulated inference sleeps for
    --infer-ms, which is an ASSUMPTION, not a measurement. Nothing here
    tells you what YOLO26n actually costs on your accelerator.
  • real ISP/CSI capture behaviour, real image quality, or real detection
    accuracy on real drones.

Absolute FPS printed here is therefore an upper bound for a machine of this
speed, not a prediction for the Pi. What transfers is the RATIO: how much
the integration layer adds on top of capture + inference.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ═══════════════════════════════════════════════════════════════
#  Fake picamera2
# ═══════════════════════════════════════════════════════════════

class FakePicamera2:
    """
    Mimics the subset of Picamera2 that CameraManager uses.

    Frame timing reproduces the real constraint that matters:
    capture_array() BLOCKS until the sensor has a frame, at the rate set by
    FrameDurationLimits. That is what makes the camera loop self-pacing.
    """

    #: Sensor rate the simulated cameras deliver, in fps. Set from the CLI.
    SENSOR_FPS = 37.0
    #: Draw a moving target so the detector has something to find.
    TARGET = True

    def __init__(self, camera_num=0):
        self.camera_num = camera_num
        self.started = False
        self._config = None
        self._next_frame = 0.0
        self._n = 0
        self.width, self.height = 640, 480

    def create_video_configuration(self, main=None, buffer_count=4,
                                   controls=None):
        cfg = {"main": main or {}, "buffer_count": buffer_count,
               "controls": controls or {}}
        return cfg

    def configure(self, config):
        self._config = config
        size = config.get("main", {}).get("size")
        if size:
            self.width, self.height = size

    def start(self):
        self.started = True
        self._next_frame = time.monotonic()

    def stop(self):
        self.started = False

    def close(self):
        self.started = False

    def capture_array(self):
        if not self.started:
            raise RuntimeError("camera not started")

        # Block until the sensor would have produced the next frame.
        period = 1.0 / self.SENSOR_FPS
        now = time.monotonic()
        if self._next_frame > now:
            time.sleep(self._next_frame - now)
        self._next_frame = max(self._next_frame + period, time.monotonic())

        self._n += 1
        # A fresh buffer every call, exactly like the real capture_array —
        # this is what makes it safe for the worker to publish the frame by
        # reference instead of copying it.
        frame = np.full((self.height, self.width, 3), 60, dtype=np.uint8)
        # Cheap texture so ego-motion/appearance code has something to chew.
        frame[::7, ::5] = 95
        if self.TARGET:
            # A drone-sized box drifting across the field of view.
            t = self._n / self.SENSOR_FPS
            cx = int(self.width * 0.5 + self.width * 0.30 * np.sin(t * 0.6))
            cy = int(self.height * 0.42 + 40 * np.sin(t * 0.9))
            half = 16 if self.camera_num == 0 else 42
            y0, y1 = max(0, cy - half), min(self.height, cy + half)
            x0, x1 = max(0, cx - half), min(self.width, cx + half)
            frame[y0:y1, x0:x1] = (30, 30, 35)
        return frame


def install_fake_picamera2() -> None:
    mod = types.ModuleType("picamera2")
    mod.Picamera2 = FakePicamera2
    sys.modules["picamera2"] = mod


# ═══════════════════════════════════════════════════════════════
#  Fake hailo_platform
# ═══════════════════════════════════════════════════════════════

class _VStreamInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeHEF:
    """
    Presents the tensor names this project's HEF really contains — verified
    against the binary: bestty_yolo26/conv61, conv64, conv77, conv80,
    conv91, conv94, input_layer1.
    """

    HEADS = [("bestty_yolo26/conv61", "bestty_yolo26/conv64", 8),
             ("bestty_yolo26/conv77", "bestty_yolo26/conv80", 16),
             ("bestty_yolo26/conv91", "bestty_yolo26/conv94", 32)]

    def __init__(self, path):
        self.path = path

    def get_input_vstream_infos(self):
        return [_VStreamInfo("bestty_yolo26/input_layer1", (640, 640, 3))]

    def get_output_vstream_infos(self):
        out = []
        for bbox, cls, stride in self.HEADS:
            g = 640 // stride
            out.append(_VStreamInfo(bbox, (g, g, 4)))
            out.append(_VStreamInfo(cls, (g, g, 1)))
        return out


class FakePipeline:
    """
    Produces output tensors in the layout the real decode path expects:
    LTRB regression in STRIDE units and a single-class logit map.

    A detection is planted at the location of the drawn target so the whole
    decode → NMS → tracker → fusion chain is genuinely exercised.
    """

    def __init__(self, hef, infer_ms):
        self.hef = hef
        self.infer_s = infer_ms / 1000.0
        self.calls = 0

    def infer(self, feed):
        self.calls += 1

        # ⚠️ ASSUMPTION, not a measurement: stands in for Hailo-8L latency.
        if self.infer_s > 0:
            time.sleep(self.infer_s)

        frame = next(iter(feed.values()))[0]          # (640, 640, 3) uint8
        # Find the dark target block the fake camera drew.
        dark = np.argwhere(frame[:, :, 0] < 45)
        out = {}
        for bbox_name, cls_name, stride in FakeHEF.HEADS:
            g = 640 // stride
            reg = np.zeros((1, g, g, 4), dtype=np.float32)
            cls = np.full((1, g, g, 1), -9.0, dtype=np.float32)   # logits
            if len(dark) and stride == 8:
                y0, x0 = dark.min(axis=0)
                y1, x1 = dark.max(axis=0)
                cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
                gy, gx = int(cy // stride), int(cx // stride)
                if 0 <= gy < g and 0 <= gx < g:
                    ccx = (gx + 0.5) * stride
                    ccy = (gy + 0.5) * stride
                    reg[0, gy, gx] = [max(0.1, (ccx - x0) / stride),
                                      max(0.1, (ccy - y0) / stride),
                                      max(0.1, (x1 - ccx) / stride),
                                      max(0.1, (y1 - ccy) / stride)]
                    cls[0, gy, gx, 0] = 2.5           # sigmoid -> ~0.92
            out[bbox_name] = reg
            out[cls_name] = cls
        return out


class _Ctx:
    def __init__(self, value=None):
        self._value = value

    def __enter__(self):
        return self._value if self._value is not None else self

    def __exit__(self, *a):
        return False


class FakeNetworkGroup:
    def __init__(self, hef, infer_ms):
        self.hef = hef
        self.infer_ms = infer_ms

    def create_params(self):
        return {}

    def activate(self, params=None):
        return _Ctx()


class FakeVDevice:
    _infer_ms = 12.0

    def configure(self, hef, params):
        return [FakeNetworkGroup(hef, self._infer_ms)]


def install_fake_hailo(infer_ms: float) -> None:
    FakeVDevice._infer_ms = infer_ms
    mod = types.ModuleType("hailo_platform")
    mod.HEF = FakeHEF
    mod.VDevice = FakeVDevice
    mod.HailoStreamInterface = types.SimpleNamespace(PCIe="PCIe")
    mod.ConfigureParams = types.SimpleNamespace(
        create_from_hef=lambda hef, interface: {})
    mod.InputVStreamParams = types.SimpleNamespace(
        make=lambda ng, format_type=None: {})
    mod.OutputVStreamParams = types.SimpleNamespace(
        make=lambda ng, format_type=None: {})
    mod.FormatType = types.SimpleNamespace(UINT8="uint8", FLOAT32="float32")
    mod.InferVStreams = lambda ng, i, o: _Ctx(
        FakePipeline(ng.hef, ng.infer_ms))
    sys.modules["hailo_platform"] = mod


# ═══════════════════════════════════════════════════════════════
#  Harness
# ═══════════════════════════════════════════════════════════════

def run_station(seconds: float, ui: bool, sensor_fps: float,
                infer_ms: float, max_ui_fps: float | None = None,
                display_scale: float | None = None,
                spin_ui: bool = False) -> dict:
    """
    Run the real CameraWorker (and optionally the real HUD) against the
    fakes, and report the camera pipeline rate that was achieved.
    """
    import logging

    import fusion_config
    from camera_cue import BearingProjector
    from camera_worker import CameraWorker
    from hud import HUD
    from sensor_fusion import SensorFusion
    from station_logging import EventLogger, setup
    from target_state import SubsystemHealth, SubsystemState

    cfg = fusion_config.load()
    cfg.logging.to_file = False
    cfg.logging.level = "WARNING"
    if max_ui_fps is not None:
        cfg.ui.max_ui_fps = max_ui_fps
    if display_scale is not None:
        cfg.ui.display_scale = display_scale

    setup(cfg.logging)
    logging.getLogger("station").setLevel(logging.WARNING)
    events = EventLogger(logging.getLogger("station.sim"), cfg.logging)

    FakePicamera2.SENSOR_FPS = sensor_fps

    worker = CameraWorker(cfg, events, str(BASE_DIR / "bestty_yolo260508.hef"))
    worker.start()

    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    fusion = SensorFusion(cfg, proj)
    hud = HUD(cfg, proj)
    healthy = SubsystemHealth(SubsystemState.ONLINE)

    # Wait for the worker to come up.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if worker.latest.get() is not None:
            break
        time.sleep(0.05)

    stop = threading.Event()
    ui_renders = [0]

    def ui_loop():
        """Mirrors main.py's loop, including the rate limit and skip."""
        last_render = 0.0
        last_seq = -1
        ui_dt = 1.0 / max(cfg.ui.max_ui_fps, 1.0)
        while not stop.is_set():
            now_t = time.monotonic()
            vis = worker.latest.get()
            new_frame = vis is not None and vis.seq != last_seq
            since = now_t - last_render
            due = spin_ui or (since >= ui_dt and (new_frame or since >= 0.25))
            if due:
                target = fusion.update(None, vis, healthy, healthy)
                hud.render(target, since,
                           camera_fps=vis.loop_fps if vis else 0.0,
                           ui_fps=0.0)
                ui_renders[0] += 1
                last_render = now_t
                if vis is not None:
                    last_seq = vis.seq
            else:
                time.sleep(0.001)

    ui_thread = None
    if ui:
        ui_thread = threading.Thread(target=ui_loop, daemon=True)
        ui_thread.start()

    # Measure over the window, ignoring warm-up.
    #
    # ⚠️ CPU TIME is the metric that actually transfers to the Pi.
    # Frame rate on this machine does not: the fake sensor paces itself with
    # time.sleep(), which releases the GIL and burns no CPU, and an x86 host
    # has cores to spare — so even a badly-behaved UI can spin at 500 Hz here
    # without denting camera FPS. A Raspberry Pi 5 has no such slack, and
    # there the same spinning thread is what took 37 fps down to 19.
    #
    # time.process_time() sums CPU across all threads of the process, so
    # "CPU-seconds consumed per second of wall clock" = how many cores the
    # station demands. That number is hardware-independent and is the honest
    # way to compare the two UI loops.
    time.sleep(1.0)
    start_obs = worker.latest.get()
    start_seq = start_obs.seq if start_obs else 0
    t0 = time.monotonic()
    cpu0 = time.process_time()
    time.sleep(seconds)
    cpu_used = time.process_time() - cpu0
    end_obs = worker.latest.get()
    elapsed = time.monotonic() - t0
    end_seq = end_obs.seq if end_obs else 0

    stop.set()
    if ui_thread:
        ui_thread.join(timeout=2.0)
    worker.stop()
    worker.join(timeout=5.0)

    frames = end_seq - start_seq
    return {
        "camera_fps": frames / max(elapsed, 1e-6),
        "cores": cpu_used / max(elapsed, 1e-6),
        "frames": frames,
        "elapsed": elapsed,
        "inference_ms": worker.inference_ms,
        "ui_renders": ui_renders[0],
        "ui_fps": ui_renders[0] / max(elapsed, 1e-6) if ui else 0.0,
        "tracks": len(end_obs.tracks) if end_obs else 0,
        "detections": end_obs.detection_count if end_obs else 0,
    }


def run_full_station(seconds: float, sensor_fps: float,
                     infer_ms: float) -> int:
    """
    Run the REAL main.py Station loop against the fakes.

    The window is stubbed out (imshow/waitKey/namedWindow become no-ops that
    still cost the right amount of time), so main.py's actual rate-limiting,
    frame-skip and shutdown paths are exercised — not a copy of them.
    """
    import cv2

    FakePicamera2.SENSOR_FPS = sensor_fps

    renders = [0]
    real_waitkey = cv2.waitKey

    def fake_imshow(_name, img):
        renders[0] += 1

    def fake_waitkey(ms):
        # Preserve the real behaviour that matters: it sleeps, releasing the
        # GIL, which is how the UI thread yields CPU to the sensors.
        time.sleep(max(1, int(ms)) / 1000.0)
        return -1

    cv2.imshow = fake_imshow
    cv2.waitKey = fake_waitkey
    cv2.namedWindow = lambda *a, **k: None
    cv2.setWindowProperty = lambda *a, **k: None
    cv2.getWindowProperty = lambda *a, **k: 1.0
    cv2.destroyAllWindows = lambda: None

    import main as station_main

    print("  running the real main.py Station loop (window stubbed)...")
    t0 = time.monotonic()
    cpu0 = time.process_time()
    rc = station_main.main(["--duration", str(seconds), "--no-audio",
                            "--log-level", "WARNING"])
    elapsed = time.monotonic() - t0
    cores = (time.process_time() - cpu0) / max(elapsed, 1e-6)

    cv2.waitKey = real_waitkey
    print(f"  main.py exit code : {rc}")
    print(f"  HUD frames shown  : {renders[0]} in {elapsed:.1f}s "
          f"= {renders[0]/max(elapsed,1e-6):.1f} fps")
    print(f"  CPU demand        : {cores:.2f} cores")
    ok = rc == 0 and renders[0] > 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulated Pi 5 + Hailo-8L run")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--sensor-fps", type=float, default=37.0,
                    help="frame rate the simulated sensor delivers")
    ap.add_argument("--infer-ms", type=float, default=12.0,
                    help="ASSUMED Hailo-8L inference latency (not measured)")
    ap.add_argument("--compare", action="store_true",
                    help="A/B the old spinning UI against the fixed one")
    ap.add_argument("--full", action="store_true",
                    help="run the real main.py Station loop end to end")
    args = ap.parse_args()

    install_fake_picamera2()
    install_fake_hailo(args.infer_ms)

    print("=" * 70)
    print("SIMULATED RASPBERRY PI 5 + HAILO-8L")
    print("=" * 70)
    print(f"  sensor            : {args.sensor_fps:.0f} fps, 640x480")
    print(f"  inference latency : {args.infer_ms:.0f} ms  <- ASSUMPTION, not measured")
    print(f"  window            : {args.seconds:.0f} s")
    print(f"  host              : {sys.platform}, this is NOT a Pi")
    print()

    if args.full:
        return run_full_station(args.seconds, args.sensor_fps, args.infer_ms)

    scenarios = [("camera only, no UI", dict(ui=False))]
    if args.compare:
        scenarios.append(("+ UI, OLD spinning loop",
                          dict(ui=True, spin_ui=True, display_scale=1.5,
                               max_ui_fps=30.0)))
    scenarios.append(("+ UI, fixed loop (current defaults)", dict(ui=True)))

    results = []
    for label, kwargs in scenarios:
        r = run_station(args.seconds, sensor_fps=args.sensor_fps,
                        infer_ms=args.infer_ms, **kwargs)
        results.append((label, r))
        print(f"  {label:<36} camera {r['camera_fps']:5.1f} fps"
              f"   ui {r['ui_fps']:5.1f} fps"
              f"   CPU {r['cores']:4.2f} cores"
              f"   tracks {r['tracks']}")

    print()
    base_cores = results[0][1]["cores"]
    print("  CPU demand added by the UI (the number that transfers to a Pi):")
    for label, r in results[1:]:
        extra = r["cores"] - base_cores
        print(f"    {label:<36} {extra:+.2f} cores")
    print()
    print("  A Raspberry Pi 5 has 4 cores and must also run the ISP, the")
    print("  audio thread and the DOA subprocess. Anything approaching a")
    print("  whole extra core here is what starved the camera on target.")

    print()
    ok = results[-1][1]["camera_fps"] >= args.sensor_fps * 0.95
    print(f"  RESULT: camera holds {results[-1][1]['camera_fps']:.1f} fps "
          f"with the UI running -> {'PASS' if ok else 'FAIL'}")
    if results[-1][1]["tracks"] == 0:
        print("  WARNING: no tracks confirmed — the decode/track chain "
              "did not produce a target")
        ok = False
    else:
        print(f"  detection chain produced {results[-1][1]['tracks']} "
              f"confirmed track(s) — decode, NMS and tracking all ran")

    print()
    print("  Reminder: absolute FPS here reflects THIS machine. What")
    print("  transfers to the Pi is the cost the UI adds, shown above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
test_integration.py — Validation of the unified station.

Runs without a microphone, a camera, a Hailo device or a display: the
sensors are replaced by synthetic observation streams, so the FUSION LOGIC
— which is what the integration adds — is what gets tested.

    python test_integration.py            # all scenarios
    python test_integration.py -v         # + per-step state traces

Covers the ten scenarios required of the integration, plus regression tests
for the specific bugs found and fixed during the audit.

WHAT THIS DOES NOT TEST (and cannot, off the target hardware):
    • real Hailo inference / HEF decoding
    • real Picamera2 capture and camera switching against real optics
    • real microphone capture and ONNX classification of real drone audio
Those need the Pi. See the report's "How to run" section.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import List

import numpy as np

from camera_cue import BearingProjector
from fusion_config import load as load_config
from sensor_fusion import Priority, SensorFusion, SystemState
from station_logging import EventLogger, setup
from target_state import (AcousticObservation, Approach, Freshness,
                          LatestValue, SubsystemHealth, SubsystemState,
                          VisualObservation, VisualTrack, classify_age, now)

VERBOSE = "-v" in sys.argv

HEALTHY = SubsystemHealth(SubsystemState.ONLINE)
OFFLINE = SubsystemHealth(SubsystemState.OFFLINE, "device lost")

_results: List[tuple] = []


# ═══════════════════════════════════════════════════════════════
#  Harness
# ═══════════════════════════════════════════════════════════════

class Clock:
    """A controllable time source, so timeouts are tested without sleeping."""

    def __init__(self):
        self.t = 1000.0

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def acoustic(state="ALARM", p=0.95, dist=None, lo=None, hi=None,
             bearing=142.0, conf=0.8, seq=0, t=None, threshold=0.775
             ) -> AcousticObservation:
    return AcousticObservation(
        engine_state=state, p_smoothed=p, threshold=threshold,
        bearing_deg=bearing, bearing_confidence=conf,
        distance_m=dist, distance_lo_m=lo, distance_hi_m=hi,
        timestamp=t if t is not None else now(), seq=seq)


def visual(n=1, score=0.9, bbox=(300.0, 200.0, 60.0, 40.0), state="Confirmed",
           t=None, dist=5.0) -> VisualObservation:
    tracks = tuple(
        VisualTrack(track_id=i + 1, bbox=bbox, state=state,
                    quality_score=0.5, detection_score=score,
                    detection_age_s=0.0, distance_m=dist)
        for i in range(n))
    return VisualObservation(
        frame=None, frame_width=640, frame_height=480, tracks=tracks,
        active_camera=0, camera_name="FAR/IMX477",
        timestamp=t if t is not None else now(), seq=0)


def check(name: str, condition: bool, detail: str = "") -> bool:
    _results.append((name, condition, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"   [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


def new_fusion(config=None):
    cfg = config or load_config()
    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    return cfg, SensorFusion(cfg, proj)


def drive(fusion, a_obs, v_obs, n=1, a_health=HEALTHY, v_health=HEALTHY):
    """Feed n identical updates; returns the final snapshot."""
    snap = None
    for _ in range(n):
        snap = fusion.update(a_obs, v_obs, a_health, v_health)
        if VERBOSE:
            print(f"        {snap.state.value:<32} {snap.priority.value}")
    return snap


def header(title: str) -> None:
    print(f"\n{title}")
    print("   " + "-" * (len(title) + 4))


# ═══════════════════════════════════════════════════════════════
#  TEST 1 — No drone
# ═══════════════════════════════════════════════════════════════

def test_1_no_drone():
    header("TEST 1 — no drone present")
    cfg, f = new_fusion()
    snap = drive(f, acoustic("LISTEN", p=0.05, bearing=None), visual(0), n=20)
    check("state is SEARCHING", snap.state is SystemState.SEARCHING,
          snap.state.value)
    check("no sensor owns a target", snap.priority is Priority.NONE)
    check("bearing reported as N/A", snap.bearing_text() == "N/A")
    check("distance reported as N/A", snap.distance_text() == "N/A")
    check("closure is UNKNOWN, not fabricated",
          snap.kinematics.approach is Approach.UNKNOWN)


# ═══════════════════════════════════════════════════════════════
#  TEST 2 — Acoustic only, out of camera range
# ═══════════════════════════════════════════════════════════════

def test_2_acoustic_only():
    header("TEST 2 — acoustic detection only, target at 200 m")
    cfg, f = new_fusion()

    snap = drive(f, acoustic("TRACK", p=0.90, dist=200.0), visual(0), n=3)
    check("suspicion -> ACOUSTIC_DETECTED",
          snap.state is SystemState.ACOUSTIC_DETECTED, snap.state.value)

    for i in range(10):
        snap = f.update(acoustic("ALARM", p=0.95, dist=200.0, seq=i + 10),
                        visual(0), HEALTHY, HEALTHY)
    check("confirmed -> ACOUSTIC_TRACKING",
          snap.state is SystemState.ACOUSTIC_TRACKING, snap.state.value)
    check("microphone is primary", snap.priority is Priority.ACOUSTIC)
    check("200 m is correctly judged OUT of camera range",
          snap.in_camera_range is False,
          f"gate = {snap.camera_range_gate_m:.0f} m")
    check("camera was NOT armed at 200 m",
          snap.state is not SystemState.VISUAL_ACQUISITION)


# ═══════════════════════════════════════════════════════════════
#  TEST 3 — Acoustic -> visual handover
# ═══════════════════════════════════════════════════════════════

def test_3_handover():
    header("TEST 3 — acoustic detection -> visual acquisition -> tracking")
    cfg, f = new_fusion()
    seen = []

    for i in range(8):
        s = f.update(acoustic("ALARM", dist=200.0, seq=i), visual(0),
                     HEALTHY, HEALTHY)
        seen.append(s.state)
    check("far away -> ACOUSTIC_TRACKING",
          seen[-1] is SystemState.ACOUSTIC_TRACKING, seen[-1].value)

    for i in range(8):
        s = f.update(acoustic("ALARM", dist=15.0, seq=100 + i), visual(0),
                     HEALTHY, HEALTHY)
        seen.append(s.state)
    check("inside camera range -> VISUAL_ACQUISITION",
          seen[-1] is SystemState.VISUAL_ACQUISITION, seen[-1].value)
    check("microphone still primary during acquisition",
          _PRIORITY(seen[-1]) is Priority.ACOUSTIC)

    for i in range(6):
        s = f.update(acoustic("ALARM", dist=12.0, seq=200 + i), visual(1),
                     HEALTHY, HEALTHY)
        seen.append(s.state)
    check("YOLO confirms -> VISUAL_TRACKING",
          seen[-1] is SystemState.VISUAL_TRACKING, seen[-1].value)

    snap = f.update(acoustic("ALARM", dist=12.0, seq=300), visual(1),
                    HEALTHY, HEALTHY)
    check("camera becomes primary", snap.priority is Priority.CAMERA)
    check("real YOLO confidence is displayed, not a proxy",
          snap.visual_confidence_text() == "90%",
          snap.visual_confidence_text())

    order = [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] is not s]
    check("transition order is correct",
          order == [SystemState.ACOUSTIC_DETECTED,
                    SystemState.ACOUSTIC_TRACKING,
                    SystemState.VISUAL_ACQUISITION,
                    SystemState.VISUAL_TRACKING],
          " -> ".join(s.value for s in order))


def _PRIORITY(state):
    from sensor_fusion import _PRIORITY_OF_STATE
    return _PRIORITY_OF_STATE[state]


# ═══════════════════════════════════════════════════════════════
#  TEST 4 — Visual lost, acoustic remains
# ═══════════════════════════════════════════════════════════════

def test_4_visual_lost():
    header("TEST 4 — visual track lost while the acoustic contact holds")
    cfg, f = new_fusion()
    clock = Clock()

    for i in range(14):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.05)
    check("established VISUAL_TRACKING",
          snap.state is SystemState.VISUAL_TRACKING, snap.state.value)

    # Camera loses the target; acoustic keeps reporting.
    stale_visual = visual(0, t=clock.t)
    snap = f.update(acoustic("ALARM", dist=12.0, seq=100, t=clock.t),
                    stale_visual, HEALTHY, HEALTHY, t=clock.t)
    check("visual loss -> VISUAL_LOST_ACOUSTIC_TRACKING",
          snap.state is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING,
          snap.state.value)
    check("microphone becomes primary again",
          snap.priority is Priority.ACOUSTIC)
    check("last known visual position is retained for reacquisition",
          snap.visual_track is not None,
          f"bbox={snap.visual_track.bbox if snap.visual_track else None}")

    snap = f.update(acoustic("ALARM", dist=12.0, seq=101, t=clock.t),
                    visual(0, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
    check("immediately re-cues the camera -> VISUAL_ACQUISITION",
          snap.state is SystemState.VISUAL_ACQUISITION, snap.state.value)

    for i in range(6):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=200 + i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
    check("target reacquired -> VISUAL_TRACKING",
          snap.state is SystemState.VISUAL_TRACKING, snap.state.value)

    # After the visual-lost timeout the stale box must be dropped entirely.
    cfg2, f2 = new_fusion()
    clock2 = Clock()
    for i in range(14):
        f2.update(acoustic("ALARM", dist=12.0, seq=i, t=clock2.t),
                  visual(1, t=clock2.t), HEALTHY, HEALTHY, t=clock2.t)
    t_late = clock2.advance(cfg2.visual.lost_after_s + 0.5)
    snap = f2.update(acoustic("ALARM", dist=12.0, seq=99, t=t_late),
                     visual(0, t=t_late), HEALTHY, HEALTHY, t=t_late)
    check("stale visual box is DROPPED after the lost timeout",
          snap.visual_track is None,
          "no fabricated 'current' position")


# ═══════════════════════════════════════════════════════════════
#  TEST 5 — Acoustic lost, visual remains
# ═══════════════════════════════════════════════════════════════

def test_5_acoustic_lost():
    header("TEST 5 — acoustic contact lost while the camera still sees it")
    cfg, f = new_fusion()
    clock = Clock()

    for i in range(14):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.05)
    check("established VISUAL_TRACKING",
          snap.state is SystemState.VISUAL_TRACKING)

    frozen = acoustic("ALARM", dist=12.0, seq=999, t=clock.t)
    t_late = clock.advance(cfg.acoustic.lost_after_s + 2.0)
    snap = f.update(frozen, visual(1, t=t_late), HEALTHY, HEALTHY, t=t_late)

    check("camera KEEPS tracking despite acoustic loss",
          snap.state is SystemState.VISUAL_TRACKING, snap.state.value)
    check("camera remains primary", snap.priority is Priority.CAMERA)
    check("acoustic reading is marked LOST",
          snap.acoustic_freshness is Freshness.LOST,
          f"age {snap.acoustic_age_s:.1f}s")
    check("stale acoustic data is not treated as usable",
          not snap.acoustic_freshness.usable)


# ═══════════════════════════════════════════════════════════════
#  TEST 6 — Both sensors lose the target
# ═══════════════════════════════════════════════════════════════

def test_6_both_lost():
    header("TEST 6 — both sensors lose the target")
    cfg, f = new_fusion()
    clock = Clock()

    for i in range(14):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.05)
    check("established VISUAL_TRACKING",
          snap.state is SystemState.VISUAL_TRACKING)

    # Everything goes quiet. Time advances past every timeout.
    silent_a = acoustic("LISTEN", p=0.02, bearing=None, seq=500, t=clock.t)
    for step in range(14):
        t = clock.advance(0.5)
        snap = f.update(
            AcousticObservation(**{**silent_a.__dict__,
                                   "seq": 500 + step, "timestamp": t}),
            visual(0, t=t), HEALTHY, HEALTHY, t=t)
        if snap.state is SystemState.TARGET_LOST:
            break
    check("-> TARGET_LOST", snap.state is SystemState.TARGET_LOST,
          snap.state.value)

    for step in range(12):
        t = clock.advance(0.5)
        snap = f.update(
            AcousticObservation(**{**silent_a.__dict__,
                                   "seq": 600 + step, "timestamp": t}),
            visual(0, t=t), HEALTHY, HEALTHY, t=t)
        if snap.state is SystemState.SEARCHING:
            break
    check("-> SEARCHING after the display hold",
          snap.state is SystemState.SEARCHING, snap.state.value)
    check("no target owner", snap.priority is Priority.NONE)


# ═══════════════════════════════════════════════════════════════
#  TEST 7 — Camera switch oscillation
# ═══════════════════════════════════════════════════════════════

def test_7_switch_oscillation():
    header("TEST 7 — distance fluctuating around the switching threshold")
    from camera_worker import CameraSwitchPolicy

    cfg = load_config()
    setup(cfg.logging)
    events = EventLogger(logging.getLogger("station.test"), cfg.logging)

    class StubManager:
        FAR_CAMERA_ID, NEAR_CAMERA_ID = 0, 1

        def __init__(self):
            self.active, self.switches = 0, 0

        def get_active_camera(self):
            return self.active

        def switch_to_near(self):
            if self.active == 1:
                return False
            self.active, self.switches = 1, self.switches + 1
            return True

        def switch_to_far(self):
            if self.active == 0:
                return False
            self.active, self.switches = 0, self.switches + 1
            return True

    import camera_manager
    real = camera_manager.CameraManager
    camera_manager.CameraManager = StubManager
    try:
        # Mid-band hover with jitter crossing BOTH thresholds.
        results = {}
        for frames in (1, cfg.switching.confirm_frames):
            cfg.switching.confirm_frames = frames
            policy = CameraSwitchPolicy(cfg, events)
            mgr = StubManager()
            rng = np.random.default_rng(1)
            for _ in range(300):
                d = 1.75 + float(rng.normal(0.0, 0.35))
                policy.evaluate(mgr, mgr.get_active_camera(), d, 100.0)
            results[frames] = mgr.switches

        single = results[1]
        confirmed = results[cfg.switching.confirm_frames]
        check("single-frame decision oscillates (the original behaviour)",
              single > 10, f"{single} switches / 300 frames")
        check("confirmation gate stops the oscillation",
              confirmed == 0, f"{confirmed} switches / 300 frames")

        # A genuine approach must still switch, exactly once each way.
        policy = CameraSwitchPolicy(cfg, events)
        mgr = StubManager()
        for d in np.linspace(3.0, 0.8, 60):
            policy.evaluate(mgr, mgr.get_active_camera(), float(d), 100.0)
        inbound = mgr.switches
        for d in np.linspace(0.8, 3.0, 60):
            policy.evaluate(mgr, mgr.get_active_camera(), float(d), 100.0)
        check("a genuine approach still switches FAR->NEAR once",
              inbound == 1, f"{inbound} switch(es)")
        check("and NEAR->FAR once on departure",
              mgr.switches == 2 and mgr.active == 0,
              f"{mgr.switches} total, now on camera {mgr.active}")
    finally:
        camera_manager.CameraManager = real


# ═══════════════════════════════════════════════════════════════
#  TEST 8 / 9 — Subsystem failure isolation
# ═══════════════════════════════════════════════════════════════

def test_8_microphone_fails():
    header("TEST 8 — microphone crashes; the camera must keep working")
    cfg, f = new_fusion()
    clock = Clock()

    for i in range(14):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.05)

    t = clock.advance(5.0)
    snap = f.update(None, visual(1, t=t), OFFLINE, HEALTHY, t=t)
    check("camera keeps tracking with the microphone offline",
          snap.state is SystemState.VISUAL_TRACKING, snap.state.value)
    check("UI can report the microphone as offline",
          not snap.acoustic_online and snap.visual_online)
    check("no acoustic values are invented while offline",
          snap.bearing_text() == "N/A")


def test_9_camera_fails():
    header("TEST 9 — camera fails; the microphone must keep working")
    cfg, f = new_fusion()
    clock = Clock()

    for i in range(14):
        snap = f.update(acoustic("ALARM", dist=12.0, seq=i, t=clock.t),
                        visual(1, t=clock.t), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.05)
    check("established VISUAL_TRACKING",
          snap.state is SystemState.VISUAL_TRACKING)

    t = clock.advance(0.5)
    snap = f.update(acoustic("ALARM", dist=12.0, seq=500, t=t), None,
                    HEALTHY, OFFLINE, t=t)
    check("falls back to acoustic when the camera dies",
          snap.state in (SystemState.VISUAL_LOST_ACOUSTIC_TRACKING,
                         SystemState.ACOUSTIC_TRACKING,
                         SystemState.VISUAL_ACQUISITION), snap.state.value)
    check("microphone is primary again", snap.priority is Priority.ACOUSTIC)
    check("UI can report the camera as offline",
          snap.acoustic_online and not snap.visual_online)

    # And it must settle into acoustic tracking, not thrash.
    for i in range(10):
        t = clock.advance(0.5)
        snap = f.update(acoustic("ALARM", dist=200.0, seq=600 + i, t=t),
                        None, HEALTHY, OFFLINE, t=t)
    check("settles in ACOUSTIC_TRACKING with the camera gone",
          snap.state is SystemState.ACOUSTIC_TRACKING, snap.state.value)


# ═══════════════════════════════════════════════════════════════
#  TEST 10 — Load / memory behaviour
# ═══════════════════════════════════════════════════════════════

def test_10_load():
    header("TEST 10 — sustained load: no deadlock, no unbounded growth")
    cfg, f = new_fusion()
    clock = Clock()

    import tracemalloc
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    t0 = time.monotonic()
    iterations = 20000
    # Track the PEAK sizes, not the final ones: the run ends during a quiet
    # phase, when the histories have just been cleared, so checking the
    # final length would pass without ever exercising the caps.
    peak_trail = peak_range = peak_transitions = 0
    for i in range(iterations):
        t = clock.advance(0.01)
        state = "ALARM" if (i // 200) % 2 == 0 else "LISTEN"
        dist = 50.0 - (i % 400) * 0.1
        f.update(acoustic(state, dist=dist, seq=i, t=t),
                 visual(1 if (i // 100) % 3 else 0, t=t), HEALTHY, HEALTHY,
                 t=t)
        peak_trail = max(peak_trail, len(f.history._trail))
        peak_range = max(peak_range, len(f.history._range_hist))
        peak_transitions = max(peak_transitions, len(f.transitions))
    elapsed = time.monotonic() - t0

    end_snapshot = tracemalloc.take_snapshot()
    diff = end_snapshot.compare_to(start_snapshot, "lineno")
    growth_kb = sum(s.size_diff for s in diff) / 1024.0
    tracemalloc.stop()

    check("completed without deadlock", True,
          f"{iterations} fusion updates in {elapsed:.2f}s "
          f"({iterations/elapsed:,.0f}/s)")
    check("memory growth stays bounded", abs(growth_kb) < 2048,
          f"{growth_kb:+.0f} KB after {iterations} updates")
    check("transition history is capped",
          0 < peak_transitions <= 200, f"peak {peak_transitions} entries")
    check("radar trail is capped",
          0 < peak_trail <= 256, f"peak {peak_trail} points")
    check("range history is capped",
          0 < peak_range <= 64, f"peak {peak_range} points")


# ═══════════════════════════════════════════════════════════════
#  Regression tests for bugs found in the audit
# ═══════════════════════════════════════════════════════════════

def test_regressions():
    header("REGRESSION — specific bugs found during the audit")

    # BUG F1: "distance unknown" was conflated with "measured out of range"
    cfg, f = new_fusion()
    for i in range(12):
        snap = f.update(acoustic("ALARM", dist=200.0, seq=i), visual(0),
                        HEALTHY, HEALTHY)
    check("F1: a measured 200 m does NOT arm the camera",
          snap.state is SystemState.ACOUSTIC_TRACKING, snap.state.value)

    cfg, f = new_fusion()
    for i in range(12):
        snap = f.update(acoustic("ALARM", dist=None, seq=i), visual(0),
                        HEALTHY, HEALTHY)
    check("F1: an UNKNOWN distance still allows acquisition",
          snap.state is SystemState.VISUAL_ACQUISITION, snap.state.value)
    check("F1: unknown range reports None, not False",
          snap.in_camera_range is None)

    # BUG F2: the target-lost timeout must be measured from the last sensor
    # CONTACT, not from the last state change. A target tracked for a long
    # time and then lost was declared lost instantly, because the time in
    # state already exceeded the 5 s timeout.
    cfg, f = new_fusion()
    clock = Clock()
    for i in range(200):                     # ~100 s in ACOUSTIC_TRACKING
        t = clock.advance(0.5)
        snap = f.update(acoustic("ALARM", dist=200.0, seq=i, t=t),
                        visual(0, t=t), HEALTHY, HEALTHY, t=t)
    check("F2: long track is held in ACOUSTIC_TRACKING",
          snap.state is SystemState.ACOUSTIC_TRACKING, snap.state.value)
    check("F2: time in state far exceeds the lost timeout",
          snap.state_age_s > cfg.fusion.target_lost_after_s,
          f"{snap.state_age_s:.0f}s in state vs "
          f"{cfg.fusion.target_lost_after_s:.0f}s timeout")

    silent = acoustic("LISTEN", p=0.02, bearing=None, seq=900, t=clock.t)
    t = clock.advance(cfg.acoustic.lost_after_s + 0.5)
    snap = f.update(AcousticObservation(**{**silent.__dict__, "timestamp": t}),
                    visual(0, t=t), HEALTHY, HEALTHY, t=t)
    check("F2: not declared lost immediately after contact drops",
          snap.state is not SystemState.TARGET_LOST, snap.state.value)

    t = clock.advance(cfg.fusion.target_lost_after_s + 0.5)
    snap = f.update(
        AcousticObservation(**{**silent.__dict__, "seq": 901, "timestamp": t}),
        visual(0, t=t), HEALTHY, HEALTHY, t=t)
    check("F2: declared lost once the no-contact timeout truly elapses",
          snap.state is SystemState.TARGET_LOST, snap.state.value)

    # BUG C1: CameraManager.get_frame() must never raise
    import camera_manager
    mgr = camera_manager.CameraManager.__new__(camera_manager.CameraManager)
    mgr.picams = {}
    mgr.active_camera = 0
    mgr.consecutive_capture_failures = 0
    mgr.total_capture_failures = 0
    mgr.total_frames = 0
    mgr.failover_after_failures = 15
    mgr.failure_log_interval = 999.0
    mgr._last_failure_log = time.time()
    mgr.last_switch_time = 0.0
    try:
        result = mgr.get_frame()
        ok = result is None
        err = ""
    except Exception as exc:
        ok, err = False, f"raised {type(exc).__name__}"
    check("C1: get_frame() returns None instead of KeyError", ok, err)

    # Failover after repeated failures
    class FakeCam:
        def capture_array(self):
            raise RuntimeError("device gone")

    mgr.picams = {0: FakeCam(), 1: object()}
    mgr.active_camera = 0
    mgr.consecutive_capture_failures = 0
    for _ in range(20):
        mgr.get_frame()
    check("C1: fails over to the other camera after repeated failures",
          mgr.active_camera == 1, f"active camera is now {mgr.active_camera}")

    # BUG D1: bounded xvf_host search
    import doa
    t0 = time.monotonic()
    doa.find_xvf_host()
    elapsed = time.monotonic() - t0
    check("D1: xvf_host search is time-bounded", elapsed < 3.5,
          f"{elapsed:.2f}s (was >120s, unbounded recursive glob)")

    # Detector scores really reach the tracker
    import TWO_CAMERAS_FIXED as tc
    tracker = tc.AdvancedADASTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10):
        tracker.process_frame(
            frame, [np.array([100.0 + 5 * i, 200.0, 40.0, 30.0],
                             dtype=np.float32)], True, [0.87])
    confirmed = [t for t in tracker.tracks if t.state == tc.TrackState.Confirmed]
    check("V1: real detector confidence reaches the track",
          bool(confirmed) and abs(confirmed[0].last_detection_score - 0.87) < 1e-6,
          f"score={confirmed[0].last_detection_score if confirmed else None}")

    tracker2 = tc.AdvancedADASTracker()
    for i in range(10):
        tracker2.process_frame(
            frame, [np.array([100.0 + 5 * i, 200.0, 40.0, 30.0],
                             dtype=np.float32)], True)
    c2 = [t for t in tracker2.tracks if t.state == tc.TrackState.Confirmed]
    check("V1: the original 3-argument call still works unchanged",
          bool(c2) and c2[0].last_detection_score is None)

    # Bearing cue must refuse to guess
    cfg = load_config()
    proj = BearingProjector(cfg.geometry, 640)
    cue = proj.project(0, 142.0, 0.9)
    check("G1: bearing->pixel refuses without mount calibration",
          not cue.available and "boresight" in cue.reason, cue.reason)

    cfg.geometry.camera_boresight_deg[0] = 130.0
    proj = BearingProjector(cfg.geometry, 640)
    cue = proj.project(0, 130.0, 1.0)
    check("G1: once calibrated, boresight maps to the image centre",
          cue.in_view and abs(cue.x_px - 320.0) < 0.5, f"x={cue.x_px:.1f}")
    cue_far = proj.project(0, 250.0, 1.0)
    check("G1: a target outside the FOV is reported off-screen, not clamped",
          cue_far.available and not cue_far.in_view
          and cue_far.x_px is None, cue_far.reason)

    # Derived camera range must come from the optics, not a guess
    rng = cfg.derive_visual_range_m(0)
    expected = 1274.0 * 0.25 / 16.0
    check("R1: camera range is derived from the optics",
          rng is not None and abs(rng - expected) < 0.1,
          f"{rng:.1f} m from focal 1274 px, 0.25 m drone, 16 px min box")
    cfg.geometry.camera_focal_px[0] = None
    check("R1: no focal calibration -> no derived range (not a guess)",
          cfg.derive_visual_range_m(0) is None)


def test_honesty():
    header("HONESTY — the UI must never invent a sensor value")
    cfg, f = new_fusion()

    snap = drive(f, acoustic("TRACK", p=0.9, bearing=None, dist=None),
                 visual(0), n=3)
    check("no bearing -> 'N/A'", snap.bearing_text() == "N/A")
    check("no distance -> 'N/A'", snap.distance_text() == "N/A")
    check("no visual track -> confidence 'N/A'",
          snap.visual_confidence_text() == "N/A")

    obs = acoustic(dist=87.0, lo=52.0, hi=145.0)
    check("acoustic distance is shown as an estimate with its interval",
          obs.distance_text() == "~87 m (52-145)", obs.distance_text())
    check("acoustic distance is never shown to false precision",
          "87.000" not in obs.distance_text())

    amb = acoustic(bearing=35.0)
    amb = AcousticObservation(**{**amb.__dict__, "bearing_ambiguous": True})
    check("2-mic mirror ambiguity is disclosed",
          "mirror" in amb.bearing_text(), amb.bearing_text())

    # A track with no fresh detection must not reuse an old confidence
    stale_track = VisualTrack(1, (10.0, 10.0, 50.0, 50.0), "Confirmed", 0.5,
                             None, 2.0, None)
    v = VisualObservation(tracks=(stale_track,), timestamp=now())
    cfg2, f2 = new_fusion()
    for i in range(6):
        snap = f2.update(acoustic("ALARM", dist=12.0, seq=i), v,
                         HEALTHY, HEALTHY)
    check("a coasting track reports COASTING, not a stale percentage",
          snap.visual_confidence_text() == "COASTING",
          snap.visual_confidence_text())

    # A confidence measured before the track was lost must be marked as
    # past, not shown in the same form as a live reading.
    cfg5, f5 = new_fusion()
    clock5 = Clock()
    for i in range(14):
        snap = f5.update(acoustic("ALARM", dist=12.0, seq=i, t=clock5.t),
                         visual(1, score=0.93, t=clock5.t), HEALTHY, HEALTHY,
                         t=clock5.t)
        clock5.advance(0.05)
    snap = f5.update(acoustic("ALARM", dist=12.0, seq=99, t=clock5.t),
                     visual(0, t=clock5.t), HEALTHY, HEALTHY, t=clock5.t)
    check("a past confidence is marked LAST, not shown as current",
          snap.visual_confidence_text() == "LAST 93%",
          snap.visual_confidence_text())

    # Derived closure must be UNKNOWN without enough evidence
    cfg3, f3 = new_fusion()
    clock = Clock()
    for i in range(2):
        t = clock.advance(0.5)
        snap = f3.update(acoustic("ALARM", dist=100.0, seq=i, t=t),
                         visual(0, t=t), HEALTHY, HEALTHY, t=t)
    check("closure UNKNOWN with too few samples",
          snap.kinematics.approach is Approach.UNKNOWN,
          f"{snap.kinematics.samples} samples")

    cfg4, f4 = new_fusion()
    clock = Clock()
    for i, d in enumerate([200.0, 195.0, 190.0, 185.0, 180.0, 175.0]):
        t = clock.advance(0.5)
        snap = f4.update(acoustic("ALARM", dist=d, seq=i, t=t), visual(0, t=t),
                         HEALTHY, HEALTHY, t=t)
    k = snap.kinematics
    check("closure is derived correctly from the distance series",
          k.approach is Approach.APPROACHING and abs(k.speed_mps + 10.0) < 0.5,
          f"{k.text()}")


def test_thread_safety():
    header("THREADING — publishers never block, readers never tear")
    import threading

    box: LatestValue[int] = LatestValue()
    stop = threading.Event()
    errors: List[str] = []
    reads = [0]

    def producer(base):
        i = 0
        while not stop.is_set():
            box.publish(base + i)
            i += 1

    def consumer():
        while not stop.is_set():
            v = box.get()
            reads[0] += 1
            if v is not None and not isinstance(v, int):
                errors.append(f"torn read: {v!r}")

    threads = [threading.Thread(target=producer, args=(0,), daemon=True),
               threading.Thread(target=producer, args=(10 ** 6,), daemon=True),
               threading.Thread(target=consumer, daemon=True),
               threading.Thread(target=consumer, daemon=True)]
    for t in threads:
        t.start()
    time.sleep(0.6)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    check("no torn reads under concurrent publish/read", not errors,
          f"{box.count:,} publishes, {reads[0]:,} reads")
    check("all threads terminated", not any(t.is_alive() for t in threads))

    # Freshness classification is the guard against stale data
    check("freshness: fresh", classify_age(0.5, 1.5, 3.0) is Freshness.FRESH)
    check("freshness: stale", classify_age(2.0, 1.5, 3.0) is Freshness.STALE)
    check("freshness: lost", classify_age(9.0, 1.5, 3.0) is Freshness.LOST)
    check("freshness: never", classify_age(None, 1.5, 3.0) is Freshness.NEVER)


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    logging.getLogger("station").setLevel(logging.CRITICAL)

    print("=" * 70)
    print("UNIFIED DRONE DETECTION STATION — integration validation")
    print("=" * 70)

    for fn in (test_1_no_drone, test_2_acoustic_only, test_3_handover,
               test_4_visual_lost, test_5_acoustic_lost, test_6_both_lost,
               test_7_switch_oscillation, test_8_microphone_fails,
               test_9_camera_fails, test_10_load, test_regressions,
               test_honesty, test_thread_safety):
        try:
            fn()
        except Exception as exc:
            import traceback
            _results.append((fn.__name__, False, f"raised {exc}"))
            print(f"   [FAIL] {fn.__name__} raised:")
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed != total:
        print("\nFailures:")
        for name, ok, detail in _results:
            if not ok:
                print(f"   FAIL  {name}  {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

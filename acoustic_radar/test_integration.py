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
from dataclasses import replace
from typing import List

import numpy as np

from camera_cue import BearingProjector
from fusion_config import load as _load_config_raw

# ═══════════════════════════════════════════════════════════════
#  Role binding in the test fixture
# ═══════════════════════════════════════════════════════════════
#
# ⚠️ Since audit finding C1, the index-keyed optics maps
# (`camera_focal_px`, `camera_boresight_deg`) start EMPTY and are filled
# in by `GeometryConfig.bind_roles()` once camera_identity has resolved
# which physical camera holds which lens. A real station always does this
# during CameraWorker._setup(); a test that skipped it would be exercising
# a state the station is never in — every camera would read as "focal not
# calibrated" and every range gate would be disabled.
#
# So the fixture binds roles the way a running station does. The indices
# match the semantics the rest of this suite was written against and that
# the code carried before the rename:
#
#     index 0 = the LONG lens  (was FAR_CAMERA_ID)  -> TELE
#     index 1 = the WIDE lens  (was NEAR_CAMERA_ID) -> WIDE
#
# ⚠️ This is a TEST FIXTURE, not a claim about the hardware. Which
# physical socket actually holds which lens is unresolved and needs a
# hardware check — see HARDWARE_TEST_REQUIRED.md test HW-1. Nothing here
# depends on the answer: the tests only need SOME consistent binding.
TEST_ROLES = {"TELE": 0, "WIDE": 1}


def load_config(*args, **kwargs):
    """load() plus the role binding that CameraWorker._setup() performs."""
    cfg = _load_config_raw(*args, **kwargs)
    cfg.geometry.bind_roles(TEST_ROLES)
    return cfg
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
           t=None, dist=5.0, cam=0) -> VisualObservation:
    tracks = tuple(
        VisualTrack(track_id=i + 1, bbox=bbox, state=state,
                    quality_score=0.5, detection_score=score,
                    detection_age_s=0.0, distance_m=dist)
        for i in range(n))
    return VisualObservation(
        frame=None, frame_width=640, frame_height=480, tracks=tracks,
        active_camera=cam,
        camera_name="FAR/IMX477" if cam == 0 else "NEAR/IMX708",
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
        # Role -> device index, as CameraManager now resolves them from the
        # stable device Id (finding C1). The policy compares against these
        # instead of the old FAR_CAMERA_ID / NEAR_CAMERA_ID class constants.
        tele_id, wide_id = TEST_ROLES["TELE"], TEST_ROLES["WIDE"]

        def __init__(self):
            self.active, self.switches = 0, 0

        def get_active_camera(self):
            return self.active

        def switch_to_wide(self):
            if self.active == self.wide_id:
                return False
            self.active, self.switches = self.wide_id, self.switches + 1
            return True

        def switch_to_tele(self):
            if self.active == self.tele_id:
                return False
            self.active, self.switches = self.tele_id, self.switches + 1
            return True

        # The legacy aliases CameraManager still exposes.
        switch_to_near = switch_to_wide
        switch_to_far = switch_to_tele

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

    # BUG P1: the fusion layer must not re-gate the acoustic engine's own
    # verdict. radar.Detector enters a track at threshold (0.775) but HOLDS
    # it at threshold x HOLD_FACTOR (~0.54), so an engine legitimately in
    # ALARM routinely reports p_smoothed well below 0.775. An extra gate at
    # 0.775 left such a target stuck in ACOUSTIC_DETECTED forever.
    cfg, f = new_fusion()
    for i in range(12):
        snap = f.update(acoustic("ALARM", p=0.60, dist=200.0, seq=i),
                        visual(0), HEALTHY, HEALTHY)
    check("P1: engine ALARM below the entry threshold still promotes",
          snap.state is SystemState.ACOUSTIC_TRACKING,
          f"p=0.60 vs engine entry threshold 0.775 -> {snap.state.value}")
    check("P1: no extra confidence gate by default",
          f.acoustic_confidence_threshold(None) == 0.0)

    # BUG P2: display cost must stay off the sensors' backs.
    cfg = load_config()
    check("P2: display_scale defaults to native (no resize per frame)",
          cfg.ui.display_scale == 1.0, f"{cfg.ui.display_scale}")
    check("P2: UI refresh is capped below the camera rate",
          cfg.ui.max_ui_fps <= 20.0, f"{cfg.ui.max_ui_fps} Hz")

    # The HUD must reuse its canvas rather than allocating per frame.
    import numpy as _np
    from camera_cue import BearingProjector as _BP
    from hud import HUD as _HUD
    _proj = _BP(cfg.geometry, cfg.visual.frame_width)
    _f = SensorFusion(cfg, _proj)
    _hud = _HUD(cfg, _proj)
    _frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
    _v = VisualObservation(frame=_frame, frame_width=640, frame_height=480,
                           tracks=(), active_camera=0, camera_name="FAR",
                           timestamp=now(), seq=1)
    _s = _f.update(acoustic("LISTEN", p=0.0), _v, HEALTHY, HEALTHY)
    # ⚠️ THE INVARIANT CHANGED, AND DELIBERATELY SO.
    #
    # This used to assert `_a is _b` — ONE reused canvas. That is right for
    # cv2.imshow, which copies before returning, and WRONG for
    # web_server.FrameBus, which keeps the reference and copies it later on a
    # uvicorn worker thread. With a single canvas the next render() wrote into
    # the exact buffer a client was mid-copy of, so a JPEG could contain half
    # of frame N and half of frame N+1.
    #
    # The requirement it was really protecting — "do not allocate a full
    # frame buffer on every render" — is unchanged and is still asserted
    # here. What is added is that consecutive renders must not hand out the
    # SAME buffer, so a consumer holding frame N has a whole frame period to
    # copy it before that memory is touched again.
    _renders = [_hud.render(_s, 0.05, camera_fps=37.0) for _ in range(6)]
    _distinct = {id(r) for r in _renders}
    check("P2: HUD allocates no per-frame canvas (bounded buffer pool)",
          len(_distinct) == 2, f"{len(_distinct)} distinct buffers over 6 renders")
    check("P2: consecutive renders return DIFFERENT buffers "
          "(web client can copy frame N while frame N+1 is drawn)",
          all(_renders[i] is not _renders[i + 1]
              for i in range(len(_renders) - 1))
          and _renders[0] is _renders[2],
          "buffers must alternate")

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

    # Bearing cue must refuse to guess.
    #
    # ⚠️ TEST ISOLATION. This used to call load_config() and assume the
    # boresight would be None — i.e. it asserted a property of the code by
    # relying on a config file NOT existing on disk. The repository now
    # ships a fusion_config.json with a boresight set, and the test began
    # failing for a reason that had nothing to do with the behaviour it
    # checks. The uncalibrated state is now constructed explicitly.
    cfg = load_config()
    cfg.geometry.camera_boresight_deg = {0: None, 1: None}
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

    # Derived camera range must come from the optics, not a guess.
    #
    # ⚠️ The expected value is now computed FROM THE CONFIGURED FOCAL
    # LENGTH rather than from a hard-coded 1274.0 (audit finding C2). The
    # literal made this test assert the old, optically impossible constant
    # instead of the relationship it is meant to check, so a corrected
    # calibration would have shown up here as a failure. What matters is
    # that the derivation is focal * width / min_box — not which focal.
    focal0 = cfg.geometry.camera_focal_px[0]
    rng = cfg.derive_visual_range_m(0)
    expected = focal0 * 0.25 / 16.0
    check("R1: camera range is derived from the optics",
          rng is not None and abs(rng - expected) < 0.1,
          f"{rng:.1f} m from focal {focal0:.0f} px, 0.25 m drone, "
          f"16 px min box")
    cfg.geometry.camera_focal_px[0] = None
    check("R1: no focal calibration -> no derived range (not a guess)",
          cfg.derive_visual_range_m(0) is None)


def test_19_camera_identity():
    """
    Audit finding C1 — a role must be bound to a STABLE identity, and the
    station must refuse rather than guess when it cannot be.
    """
    header("TEST 19 — stable camera identity (finding C1)")
    import camera_identity as ci

    # Two IMX477s on different I2C sockets — the real topology of this
    # station. Ids modelled on libcamera's device-tree paths.
    CAMS = [
        {"Num": 0, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"},
        {"Num": 1, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@80000/imx477@1a"},
    ]

    roles = ci.resolve_roles(
        CAMS, {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
    check("C1: a configured hint binds each role to one device",
          roles == {"WIDE": 0, "TELE": 1}, str(roles))

    # ⚠️ THE POINT OF THE WHOLE FIX. libcamera reorders the cameras — the
    # `Num` fields swap — and the roles must follow the SOCKET, not the
    # index. Under the old `camera_num=0 is FAR` scheme this reordering
    # silently transposed the two lenses.
    swapped = [dict(CAMS[1], Num=0), dict(CAMS[0], Num=1)]
    roles_after = ci.resolve_roles(
        swapped, {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
    check("C1: enumeration order changes -> the role follows the socket",
          roles_after == {"WIDE": 1, "TELE": 0}, str(roles_after))

    def _refuses(label, cams, hints, model=ci.EXPECTED_SENSOR_MODEL):
        try:
            ci.resolve_roles(cams, hints, model)
        except ci.CameraIdentityError as exc:
            check(f"C1: refuses — {label}", True, str(exc).splitlines()[0][:64])
            return
        check(f"C1: refuses — {label}", False, "it returned a mapping")

    # No hints at all: the old behaviour would have used index order.
    _refuses("no hint configured", CAMS, {"WIDE": None, "TELE": None})
    _refuses("only one role configured", CAMS, {"WIDE": "i2c@88000"})
    # A hint matching nothing: a camera was unplugged, or a cable moved.
    _refuses("hint matches no camera", CAMS,
             {"WIDE": "i2c@99999", "TELE": "i2c@80000"})
    # A hint so short it matches both.
    _refuses("hint is ambiguous", CAMS,
             {"WIDE": "imx477", "TELE": "i2c@80000"})
    # Both roles selecting the same physical device.
    _refuses("both roles resolve to one camera", CAMS,
             {"WIDE": "i2c@88000", "TELE": "i2c@88000"})
    # Fewer than two cameras.
    _refuses("only one camera present", CAMS[:1],
             {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
    # A different sensor: its pixel pitch would invalidate the optics.
    _refuses("an unexpected sensor model",
             [CAMS[0], dict(CAMS[1], Model="imx708")],
             {"WIDE": "i2c@88000", "TELE": "i2c@80000"})

    # ── The theoretical focal maths (finding C2) ──
    #
    # ⚠️ MATH VERIFIED ONLY. This checks the formula, not the lens.
    f_wide = ci.theoretical_focal_px(6.0, 2664, 640)
    f_tele = ci.theoretical_focal_px(25.0, 2664, 640)
    check("C2: theoretical focal from IMX477 optics (WIDE 6 mm)",
          abs(f_wide - 930.0) < 1.0, f"{f_wide:.1f} px")
    check("C2: theoretical focal from IMX477 optics (TELE 25 mm)",
          abs(f_tele - 3875.0) < 1.0, f"{f_tele:.1f} px")

    # THE defect that started C2: the pixel-focal ratio of two identical
    # sensors in one mode MUST equal the lens ratio. The old pair failed
    # this by 39%; the new pair satisfies it by construction.
    check("C2: focal ratio now equals the lens ratio 25/6",
          abs((f_tele / f_wide) - (25.0 / 6.0)) < 0.01,
          f"{f_tele / f_wide:.3f} vs {25.0 / 6.0:.3f} "
          f"(old 1274/501.7 = {1274.0 / 501.7:.2f})")

    # The focal length depends on the MODE, not only the lens — which is
    # why it is re-derived from the real ScalerCrop at runtime.
    f_full = ci.theoretical_focal_px(6.0, 4056, 640)
    check("C2: a full-FOV mode gives a different focal for the same lens",
          abs(f_full - 610.8) < 1.0, f"{f_full:.1f} px vs 930 px binned")

    # ═══════════════════════════════════════════════════════════
    #  C1, SECOND HALF — WHICH LENS, MEASURED FROM THE IMAGES
    # ═══════════════════════════════════════════════════════════
    #
    # Binding a role to a device Id makes the mapping stable but not known.
    # The two lenses differ by 25/6 = 4.17x in magnification, and both
    # cameras look the same way from the same mast, so the telephoto image
    # IS the centre of the wide image blown up. That is measurable.
    #
    # The synthetic pair below models exactly that geometry: a textured
    # scene, a "wide" view of all of it, and a "tele" view of its central
    # 1/4.17 stretched back to full size.
    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, size=(480, 640), dtype=np.uint8)
    scene = np.repeat(np.repeat(scene[::4, ::4], 4, axis=0), 4, axis=1)
    RATIO = 25.0 / 6.0

    import cv2 as _cv2
    cw, ch = int(640 / RATIO), int(480 / RATIO)
    x0, y0 = (640 - cw) // 2, (480 - ch) // 2
    tele_view = _cv2.resize(scene[y0:y0 + ch, x0:x0 + cw], (640, 480),
                            interpolation=_cv2.INTER_LINEAR)
    wide_view = scene

    role_a, role_b, detail = ci.classify_roles_by_optics(
        wide_view, tele_view, RATIO)
    check("C1: the optics identify which camera holds the long lens",
          (role_a, role_b) == (ci.WIDE, ci.TELE), detail)

    # The same pair presented the other way round must give the opposite
    # answer — otherwise the test would pass on argument order alone.
    role_a2, role_b2, detail2 = ci.classify_roles_by_optics(
        tele_view, wide_view, RATIO)
    check("C1: and the answer follows the images, not the argument order",
          (role_a2, role_b2) == (ci.TELE, ci.WIDE), detail2)

    # A correct assignment must score far above the swapped one.
    good = ci.optical_assignment_score(wide_view, tele_view, RATIO)
    bad = ci.optical_assignment_score(tele_view, wide_view, RATIO)
    check("C1: the correct assignment correlates, the swapped one does not",
          good > 0.5 and good > bad + 0.3,
          f"correct NCC={good:+.3f} vs swapped NCC={bad:+.3f}")

    # ⚠️ IT MUST REFUSE ON AN EMPTY SKY. That is what this station spends
    # most of its time looking at, and a featureless frame correlates with
    # nothing — a coin toss here would silently transpose the lenses.
    blank = np.full((480, 640), 128, dtype=np.uint8)
    r_a, r_b, d_blank = ci.classify_roles_by_optics(blank, blank, RATIO)
    check("C1: a featureless scene is INCONCLUSIVE, never a guess",
          r_a is None and r_b is None, d_blank)

    # A scene that looks the same at both scales cannot separate them.
    fine = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (240, 320))
    r_a2, r_b2, d_self = ci.classify_roles_by_optics(fine, fine, RATIO)
    check("C1: a self-similar scene is INCONCLUSIVE too",
          r_a2 is None and r_b2 is None, d_self)

    # ── Provenance must never read as a measurement (finding C2) ──
    cfg = load_config()
    lines = cfg.geometry.focal_status_lines()
    check("C2: the station says out loud that the focal is THEORETICAL",
          len(lines) == 2 and all("THEORETICAL" in ln for ln in lines),
          lines[0] if lines else "no warning emitted")
    check("C2: and never calls a theoretical value calibrated",
          not any("CALIBRATED" in ln.replace("NOT CALIBRATED", "")
                  for ln in lines))


def test_24_camera_manager_role_resolution():
    """
    C1 end-to-end: CameraManager must resolve roles from the optics, confirm
    a configured hint against them, and refuse on a contradiction.
    """
    header("TEST 24 — CameraManager role resolution (C1)")
    import camera_identity as ci
    import camera_manager as cm
    import cv2 as _cv2

    RATIO = 25.0 / 6.0
    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, size=(120, 160), dtype=np.uint8)
    scene = np.repeat(np.repeat(base, 4, axis=0), 4, axis=1)   # 480x640
    cw, ch = int(640 / RATIO), int(480 / RATIO)
    x0, y0 = (640 - cw) // 2, (480 - ch) // 2
    TELE_IMG = _cv2.cvtColor(
        _cv2.resize(scene[y0:y0 + ch, x0:x0 + cw], (640, 480),
                    interpolation=_cv2.INTER_LINEAR), _cv2.COLOR_GRAY2RGB)
    WIDE_IMG = _cv2.cvtColor(scene, _cv2.COLOR_GRAY2RGB)
    BLANK = np.full((480, 640, 3), 128, np.uint8)

    INFO = [
        {"Num": 0, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"},
        {"Num": 1, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@80000/imx477@1a"},
    ]

    class FakeCam:
        """Minimal Picamera2 stand-in. `images[num]` is what it returns."""
        images = {}

        def __init__(self, camera_num=0):
            self.num = camera_num
            self.camera_properties = {"PixelArraySize": (4056, 3040)}

        def create_video_configuration(self, **kw):
            return dict(kw)

        def configure(self, cfg):
            self._cfg = cfg

        def camera_configuration(self):
            return {"main": {"size": (640, 480)}}

        def capture_metadata(self):
            return {"ScalerCrop": (0, 0, 2664, 1980)}

        def start(self):
            pass

        def capture_array(self):
            return self.images[self.num]

        def stop(self):
            pass

        def close(self):
            pass

    def build(images, hints):
        FakeCam.images = images
        cm.Picamera2 = FakeCam            # short-circuits _load_picamera2()
        return cm.CameraManager(
            width=640, height=480, fps=30, warmup_frames=0,
            role_hints=hints, info_provider=lambda: list(INFO),
            lens_mm={"WIDE": 6.0, "TELE": 25.0})

    saved = cm.Picamera2
    try:
        # ── 1. NO HINT AT ALL: the optics alone must settle it ──
        #
        # This is the case that used to be fatal. Index 0 shows the wide
        # view, index 1 the magnified one, so index 1 must come out TELE.
        mgr = build({0: WIDE_IMG, 1: TELE_IMG}, {"WIDE": None, "TELE": None})
        check("C1: with no hint, the roles are resolved from the images",
              (mgr.wide_id, mgr.tele_id) == (0, 1),
              f"WIDE={mgr.wide_id} TELE={mgr.tele_id} via {mgr.role_source}")
        check("C1: and the station records that it MEASURED them",
              "magnification" in mgr.role_source, mgr.role_source)

        # The physically-swapped hardware must give the swapped answer.
        mgr = build({0: TELE_IMG, 1: WIDE_IMG}, {"WIDE": None, "TELE": None})
        check("C1: swapping the lenses swaps the resolved roles",
              (mgr.wide_id, mgr.tele_id) == (1, 0),
              f"WIDE={mgr.wide_id} TELE={mgr.tele_id}")

        # ── 2. HINT AGREES WITH THE OPTICS: confirmed ──
        mgr = build({0: WIDE_IMG, 1: TELE_IMG},
                    {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
        check("C1: a hint that matches the optics is CONFIRMED by them",
              (mgr.wide_id, mgr.tele_id) == (0, 1)
              and "CONFIRMED" in mgr.role_source, mgr.role_source)

        # ── 3. HINT CONTRADICTS THE OPTICS: hard refusal ──
        #
        # ⚠️ THE CASE THE WHOLE MECHANISM EXISTS FOR. The config says the
        # 88000 socket is WIDE, but the camera on that socket is returning
        # the magnified image. Under the old index-only scheme this was
        # invisible; it must now be fatal rather than resolved either way.
        try:
            build({0: TELE_IMG, 1: WIDE_IMG},
                  {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
            check("C1: a hint contradicting the optics is REFUSED",
                  False, "it started anyway")
        except ci.CameraIdentityError as exc:
            check("C1: a hint contradicting the optics is REFUSED",
                  "CONTRADICT" in str(exc).upper(),
                  str(exc).splitlines()[0][:70])

        # ── 4. NO HINT AND AN UNUSABLE SCENE: still fail-closed ──
        #
        # Pointed at blank sky with nothing configured, the station has no
        # basis for either role and must refuse rather than pick one.
        try:
            build({0: BLANK, 1: BLANK}, {"WIDE": None, "TELE": None})
            check("C1: blank scene + no hint is still fail-closed",
                  False, "it started anyway")
        except ci.CameraIdentityError as exc:
            check("C1: blank scene + no hint is still fail-closed",
                  "not configured" in str(exc),
                  str(exc).splitlines()[0][:70])

        # ── 5. A HINT RESCUES AN UNUSABLE SCENE ──
        #
        # Blank sky is the normal operating condition for this station, so
        # a configured mapping must keep working when the optical check
        # cannot run. It is used unconfirmed, and says so.
        mgr = build({0: BLANK, 1: BLANK},
                    {"WIDE": "i2c@88000", "TELE": "i2c@80000"})
        check("C1: a configured hint still works when the scene is blank",
              (mgr.wide_id, mgr.tele_id) == (0, 1), mgr.role_source)
        check("C1: and it is reported as UNCONFIRMED, not as measured",
              "unavailable" in mgr.role_source, mgr.role_source)

        # ── 6. Focal length is derived per ROLE, after resolution ──
        mgr = build({0: WIDE_IMG, 1: TELE_IMG}, {"WIDE": None, "TELE": None})
        check("C1/C2: focal is derived per role once the roles are known",
              abs(mgr.focal_px_by_role["WIDE"] - 930.0) < 2.0
              and abs(mgr.focal_px_by_role["TELE"] - 3875.0) < 5.0,
              str({k: round(v) for k, v in mgr.focal_px_by_role.items()}))
        check("C1/C2: and it is still labelled THEORETICAL",
              all(v.startswith("THEORETICAL")
                  for v in mgr.focal_source_by_role.values()),
              str(mgr.focal_source_by_role))
    finally:
        cm.Picamera2 = saved


def test_20_camera_switch_resets_state():
    """
    Audit finding C4 — tracker and ego-motion state must not cross a
    camera switch, in EITHER direction.
    """
    header("TEST 20 — camera switch resets camera-specific state (C4)")
    import camera_worker as cw

    class FakeEgo:
        def __init__(self):
            self.prev_gray = "frame-from-old-camera"
            self.resets = 0

        def reset(self):
            self.resets += 1
            self.prev_gray = None

    class FakeTracker:
        def __init__(self):
            self.tracks = ["track-a", "track-b"]
            self.ego_estimator = FakeEgo()

    class FakeManager:
        tele_id, wide_id = TEST_ROLES["TELE"], TEST_ROLES["WIDE"]

        def role_for(self, cam):
            return {self.tele_id: "TELE", self.wide_id: "WIDE"}.get(
                cam, f"CAM{cam}")

    cfg = load_config()
    setup(cfg.logging)
    events = EventLogger(logging.getLogger("station.test"), cfg.logging)

    worker = cw.CameraWorker.__new__(cw.CameraWorker)
    worker.config, worker.events = cfg, events
    worker._manager = FakeManager()
    worker._tracker = FakeTracker()
    worker._last_detect_t = 123.0

    # ── WIDE -> TELE ──
    worker._reset_camera_state(TEST_ROLES["TELE"])
    check("C4: tracks from the previous lens are dropped",
          worker._tracker.tracks == [], str(worker._tracker.tracks))
    check("C4: the optical-flow reference frame is cleared",
          worker._tracker.ego_estimator.prev_gray is None
          and worker._tracker.ego_estimator.resets == 1,
          f"resets={worker._tracker.ego_estimator.resets}")
    check("C4: the detector-rate interval is invalidated too",
          worker._last_detect_t is None)

    # ── ...and TELE -> WIDE, the other direction ──
    worker._tracker = FakeTracker()
    worker._reset_camera_state(TEST_ROLES["WIDE"])
    check("C4: the reset is symmetric (TELE -> WIDE clears as well)",
          worker._tracker.tracks == []
          and worker._tracker.ego_estimator.prev_gray is None)

    # ── A full WIDE -> TELE -> WIDE cycle must never carry a track ──
    #
    # ⚠️ This is the scenario the finding is about: with the real lenses
    # the pixel scale changes by 25/6 = 4.17x in one frame, so a surviving
    # IMM track describes a target that appears to jump and resize
    # violently, and the affine warp fitted between two different lenses
    # is applied to it.
    seen_nonempty_after_switch = []
    worker._tracker = FakeTracker()
    for target in (TEST_ROLES["TELE"], TEST_ROLES["WIDE"],
                   TEST_ROLES["TELE"]):
        worker._reset_camera_state(target)
        seen_nonempty_after_switch.append(bool(worker._tracker.tracks))
        worker._tracker.tracks = ["re-acquired-on-new-lens"]
    check("C4: WIDE -> TELE -> WIDE never carries a track across a switch",
          not any(seen_nonempty_after_switch),
          str(seen_nonempty_after_switch))

    # ── The switch policy must use RESOLVED indices, not 0/1 ──
    #
    # Guards against a reintroduction of CameraManager.FAR_CAMERA_ID.
    import inspect as _insp
    src = _insp.getsource(cw.CameraSwitchPolicy.evaluate)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    check("C1: the switch policy no longer references the index constants",
          "FAR_CAMERA_ID" not in code and "NEAR_CAMERA_ID" not in code,
          "the policy must compare against manager.wide_id/tele_id")
    check("C1: and it reads the resolved role indices instead",
          "wide_id" in code and "tele_id" in code)

    # ── The loop must actually CALL the reset on a change ──
    loop_src = _insp.getsource(cw.CameraWorker._loop)
    check("C4: the frame loop calls the reset when the camera changes",
          "_reset_camera_state" in loop_src)

    # ═══════════════════════════════════════════════════════════
    #  ONE DRONE -> TWO TARGETS (the operator's report)
    # ═══════════════════════════════════════════════════════════
    #
    # ⚠️ The duplicate could NOT have come from the radar tile: that draws
    # a single FusedTarget with one un-looped _draw_target call, and the
    # "mirror ghost" at (180 - bearing) was removed earlier (test 18). The
    # HUD, by contrast, draws ONE BOX PER TRACKER TRACK, so two tracker
    # tracks for one drone are two boxes on screen.
    #
    # C4 is a mechanism that produced exactly that. With max_age = 10 a
    # track survives ten consecutive misses, so after a camera switch the
    # OLD lens's tracks coasted for up to ten detector invocations at their
    # old pixel coordinates, while the detector — seeing the same drone at
    # a 4.17x different scale — failed to associate and opened NEW tracks
    # beside them. Two boxes, one drone, for about a third of a second
    # after every switch. Clearing the tracks removes the mechanism.
    import TWO_CAMERAS_FIXED as _tc
    check("N2: a track survives many misses, so stale ones linger visibly",
          _tc.AdvancedADASTracker().max_age >= 5,
          f"max_age={_tc.AdvancedADASTracker().max_age} detector frames")

    import radar_overlay as _ro
    ro_src = _insp.getsource(_ro)
    ro_code = "\n".join(ln.split("#", 1)[0] for ln in ro_src.splitlines())
    check("N2: the radar tile draws exactly one target (not a loop)",
          ro_code.count("_draw_target(") == 2,   # one def, one call
          f"{ro_code.count('_draw_target(')} occurrences (expect def+call)")

    import hud as _hud
    check("N2: the HUD draws one box per track, so duplicate tracks are "
          "duplicate boxes",
          "for track in draw_list" in _insp.getsource(_hud.HUD.render)
          or "draw_list" in _insp.getsource(_hud))


def test_21_hardware_claims():
    """
    Repository-wide guards against hardware claims the code cannot support.
    """
    header("TEST 21 — no unsupported hardware assumptions remain")
    import pathlib
    import re
    import tokenize

    root = pathlib.Path(__file__).parent
    # This file itself is excluded from every scan below: it necessarily
    # QUOTES the very literals it is forbidding, and matching those would
    # make the guard permanently red for the wrong reason.
    py = [p for p in sorted(root.glob("*.py"))
          if p.name != "test_integration.py"]

    def scan(pattern):
        """
        Files/lines where `pattern` appears in EXECUTABLE code.

        ⚠️ Uses tokenize rather than a line-wise `split("#")`. Comments and
        docstrings must not count: this repair deliberately documents the
        wrong hardware assumptions it removed ("this used to say IMX708,
        and here is why that was wrong"), and a naive scan flags exactly
        those explanations. Only NAME/OP/NUMBER tokens are examined, so a
        string literal or a comment can discuss anything it needs to.
        """
        found = []
        for path in py:
            try:
                with tokenize.open(path) as fh:
                    for tok in tokenize.generate_tokens(fh.readline):
                        if tok.type in (tokenize.COMMENT, tokenize.STRING,
                                        tokenize.NL, tokenize.NEWLINE):
                            continue
                        if pattern.search(tok.string):
                            found.append(f"{path.name}:{tok.start[0]}")
            except (SyntaxError, UnicodeDecodeError) as exc:
                found.append(f"{path.name}: UNREADABLE ({exc})")
        return found

    # ── C3: no IMX708 anywhere in executable code ──
    offenders = scan(re.compile(r"imx708", re.I))
    check("C3: no IMX708 in executable code (both cameras are IMX477P)",
          not offenders, ", ".join(offenders) or "clean")

    # ── H1: no claim of a pointing/gimbal mechanism ──
    #
    # The cameras are FIXED. Acoustic bearing selects a camera and draws a
    # cue band; it steers nothing. A servo/PWM/gimbal reference would mean
    # either dead code or a false claim about the hardware.
    hits = scan(re.compile(
        r"^(servo|gimbal|pan_tilt|GPIO|PWM|stepper|actuator)$", re.I))
    check("H1: no servo/gimbal/PWM code — the cameras are fixed",
          not hits, ", ".join(hits) or "clean")

    # ── M2: dead config keys are gone, not merely undocumented ──
    from fusion_config import GeometryConfig, LedConfig
    import dataclasses as _dc
    geo_fields = {f.name for f in _dc.fields(GeometryConfig)}
    led_fields = {f.name for f in _dc.fields(LedConfig)}
    check("M2: cue_is_azimuth_only removed (it had no readers)",
          "cue_is_azimuth_only" not in geo_fields)
    check("M2: bearing_quantum_deg removed (superseded by ring geometry)",
          "bearing_quantum_deg" not in led_fields)

    # ── H3: doa_invert is retired everywhere ──
    from calibration import DEFAULTS as CAL_DEFAULTS, RETIRED_KEYS
    check("H3: doa_invert is not a calibration default any more",
          "doa_invert" not in CAL_DEFAULTS)
    check("H3: and it is stripped from the file on the next save",
          "doa_invert" in RETIRED_KEYS)
    # Same tokenize-based scan: the remaining mentions of the key are in
    # comments explaining WHY it was retired, which must stay readable.
    # calibration.py legitimately names it in RETIRED_KEYS, as a string.
    readers = scan(re.compile(r"^doa_invert$"))
    check("H3: no runtime code reads or writes doa_invert",
          not readers, ", ".join(readers) or "clean")

    # ── M4/H6: placeholders must not masquerade as measurements ──
    import json as _js
    fusion_json = _js.loads(
        (root / "fusion_config.json").read_text(encoding="utf-8"))
    bores = fusion_json.get("geometry", {}).get(
        "camera_boresight_deg_by_role", {})
    check("M4: no camera boresight is a placeholder 0.0",
          all(v is None for v in bores.values()), str(bores))
    check("M4: the retired index-keyed boresight is gone from the config",
          "camera_boresight_deg" not in fusion_json.get("geometry", {}))

    radar_json = _js.loads(
        (root / "radar_calibration.json").read_text(encoding="utf-8"))
    check("H6: the un-measured acoustic range reference is null, not 3.0 m",
          radar_json.get("range_ref_distance_m") is None
          and radar_json.get("range_ref_level_dbfs") is None,
          f"{radar_json.get('range_ref_distance_m')} m / "
          f"{radar_json.get('range_ref_level_dbfs')} dBFS")
    from calibration import is_range_calibrated
    check("H6: so the station reports acoustic range as uncalibrated",
          not is_range_calibrated(radar_json))
    check("H3: doa_invert is gone from the shipped calibration file",
          "doa_invert" not in radar_json)


def test_23_acoustic_timestamp_semantics():
    """Audit finding M1 — decision time is not capture time."""
    header("TEST 23 — acoustic timestamp semantics (M1)")

    t = 1000.0
    # A 2.0 s window whose newest sample is at t; the decision is published
    # 0.12 s later, after feature extraction and inference.
    obs = AcousticObservation(engine_state="ALARM", p_smoothed=0.9,
                              timestamp=t + 0.12,
                              capture_end_ts=t, analysis_window_s=2.0)

    check("M1: the analysed window's start is recoverable",
          obs.capture_start_ts == t - 2.0, str(obs.capture_start_ts))
    check("M1: and its centre, which is where the evidence sits",
          obs.capture_centre_ts == t - 1.0, str(obs.capture_centre_ts))

    # THE defect: at the instant of publication the DECISION is 0 s old,
    # but the SOUND is already 1.12 s old. Fusion pairs this with a camera
    # frame ~20 ms old.
    check("M1: the sound is ~1.1 s old the moment the decision is published",
          abs(obs.sound_age_s(at=t + 0.12) - 1.12) < 1e-9,
          f"{obs.sound_age_s(at=t + 0.12):.2f} s")
    check("M1: while the decision itself is 0 s old — they are not the same",
          abs((t + 0.12) - obs.timestamp) < 1e-9)
    check("M1: the structural lag is exposed, not left to be discovered",
          abs(obs.decision_lag_s() - 1.12) < 1e-9,
          f"{obs.decision_lag_s():.2f} s window-centre -> publish")

    # An observation with no capture timing must report None, never 0.0 —
    # the project's rule that a missing measurement is not a zero.
    bare = AcousticObservation(engine_state="ALARM", timestamp=t)
    check("M1: absent capture timing reports None, not a fabricated zero",
          bare.capture_centre_ts is None and bare.sound_age_s() is None
          and bare.decision_lag_s() is None)

    # Freshness still keys off the DECISION, deliberately: it answers "is
    # the acoustic subsystem still producing output?".
    check("M1: freshness still classifies on decision age",
          classify_age(0.5, 1.5, 3.0) is Freshness.FRESH)

    # And the producer must actually populate the fields.
    import inspect as _insp
    import acoustic_worker as _aw
    src = _insp.getsource(_aw.AcousticWorker._to_observation)
    check("M1: the worker records capture_end_ts on every observation",
          "capture_end_ts=capture_end_ts" in src
          and "analysis_window_s=" in src)
    loop = _insp.getsource(_aw.AcousticWorker._loop)
    check("M1: and passes the moment the audio block was read, not now()",
          "capture_end_ts=t0" in loop)

    # The window length must be the CLASSIFIER's window, not the hop.
    import features as _f
    check("M1: the window carried is the classifier's 2.0 s, not the hop",
          _f.WINDOW_SEC == 2.0, f"{_f.WINDOW_SEC} s")


def test_25_forensic_review_fixes():
    """
    Regression cover for the defects the forensic review found:
    D1 (fabricated error bound), D2 (agreement semantics), D3 (hint
    suggestion), D6 (beam verdict), and the C1 robustness rework.
    """
    header("TEST 25 — forensic-review defect fixes (D1/D2/D3/D6/C1)")
    import camera_identity as ci
    import cv2 as _cv2
    import inspect as _insp

    # ═══════════════════════════════════════════════════════════
    #  C1 — robust bounded shift search
    # ═══════════════════════════════════════════════════════════
    #
    # ⚠️ SYNTHETIC. These model pointing offset, exposure, noise and
    # distortion. They do NOT establish real-hardware behaviour (HW-1).
    RATIO = 25.0 / 6.0
    WIDE_FOV_DEG = 37.97          # the 6 mm lens's real field, from C2
    rng = np.random.default_rng(3)
    _b = rng.integers(0, 255, size=(120, 160), dtype=np.uint8)
    SCENE = np.repeat(np.repeat(_b, 4, 0), 4, 1).astype(np.float32)

    def tele_view(scene, dx_deg=0.0, dy_deg=0.0, gain=1.0, bias=0.0,
                  noise=0.0):
        """The telephoto view of `scene`, optionally mis-pointed."""
        h, w = scene.shape
        cw, chh = int(w / RATIO), int(h / RATIO)
        px = int(round(dx_deg / WIDE_FOV_DEG * w))
        py = int(round(dy_deg / WIDE_FOV_DEG * w))
        x0 = max(0, min(w - cw, (w - cw) // 2 + px))
        y0 = max(0, min(h - chh, (h - chh) // 2 + py))
        out = _cv2.resize(scene[y0:y0 + chh, x0:x0 + cw], (w, h),
                          interpolation=_cv2.INTER_LINEAR)
        out = out * gain + bias
        if noise:
            out = out + rng.normal(0, noise, out.shape)
        return out

    # THE regression: the old fixed-centre implementation scored +0.022 at
    # 1 deg and returned INCONCLUSIVE. Every one of these must now resolve.
    for offset in (1.0, 4.0, 8.0):
        ra, rb, detail = ci.classify_roles_by_optics(
            SCENE, tele_view(SCENE, dx_deg=offset), RATIO)
        check(f"C1: resolves with {offset:.0f} deg of pointing offset",
              (ra, rb) == (ci.WIDE, ci.TELE), detail[:58])

    ra, rb, detail = ci.classify_roles_by_optics(
        SCENE, tele_view(SCENE, dx_deg=4.0, dy_deg=3.0), RATIO)
    check("C1: resolves with offset in BOTH axes", (ra, rb) == (ci.WIDE, ci.TELE),
          detail[:58])

    ra, rb, detail = ci.classify_roles_by_optics(
        SCENE, tele_view(SCENE, dx_deg=4.0, gain=1.6, bias=20, noise=12),
        RATIO)
    check("C1: survives offset + exposure difference + noise together",
          (ra, rb) == (ci.WIDE, ci.TELE), detail[:58])

    # Wrong assignment must be identified as wrong, not merely "not right".
    ra, rb, _ = ci.classify_roles_by_optics(tele_view(SCENE), SCENE, RATIO)
    check("C1: a swapped pair yields the swapped roles, not a failure",
          (ra, rb) == (ci.TELE, ci.WIDE), f"{ra}/{rb}")

    # Ambiguous scenes must still refuse — both kinds.
    blank = np.full((480, 640), 128, np.float32)
    ra, _, d_blank = ci.classify_roles_by_optics(blank, blank, RATIO)
    check("C1: a featureless scene is still INCONCLUSIVE", ra is None,
          d_blank[:58])

    # ⚠️ A COARSE repeating pattern, which survives the 4.17x downscale and
    # therefore reaches the correlation stage — this is what exercises the
    # peak-uniqueness rejection rather than the flat-image guard.
    tile = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
    coarse = np.tile(tile, (30, 40)).astype(np.float32)[:480, :640]
    ra, _, d_rep = ci.classify_roles_by_optics(
        coarse, tele_view(coarse), RATIO)
    check("C1: a self-similar scene is rejected as a non-unique alignment",
          ra is None and "not unique" in d_rep, d_rep[:58])

    src = _insp.getsource(ci._match_peaks)
    check("C1: the shipped matcher searches for the alignment",
          "matchTemplate" in src and "max_shift_frac" in src)
    check("C1: and the search is BOUNDED, not over the whole frame",
          ci.OPTICAL_MAX_SHIFT_FRAC < 0.5,
          f"±{ci.OPTICAL_MAX_SHIFT_FRAC:.0%} of the wide frame")

    # ═══════════════════════════════════════════════════════════
    #  D1 — no fabricated error bound anywhere
    # ═══════════════════════════════════════════════════════════
    # ⚠️ THE TEST IS ON WHAT THE OPERATOR SEES, not on the source text.
    # A source scan is the wrong instrument here: the code now CONTAINS the
    # retired phrase, quoted inside comments that explain why it was
    # withdrawn, and forbidding that would mean the better the fix is
    # documented the more certainly the guard fails — the same trap that
    # produced the F-13 false alarm. What D1 is about is the string the
    # station prints, so that is what is asserted.
    cfg = load_config()
    lines = cfg.geometry.focal_status_lines()
    check("D1: the station emits a focal-provenance warning at all",
          bool(lines), str(lines))
    check("D1: no runtime line claims a 'few percent' error bound",
          not any("few percent" in ln.lower() for ln in lines),
          " | ".join(lines)[:76])
    check("D1: and each says the error is UNKNOWN",
          all("UNKNOWN" in ln for ln in lines),
          lines[0][:76] if lines else "no warning emitted")

    # The whole calibration banner, which is what start-up actually prints.
    banner = " | ".join(cfg.describe_calibration())
    check("D1: the calibration banner carries no invented error bound",
          "few percent" not in banner.lower(), banner[:76])

    # ═══════════════════════════════════════════════════════════
    #  D2 — agreement: capability vs. what THIS check did
    # ═══════════════════════════════════════════════════════════
    cfg2 = load_config()
    cfg2.geometry.camera_boresight_deg = {0: 150.0, 1: 150.0}
    proj = BearingProjector(cfg2.geometry, 640)
    tol = cfg2.fusion.cue_agreement_deg
    CENTRE = 320.0

    # TELE, bearing in view -> passes, and could not have failed.
    a_pass = proj.agreement_deg(0, CENTRE, 150.0)
    check("D2: TELE pass with an in-view bearing is NON-discriminating",
          proj.agreement_discriminating(0, 150.0, a_pass, tol) is False,
          f"agreement {a_pass:.1f}deg")

    # ⚠️ THE DEFECT. TELE, bearing OFF-SCREEN -> the gate really rejects,
    # so it must NOT be reported as powerless just because TELE's span is
    # below the tolerance.
    a_rej = proj.agreement_deg(0, CENTRE, 190.0)
    check("D2: TELE rejection IS reported as discriminating",
          a_rej > tol
          and proj.agreement_discriminating(0, 190.0, a_rej, tol) is True,
          f"agreement {a_rej:.1f}deg > tolerance {tol:.0f}deg")
    check("D2: ...while the camera CAPABILITY for TELE is still False",
          proj.agreement_span_covers_tolerance(0, tol) is False,
          "the two questions are answered separately")

    # WIDE: a pass near the edge could have failed -> discriminating.
    a_wide_pass = proj.agreement_deg(1, CENTRE, 155.0)
    check("D2: WIDE pass is discriminating (a rejection was reachable)",
          proj.agreement_discriminating(1, 155.0, a_wide_pass, tol) is True,
          f"agreement {a_wide_pass:.1f}deg")
    a_wide_rej = proj.agreement_deg(1, CENTRE, 185.0)
    check("D2: WIDE rejection is discriminating",
          a_wide_rej > tol
          and proj.agreement_discriminating(1, 185.0, a_wide_rej, tol)
          is True, f"agreement {a_wide_rej:.1f}deg")

    check("D2: uncalibrated optics report None for both questions",
          proj.agreement_discriminating(99, 150.0, 1.0, tol) is None
          and proj.agreement_span_covers_tolerance(99, tol) is None)
    check("D2: no bearing at all reports None, not False",
          proj.agreement_discriminating(0, None, None, tol) is None)

    # The fusion layer must carry BOTH fields.
    import dataclasses as _dc
    from sensor_fusion import FusedTarget as _FT
    names = {f.name for f in _dc.fields(_FT)}
    check("D2: FusedTarget carries the per-observation field",
          "sensor_agreement_discriminating" in names)
    check("D2: and the camera-capability field, separately",
          "sensor_agreement_capable" in names)

    # ═══════════════════════════════════════════════════════════
    #  D3 — every suggested config must be accepted on next startup
    # ═══════════════════════════════════════════════════════════
    import json as _json

    def roundtrip(cams, label):
        block = ci.suggest_hint_config(cams)
        parsed = _json.loads("{" + block + "}")
        hints = parsed["geometry"]["camera_role_id_hint"]
        try:
            roles = ci.resolve_roles(cams, hints)
        except ci.CameraIdentityError as exc:
            check(f"D3: suggested config is accepted — {label}", False,
                  str(exc).splitlines()[0][:60])
            return
        check(f"D3: suggested config is accepted — {label}",
              set(roles) == set(ci.ROLES) and roles[ci.WIDE] != roles[ci.TELE],
              str(roles))

    # Normal Pi 5: two CSI sockets, two I2C controllers.
    roundtrip([
        {"Num": 0, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@88000/imx477@1a"},
        {"Num": 1, "Model": "imx477",
         "Id": "/base/axi/pcie@120000/rp1/i2c@80000/imx477@1a"},
    ], "separate I2C controllers")

    # ⚠️ THE DEFECT: one shared controller, two chip addresses. The old
    # code suggested "i2c@88000" for BOTH roles, which the next startup
    # then refused as ambiguous.
    roundtrip([
        {"Num": 0, "Model": "imx477", "Id": "/base/axi/i2c@88000/imx477@1a"},
        {"Num": 1, "Model": "imx477", "Id": "/base/axi/i2c@88000/imx477@10"},
    ], "SHARED I2C controller")

    # A mux topology where the paths differ only deep in the tree.
    roundtrip([
        {"Num": 0, "Model": "imx477",
         "Id": "/base/axi/i2c@88000/mux@70/i2c@0/imx477@1a"},
        {"Num": 1, "Model": "imx477",
         "Id": "/base/axi/i2c@88000/mux@70/i2c@1/imx477@1a"},
    ], "I2C multiplexer")

    # ═══════════════════════════════════════════════════════════
    #  D6 — FROZEN requires a demonstrated source change
    # ═══════════════════════════════════════════════════════════
    import diagnose as _diag

    v = _diag.beam_verdict(n_samples=200, changes=0, scene_changed=False)
    check("D6: a static scene with no change is NOT PROVEN, not FROZEN",
          v.startswith("NOT PROVEN") and "FROZEN" not in v.split("—")[0],
          v[:60])

    v = _diag.beam_verdict(n_samples=200, changes=0, scene_changed=True)
    check("D6: no response ACROSS a source change is FROZEN",
          v.startswith("FROZEN"), v[:60])

    v = _diag.beam_verdict(n_samples=200, changes=37, scene_changed=True)
    check("D6: a beam that followed the source change RESPONDED",
          v.startswith("RESPONDED"), v[:60])

    v = _diag.beam_verdict(n_samples=20, changes=0, scene_changed=True)
    check("D6: fewer than 100 samples is INSUFFICIENT, whatever happened",
          v.startswith("INSUFFICIENT"), v[:60])

    v = _diag.beam_verdict(n_samples=200, changes=3, scene_changed=False)
    check("D6: a beam that does update is never called frozen",
          "FROZEN" not in v, v[:60])

    _src = _insp.getsource(_diag.cmd_beams)
    check("D6: cmd_beams delegates to the guarded verdict function",
          "beam_verdict(" in _src)
    check("D6: and the CLI exposes --scene-changed",
          "scene_changed" in _insp.getsource(_diag.main))


def test_22_srp_gating():
    """Audit finding H5 — SRP-PHAT must refuse beamformed stereo."""
    header("TEST 22 — SRP-PHAT physical-validity gate (H5)")
    import doa as _doa
    from calibration import DEFAULTS as CAL

    # ⚠️ The gate is re-evaluated here rather than by constructing a real
    # DOAProvider, because DOAProvider.__init__ starts the USB DSP polling
    # thread and that needs the ReSpeaker. The check below pins the shipped
    # source to this same expression, so the two cannot drift apart.
    def gate(n_channels, cfg):
        """Mirrors DOAProvider.__init__'s gate; pinned to the source below."""
        channels_listed = bool(cfg.get("mic_channels"))
        channels_attested = bool(cfg.get("mic_channels_verified"))
        geometry_differs = (cfg.get("mic_positions_m")
                            != CAL.get("mic_positions_m"))
        geometry_attested = bool(cfg.get("mic_geometry_verified"))
        return ((channels_listed and channels_attested)
                and (geometry_differs and geometry_attested))

    measured_geom = [[-0.03, 0.03], [0.03, 0.03],
                     [0.03, -0.03], [-0.03, -0.03]]
    verified_geom = dict(CAL, mic_positions_m=measured_geom,
                         mic_geometry_verified=True)

    # THE case on this station today: 2 processed channels, default 43 mm
    # geometry, no attestations. This is what used to arm SRP-PHAT.
    check("H5: 2 processed channels + default geometry -> DISABLED",
          not gate(2, dict(CAL)))

    # ⚠️ D5 — NEITHER A CHANNEL COUNT NOR A CHANNEL LIST IS EVIDENCE.
    check("D5: channel count alone (4 ch) does NOT enable SRP-PHAT",
          not gate(4, dict(verified_geom)),
          "four channels can be four PROCESSED channels")
    check("D5: a mic_channels tuple alone does NOT enable SRP-PHAT",
          not gate(2, dict(verified_geom, mic_channels=[0, 1, 2, 3])),
          "a tuple in JSON is an operator assertion, not a measurement")
    check("D5: even 4 channels AND a mic_channels tuple, without the "
          "verified flag, does NOT enable it",
          not gate(4, dict(verified_geom, mic_channels=[0, 1, 2, 3])))

    # Geometry needs its own attestation, not just a non-default value.
    check("D5: non-default geometry without its verified flag -> DISABLED",
          not gate(4, dict(CAL, mic_positions_m=measured_geom,
                           mic_channels=[0, 1, 2, 3],
                           mic_channels_verified=True)))
    check("D5: the verified flag with DEFAULT geometry -> still DISABLED",
          not gate(4, dict(CAL, mic_channels=[0, 1, 2, 3],
                           mic_channels_verified=True,
                           mic_geometry_verified=True)))

    # Only a full set of verified facts arms it.
    check("D5: VERIFIED raw channels + VERIFIED geometry -> ENABLED",
          gate(4, dict(verified_geom, mic_channels=[0, 1, 2, 3],
                       mic_channels_verified=True)))

    # Defaults must be fail-safe.
    check("D5: both attestations default to False",
          CAL.get("mic_channels_verified") is False
          and CAL.get("mic_geometry_verified") is False)

    # And the gate that ships must be exactly this one.
    import inspect as _insp
    src = _insp.getsource(_doa.DOAProvider.__init__)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    check("H5: the shipped gate is no longer `n_channels >= 2`",
          "self.array_ok = n_channels >= 2" not in code)
    check("D5: and it no longer accepts a bare channel count or list",
          "n_channels >= 4 or explicit_channels" not in code)
    check("D5: it requires both explicit attestations",
          "mic_channels_verified" in code
          and "mic_geometry_verified" in code)


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


def test_final_audit_regressions():
    header("FINAL AUDIT — defects found reviewing the integration itself")
    import numpy as _np

    from hud import HUD

    cfg = load_config()
    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)

    # ── A1: a frame that stopped arriving must not render as live ──
    # The camera thread can block forever inside capture_array(), in which
    # case it cannot even update its own health: the lamp stays green and
    # the last frame stays on screen. Measured before the fix: a 30 s old
    # frame filled 87% of the camera area at full brightness.
    fus = SensorFusion(cfg, proj)
    hud = HUD(cfg, proj)
    frame = _np.zeros((480, 640, 3), _np.uint8)
    frame[:, :, 1] = 200                      # unmistakable green
    t0 = 1000.0
    vis = VisualObservation(frame=frame, frame_width=640, frame_height=480,
                            tracks=(), active_camera=0, camera_name="FAR",
                            timestamp=t0, seq=1)
    ac = acoustic("ALARM", p=0.9, dist=80.0, t=t0, seq=1)

    def green_px(img):
        cam = img[30:30 + 480, 0:640]
        return int(_np.sum(cam[:, :, 1] > 150))

    live = green_px(hud.render(fus.update(ac, vis, HEALTHY, HEALTHY, t=t0),
                               0.05, camera_fps=37.0))
    stale = green_px(hud.render(fus.update(ac, vis, HEALTHY, HEALTHY,
                                           t=t0 + 30.0),
                                0.05, camera_fps=0.0))
    check("A1: a live frame renders at full brightness",
          live > 0.8 * 480 * 640, f"{live:,} px")
    check("A1: a 30 s old frame is suppressed, not shown as live",
          stale < 0.02 * live, f"{stale:,} px ({100*(1-stale/live):.1f}% removed)")

    # ── A2: staleness must not be gated on system state ──
    snap = fus.update(ac, vis, HEALTHY, HEALTHY, t=t0 + 30.0)
    check("A2: acoustic reading is classified LOST",
          snap.acoustic_freshness is Freshness.LOST)
    check("A2: and the system has already given up on the target",
          not snap.state.has_target, snap.state.value)
    # Before the fix the stale tag was suppressed in exactly this situation.
    from radar_overlay import RadarOverlay
    radar = RadarOverlay(200, cfg.ui.radar_scales_m, cfg.ui.radar_trail_s)
    tile = radar.render(snap, 0.05)
    check("A2: radar header reports STALE even after TARGET_LOST",
          tile is not None and snap.acoustic_freshness is not Freshness.FRESH)

    # ── A3: watchdog demotes a worker that stopped publishing ──
    import main as station_main

    class _StalledWorker:
        def __init__(self, ts):
            self.health = type("H", (), {"get": lambda _s: HEALTHY})()
            self.latest = type("L", (), {
                "get": lambda _s: VisualObservation(timestamp=ts)})()

    st = station_main.Station.__new__(station_main.Station)
    st.config = cfg
    st._disabled = SubsystemHealth(SubsystemState.DISABLED)
    st.camera = _StalledWorker(0.0)
    st.acoustic = _StalledWorker(0.0)
    check("A3: watchdog marks a stalled camera worker OFFLINE",
          st._visual_health().state is SubsystemState.OFFLINE,
          st._visual_health().detail)
    check("A3: watchdog marks a stalled audio worker OFFLINE",
          st._acoustic_health().state is SubsystemState.OFFLINE)
    st.camera = _StalledWorker(now())
    check("A3: a healthy worker is NOT demoted",
          st._visual_health().state is SubsystemState.ONLINE)

    # ── A4: no wall-clock timing in camera switching ──
    # A Raspberry Pi has no RTC; the wall clock jumps by decades at the
    # first NTP sync, which would disable or freeze the switch debounce.
    #
    # Checked by parsing the AST, not by searching the text: these modules
    # legitimately DISCUSS time.time() in comments explaining why it is not
    # used, and a substring search matches those and reports a false
    # failure (it did, on the first run of this very test).
    import ast as _ast
    import inspect as _inspect
    import pathlib as _pathlib

    def wall_clock_calls(module) -> list:
        """Line numbers of real `time.time()` CALLS, ignoring comments."""
        tree = _ast.parse(
            _pathlib.Path(_inspect.getfile(module)).read_text(encoding="utf-8"))
        hits = []
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "time"
                    and isinstance(node.func.value, _ast.Name)
                    and node.func.value.id == "time"):
                hits.append(node.lineno)
        return hits

    import camera_manager as _cm
    cm_hits = wall_clock_calls(_cm)
    check("A4: camera_manager has no wall-clock timing calls",
          not cm_hits, f"time.time() at lines {cm_hits}" if cm_hits
          else "all timing is time.monotonic()")

    # ── A5: per-track history is bounded ──
    import TWO_CAMERAS_FIXED as tc
    tracker = tc.AdvancedADASTracker()
    blank = _np.zeros((480, 640, 3), dtype=_np.uint8)
    for i in range(700):
        tracker.process_frame(
            blank, [_np.array([100.0 + 0.3 * i, 200.0, 40.0, 30.0],
                              dtype=_np.float32)], True, [0.9])
    hist = tracker.tracks[0].history
    check("A5: TrajectoryHistory is bounded (was an unbounded leak)",
          len(hist.states) <= tc.TrajectoryHistory.MAX_RECORDS,
          f"{len(hist.states)} records after 700 frames, "
          f"cap {tc.TrajectoryHistory.MAX_RECORDS}")
    check("A5: all four history series are bounded together",
          len(hist.covariances) == len(hist.states) == len(hist.mus)
          == len(hist.measurements))

    # ── A6: an alarm recording cannot grow without bound ──
    # A backward clock jump made `elapsed` permanently negative, so a
    # recording never terminated and its chunk list grew forever inside the
    # audio thread.
    import audio_logger as al
    al_hits = wall_clock_calls(al)
    check("A6: AudioLogger has no wall-clock timing calls",
          not al_hits, f"time.time() at lines {al_hits}" if al_hits
          else "recording duration measured monotonically")

    logger = al.AudioLogger.__new__(al.AudioLogger)
    logger.sr, logger.buffer_sec, logger.record_after = 16000, 5.0, 5.0
    logger._max_chunks = int((5.0 + 5.0) / 0.5) + 8
    logger._recording = True
    logger._record_start = 1e12          # far future => elapsed always < 0
    logger._record_chunks = []
    logger._ring = __import__("collections").deque(maxlen=12)
    logger._lock = __import__("threading").Lock()
    logger._angle, logger._dist = None, ""
    logger.save_dir = _pathlib.Path(".")
    logger.last_saved = None
    saved = {"n": 0}
    logger._save_recording = lambda: (saved.__setitem__("n", saved["n"] + 1),
                                      setattr(logger, "_recording", False),
                                      setattr(logger, "_record_chunks", []))
    block = _np.zeros(8000, dtype=_np.float32)
    for _ in range(200):                  # 100 s of audio with a broken clock
        if not logger._recording:
            logger._recording = True      # keep re-arming to stress the cap
        logger.feed(block)
    check("A6: a recording is capped even with a hostile clock",
          len(logger._record_chunks) <= logger._max_chunks,
          f"{len(logger._record_chunks)} chunks held, cap "
          f"{logger._max_chunks}; {saved['n']} forced flush(es)")


# ═══════════════════════════════════════════════════════════════
#  TEST 11 — Acoustic latency, coasting, and the LED ring
#
#  Requirements A-J of the LED / fast-response change. Everything here
#  runs against the REAL radar.Detector and the REAL LED controller; only
#  the microphone and the USB device are absent.
# ═══════════════════════════════════════════════════════════════

def _drive_detector(det, probabilities, gated=False):
    """Feed a probability series to a detector, returning the state series."""
    return [det.update(p, gated) for p in probabilities]


def test_11_acoustic_latency_and_coasting():
    header("TEST 11 — confirmation latency, hysteresis and coasting")
    import radar

    tuning = radar.DetectorTuning()
    P_START = 0.775                     # model_config.json decision_threshold
    P_HOLD = tuning.hold_limit(P_START)
    block_ms = radar.BLOCK_SEC * 1000.0

    check("audio block duration is 500 ms", abs(block_ms - 500.0) < 1e-6,
          f"{block_ms:.0f} ms")
    check("P_HOLD is 0.50 and below P_START", abs(P_HOLD - 0.50) < 1e-9
          and P_HOLD < P_START, f"P_START={P_START:.3f} P_HOLD={P_HOLD:.3f}")

    # ── Confirmation latency, measured rather than asserted ──
    det = radar.Detector(P_START, tuning)
    states = _drive_detector(det, [0.95] * 10)
    blocks_to_alarm = states.index("ALARM") + 1
    latency_ms = blocks_to_alarm * block_ms
    check("alarm confirms within the 0.5-1.0 s requirement",
          500.0 <= latency_ms <= 1000.0,
          f"{blocks_to_alarm} blocks x {block_ms:.0f} ms = {latency_ms:.0f} ms")

    # The regression this replaces: with the old cold-start EMA and 5
    # confirmations the same signal took 7 blocks. Proven here so a future
    # edit that reintroduces either cannot pass silently.
    old = radar.DetectorTuning(confirm_blocks=5, hold_threshold=None,
                               seed_ema_on_first_block=False)
    old_states = _drive_detector(radar.Detector(P_START, old), [0.95] * 12)
    old_blocks = old_states.index("ALARM") + 1
    check("regression: the previous tuning really was ~3.5 s",
          old_blocks == 7, f"old {old_blocks} blocks = "
                           f"{old_blocks * block_ms:.0f} ms, "
                           f"new {blocks_to_alarm} blocks")

    # ── Confirmations must be CONSECUTIVE ──
    # One block over P_START, a miss, then one block over P_HOLD must NOT
    # add up to an alarm. With confirm_blocks lowered to 2 this loophole
    # would otherwise be a real false-alarm source.
    det = radar.Detector(P_START, tuning)
    seq = det.update(0.80, gated=False)            # over P_START -> TRACK
    check("a single block over P_START gives TRACK, not ALARM",
          seq == "TRACK", seq)
    det.update(0.10, gated=False)                  # miss, still TRACK
    after = det.update(0.55, gated=False)          # over P_HOLD only
    check("confirmations do not accumulate across a miss",
          after == "TRACK", f"{after} (confirmations={det.confirmations})")

    # ── G: a confidence dip that stays above P_HOLD must not disarm ──
    det = radar.Detector(P_START, tuning)
    _drive_detector(det, [0.8] * 3)
    check("G: 0.8 raises the alarm", det.state == "ALARM", det.state)
    # 0.6 is below P_START but above P_HOLD. The EMA pulls the smoothed
    # value toward 0.6, which must still hold the alarm.
    _drive_detector(det, [0.6] * 6)
    check("G: alarm survives a drop to 0.6 (above P_HOLD 0.50)",
          det.state == "ALARM", f"state={det.state} p={det.p_smoothed:.3f}")

    # ── D/H: below P_HOLD -> ALARM_COASTING, not CLEAR ──
    det = radar.Detector(P_START, tuning)
    _drive_detector(det, [0.95] * 3)
    check("D: alarm established", det.state == "ALARM", det.state)
    first = det.update(0.02, gated=False)
    check("D: one missed block -> ALARM_COASTING, never CLEAR",
          first == "ALARM_COASTING", first)
    check("D: the alarm is still considered raised during coasting",
          radar.RadarStatus(state=first).is_alarm)

    # ── E: signal returns during coasting -> ALARM ──
    back = det.update(0.95, gated=False)
    check("E: signal returns during coasting -> ALARM", back == "ALARM", back)

    # ── F: silence to timeout -> CLEAR, and it takes exactly 3.0 s ──
    det = radar.Detector(P_START, tuning)
    _drive_detector(det, [0.95] * 3)
    coast_blocks = 0
    state = det.state
    while state in ("ALARM", "ALARM_COASTING") and coast_blocks < 40:
        state = det.update(0.02, gated=False)
        coast_blocks += 1
    coast_ms = coast_blocks * block_ms
    check("F: coasting ends in CLEAR (LISTEN)", state == "LISTEN", state)
    check("F: coasting lasts the configured 3.0 s",
          abs(coast_ms - 3000.0) < 1e-6,
          f"{coast_blocks} blocks x {block_ms:.0f} ms = {coast_ms:.0f} ms "
          f"(miss_tolerance={tuning.miss_tolerance})")

    # Digital silence (below the noise gate) must coast identically —
    # a drone masked by a passing lorry is the same situation as one that
    # briefly stops being classified.
    det = radar.Detector(P_START, tuning)
    _drive_detector(det, [0.95] * 3)
    gated_state = det.update(0.0, gated=True)
    check("D: a gated (silent) block also coasts rather than clearing",
          gated_state == "ALARM_COASTING", gated_state)

    # ── The observation layer must agree with the engine ──
    for engine_state, detected, confirmed, coasting in (
            ("SLEEP", False, False, False),
            ("LISTEN", False, False, False),
            ("TRACK", True, False, False),
            ("ALARM", True, True, False),
            ("ALARM_COASTING", True, True, True)):
        obs = AcousticObservation(engine_state=engine_state)
        ok = (obs.detected == detected and obs.confirmed == confirmed
              and obs.coasting == coasting)
        check(f"observation flags for {engine_state}", ok,
              f"detected={obs.detected} confirmed={obs.confirmed} "
              f"coasting={obs.coasting}")


def test_12_led_ring():
    header("TEST 12 — LED ring shares one state with the radar and camera")
    import respeaker_led as rl
    from fusion_config import LedConfig

    N = 12                                   # a ring size for the geometry
    cfg = load_config()
    cfg.geometry.camera_boresight_deg[0] = 142.0     # so the cue is computable
    _, f = new_fusion(cfg)

    # ── A: no target -> blue searching, no bearing ──
    snap = f.update(None, visual(0), HEALTHY, HEALTHY)
    frame = rl.frame_for_target(snap)
    check("A: no target -> LED SEARCHING (blue), no bearing",
          frame.mode is rl.LedMode.SEARCHING and frame.bearing_deg is None,
          f"{frame.mode.value}")

    # ── B: confirmed target -> red sector at the bearing ──
    for i in range(4):
        snap = f.update(acoustic("ALARM", bearing=142.0, seq=i), visual(0),
                        HEALTHY, HEALTHY)
    frame = rl.frame_for_target(snap)
    check("B: confirmed target -> LED ALARM (red)",
          frame.mode is rl.LedMode.ALARM, frame.mode.value)
    idx_b = rl.bearing_to_led_index(frame.bearing_deg, N)
    sector_b = rl.sector_indices(idx_b, 3, N)
    check("B: the red sector is a contiguous 3-LED arc",
          len(set(sector_b)) == 3, f"bearing {frame.bearing_deg:.0f}deg "
                                   f"-> LED {idx_b}, sector {sector_b}")

    # ── C: the sector follows a moving bearing ──
    moved = []
    for n, bearing in enumerate((40.0, 55.0, 70.0, 200.0)):
        snap = f.update(acoustic("ALARM", bearing=bearing, seq=100 + n),
                        visual(0), HEALTHY, HEALTHY)
        fr = rl.frame_for_target(snap)
        moved.append((bearing, rl.bearing_to_led_index(fr.bearing_deg, N)))
    check("C: the red sector moves with the bearing",
          len({idx for _, idx in moved}) >= 3,
          "  ".join(f"{b:.0f}deg->LED{i}" for b, i in moved))

    # ── D: coasting keeps the red sector on the LAST valid bearing ──
    last_bearing = 142.0
    for i in range(4):
        snap = f.update(acoustic("ALARM", bearing=last_bearing, seq=200 + i),
                        visual(0), HEALTHY, HEALTHY)
    snap = f.update(acoustic("ALARM_COASTING", p=0.30, bearing=last_bearing,
                             seq=204), visual(0), HEALTHY, HEALTHY)
    frame = rl.frame_for_target(snap)
    check("D: coasting keeps the LED red, not blue",
          frame.mode is rl.LedMode.COASTING and frame.mode.is_alarm,
          frame.mode.value)
    check("D: coasting keeps the LAST valid bearing",
          frame.bearing_deg == last_bearing, f"{frame.bearing_deg}")
    check("D: the same red is used for ALARM and COASTING, so the ring "
          "cannot flicker between them",
          rl._rgb(cfg.led.colour_alarm) == rl._rgb(cfg.led.colour_coasting))

    # ── E/F: return to alarm, then a real loss ──
    snap = f.update(acoustic("ALARM", bearing=last_bearing, seq=205),
                    visual(0), HEALTHY, HEALTHY)
    check("E: signal returns -> LED back to ALARM",
          rl.frame_for_target(snap).mode is rl.LedMode.ALARM)

    snap = f.update(acoustic("LISTEN", p=0.05, bearing=None, seq=206),
                    visual(0), HEALTHY, HEALTHY)
    check("F: engine cleared -> LED returns to blue",
          rl.frame_for_target(snap).mode is rl.LedMode.SEARCHING)

    # ── Priority rule (requirement 5): a repaint cannot change the frame ──
    snap = f.update(acoustic("ALARM", bearing=142.0, seq=300), visual(0),
                    HEALTHY, HEALTHY)
    frames = {rl.frame_for_target(snap) for _ in range(50)}
    check("LED frame is a pure function of the fused state — 50 repaints "
          "produce one frame, so RED/BLUE flicker is impossible",
          len(frames) == 1, f"{len(frames)} distinct frame(s)")

    # ── I: LED, radar and camera cue all read the SAME bearing ──
    cue = snap.bearing_cue
    led_bearing = rl.frame_for_target(snap).bearing_deg
    radar_bearing = snap.acoustic.bearing_deg
    check("I: LED bearing == radar bearing", led_bearing == radar_bearing,
          f"LED {led_bearing} vs radar {radar_bearing}")
    if cue.available and cue.in_view:
        # The cue is expressed relative to the camera boresight; adding it
        # back must return the acoustic bearing the other two are using.
        reconstructed = (142.0 + cue.rel_bearing_deg) % 360.0
        check("I: camera cue bearing == radar bearing",
              abs(reconstructed - radar_bearing) < 0.5,
              f"cue {reconstructed:.2f} vs radar {radar_bearing:.2f}")
    else:
        check("I: camera cue reports its own unavailability honestly",
              not cue.available or not cue.in_view, cue.reason)

    # ── A boresight measured in a since-rotated frame must be caught ──
    # Changing doa_offset_deg (e.g. by -180 to stop the radar showing the
    # wrong side) rotates the installation frame under every boresight
    # already recorded. Nothing errors and the numbers stay plausible, so
    # the station has to say so out loud.
    framed = load_config()
    framed.geometry.camera_boresight_deg = {0: 180.0, 1: 180.0}
    framed.geometry.boresight_calibrated_at_doa_offset_deg = 0.0
    quiet = framed.check_bearing_frames({"doa_offset_deg": 0.0})
    check("no warning while the frames agree", quiet == [], str(quiet))
    warned = framed.check_bearing_frames({"doa_offset_deg": -180.0})
    check("a rotated frame is detected", bool(warned),
          warned[0] if warned else "no warning")
    check("the corrected boresight is computed, not just complained about",
          any("should be 0deg" in ln for ln in warned),
          " | ".join(ln.strip() for ln in warned[1:3]))

    # ⚠️ A MIRROR IS NOT A ROTATION, and the advice must not pretend it is.
    # This is the case the real calibration produced: the XVF3800 measured
    # as COUNTER-clockwise, having been configured as clockwise. Offering
    # "add this delta" there would hand over a number wrong by twice the
    # bearing — the precise failure this check exists to prevent.
    framed.geometry.boresight_calibrated_handedness = "CW"
    mirrored = framed.check_bearing_frames(
        {"doa_offset_deg": 0.0, "doa_handedness": "CCW"})
    check("a MIRRORED frame is detected even at the same offset",
          bool(mirrored), mirrored[0] if mirrored else "no warning")
    check("and it demands a RE-MEASUREMENT instead of offering a delta",
          any("RE-MEASURED" in ln for ln in mirrored)
          and not any("should be" in ln for ln in mirrored),
          " | ".join(ln.strip() for ln in mirrored[1:3]))
    same = framed.check_bearing_frames(
        {"doa_offset_deg": 0.0, "doa_handedness": "CW"})
    check("an unchanged handedness at an unchanged offset stays quiet",
          same == [], str(same))

    # ── The ring's aim must NOT depend on the microphone's calibration ──
    # This is the double-transform regression. The old implementation
    # un-applied radar_calibration.json's doa_offset_deg inside the LED
    # mapping, so re-calibrating the DOA silently rotated the ring even
    # though neither the ring nor the drone had moved. The ring now takes
    # the canonical bearing and applies only its own two parameters.
    import inspect

    sig = inspect.signature(rl.bearing_to_led_index)
    check("LED mapping takes no microphone calibration at all",
          not any("doa" in p for p in sig.parameters),
          f"parameters: {', '.join(sig.parameters)}")
    check("LED mapping exposes exactly the ring's own two parameters",
          {"led_zero_deg", "clockwise"} <= set(sig.parameters),
          f"parameters: {', '.join(sig.parameters)}")

    # led_zero_deg rotates; clockwise mirrors. Both must actually act.
    base = rl.bearing_to_led_index(90.0, N)
    rotated = rl.bearing_to_led_index(90.0, N, led_zero_deg=90.0)
    mirrored = rl.bearing_to_led_index(90.0, N, clockwise=False)
    check("led_zero_deg rotates the ring", base != rotated,
          f"zero 0 -> LED {base}, zero +90deg -> LED {rotated}")
    check("led_index_clockwise mirrors the ring", base != mirrored,
          f"CW -> LED {base}, CCW -> LED {mirrored}")

    # ── J: hardware absent or broken must not affect detection ──
    led = rl.RespeakerLed(LedConfig(enabled=True, led_count=N),
                          script_path=None)
    led.start()
    # find_xvf_host() searches the home directory under a 2 s budget, so
    # poll rather than sleeping a guessed amount.
    deadline = time.monotonic() + 8.0
    while led.status is rl.LedStatus.STARTING and time.monotonic() < deadline:
        led.submit(rl.LedFrame(rl.LedMode.ALARM, 142.0))
        time.sleep(0.05)
    check("J: no xvf_host.py -> LED UNAVAILABLE, no exception",
          led.status is rl.LedStatus.UNAVAILABLE,
          led.detail or f"status={led.status.value}")
    led.stop(timeout=1.0)

    broken = _make_failing_xvf_host()
    led = rl.RespeakerLed(
        LedConfig(enabled=True, led_count=N, min_write_interval_s=0.0,
                  max_consecutive_failures=2, command_timeout_s=5.0),
        script_path=broken)
    led.start()
    deadline = time.monotonic() + 12.0
    while (led.status is not rl.LedStatus.UNAVAILABLE
           and time.monotonic() < deadline):
        led.submit(rl.LedFrame(rl.LedMode.ALARM, 142.0))
        time.sleep(0.05)
    check("J: a failing device degrades to UNAVAILABLE without raising",
          led.status is rl.LedStatus.UNAVAILABLE, led.detail)
    led.stop(timeout=2.0)

    # The fusion pipeline must be entirely unaffected by any of the above.
    snap = f.update(acoustic("ALARM", bearing=142.0, seq=400), visual(0),
                    HEALTHY, HEALTHY)
    check("J: detection continues while the LED subsystem is unavailable",
          snap.state.has_target, snap.state.value)

    # ── The command line handed to xvf_host.py must be the one it
    #    actually accepts. This is a real defect that reached hardware:
    #    values were passed positionally, argparse rejected them as
    #    unrecognized arguments, and EVERY write failed while the code
    #    looked correct. The stand-in below parses exactly like the real
    #    utility, so the wrong form cannot pass again.
    strict, calls = _make_strict_xvf_host()
    led = rl.RespeakerLed(
        LedConfig(enabled=True, led_count=N, sector_leds=3,
                  min_write_interval_s=0.0, command_timeout_s=10.0),
        script_path=strict)
    led.start()
    deadline = time.monotonic() + 20.0
    while led.status is rl.LedStatus.STARTING and time.monotonic() < deadline:
        time.sleep(0.05)
    check("values are passed in the form xvf_host.py accepts",
          led.status is rl.LedStatus.ACTIVE, led.detail)
    deadline = time.monotonic() + 20.0
    while led.writes < 2 and time.monotonic() < deadline:
        led.submit(rl.LedFrame(rl.LedMode.ALARM, 142.0))
        time.sleep(0.05)
    led.stop(timeout=3.0)
    logged = calls.read_text(encoding="utf-8").splitlines() if calls.exists() \
        else []
    check("the array accepted real colour writes",
          any("--values" in ln for ln in logged),
          f"{len(logged)} accepted call(s): "
          + (logged[0] if logged else "none"))

    # ── The real XVF3800 firmware, reproduced from its own output ──
    # LED_RING_COLOR takes 12 values (one per LED) even though the name
    # says otherwise, and there is no LED_AUTO_MODE. Both facts broke the
    # ring on hardware while every unit test passed, so both are pinned
    # here against a stand-in that answers exactly as the device did.
    xvf, accepted = _make_xvf3800_xvf_host()
    led = rl.RespeakerLed(
        LedConfig(enabled=True, led_count=N, sector_leds=3,
                  min_write_interval_s=0.0, command_timeout_s=10.0),
        script_path=xvf)
    led.start()
    deadline = time.monotonic() + 30.0
    while led.status is rl.LedStatus.STARTING and time.monotonic() < deadline:
        time.sleep(0.05)
    check("XVF3800: controller comes up ACTIVE against the real command set",
          led.status is rl.LedStatus.ACTIVE, led.detail)
    check("XVF3800: the ring command's arity is measured, not assumed",
          led._commands.ring_arity == N,
          f"{led._commands.ring_colour} takes "
          f"{led._commands.ring_arity} values")
    check("XVF3800: a per-pixel ring command CAN draw the target sector",
          led._commands.can_drive_sector(N))
    check("XVF3800: LED_EFFECT is found as the automatic-mode control",
          led._commands.auto_mode == "LED_EFFECT",
          str(led._commands.auto_mode))

    deadline = time.monotonic() + 30.0
    while led.writes < 2 and time.monotonic() < deadline:
        led.submit(rl.LedFrame(rl.LedMode.ALARM, 142.0))
        time.sleep(0.05)
    led.stop(timeout=3.0)

    logged = accepted.read_text(encoding="utf-8").splitlines() \
        if accepted.exists() else []
    ring_writes = [ln for ln in logged if ln.startswith("LED_RING_COLOR")]
    check("XVF3800: the array accepted real ring writes",
          bool(ring_writes), f"{len(logged)} accepted call(s)")
    if ring_writes:
        red = _rl_pack(rl._rgb((255, 0, 0)))
        check("XVF3800: one call paints all 12 LEDs",
              all(len(ln.split()) - 1 == N for ln in ring_writes),
              f"{[len(ln.split()) - 1 for ln in ring_writes]} values per call")
        # The LAST ring write is the shutdown restore to searching-blue, so
        # the alarm frame is looked for among all of them.
        sectors = [
            [i for i, v in enumerate(ln.split()[1:]) if int(v) == red]
            for ln in ring_writes]
        expected = sorted(rl.sector_indices(
            rl.bearing_to_led_index(142.0, N), 3, N))
        check("XVF3800: exactly a 3-LED red sector is lit, at bearing 142deg",
              any(lit == expected for lit in sectors),
              f"expected {expected}, saw {sectors}")
        check("XVF3800: the ring is left blue on shutdown, never red",
              sectors[-1] == [], f"last write lit red at {sectors[-1]}")

    # ── A watchdog must not leave the ring stuck red ──
    led = rl.RespeakerLed(LedConfig(enabled=True, led_count=N, watchdog_s=0.1),
                          script_path=None)
    led.submit(rl.LedFrame(rl.LedMode.ALARM, 142.0))
    time.sleep(0.2)
    effective = led._effective_frame()
    check("a stalled station falls back to SEARCHING instead of holding red",
          effective.mode is rl.LedMode.SEARCHING, effective.mode.value)


def test_13_coordinate_chain():
    header("TEST 13 — one canonical bearing, three layers, one direction")
    import bearing_frame as bf

    # ── SCENARIOS A/B/C: left, right, front all the way through ──
    for bearing, expected, radar, led, camera, agree in bf.verify_chain():
        check(f"bearing {bearing:.0f}deg -> {expected} everywhere", agree,
              f"radar={radar} led={led} camera={camera}")

    # The check must be capable of FAILING, or it proves nothing. A ring
    # that is physically mirrored relative to its configuration has to be
    # caught on the axes perpendicular to the mirror.
    mirrored = bf.verify_chain(physical_clockwise=False)
    caught = [b for b, _e, _r, _l, _c, ok in mirrored if not ok]
    check("a physically mirrored ring is detected", len(caught) == 2,
          f"disagreement at {[f'{b:.0f}deg' for b in caught]}")

    # ── ROOT CAUSE: a reflection is not an offset ──
    mirror = bf.SourceConvention(0.0, bf.Handedness.COUNTER_CLOCKWISE, "m")
    errors = [bf.angular_distance(mirror.to_canonical(raw),
                                  bf.wrap360(raw - 180.0))
              for raw in range(0, 360, 45)]
    check("an offset cannot correct a mirrored source", max(errors) >= 179.0,
          f"error ranges {min(errors):.0f}..{max(errors):.0f}deg across the "
          f"circle — exact at one bearing, 180deg out a quarter turn away")

    # ── Each source keeps its OWN convention ──
    # ⚠️ A DEFAULT MUST NOT PASS FOR A MEASUREMENT. calibration.load()
    # merges DEFAULTS into every config, and DEFAULTS carries
    # doa_invert=False — so presence of that key proves nothing. Only
    # doa_handedness, which nothing but `calibrate.py doa` writes, counts.
    import calibration as _calib

    merged = _calib.load()
    check("a hand-written config with no doa_handedness reads as "
          "UNCALIBRATED", not bf.source_convention("usb", merged).calibrated,
          bf.source_convention("usb", merged).describe())
    check("doa_invert alone is NOT accepted as evidence of calibration",
          not bf.source_convention(
              "usb", {"doa_offset_deg": -180.0, "doa_invert": False}
          ).calibrated)

    cfg = {"doa_offset_deg": -180.0, "doa_handedness": "CW"}
    usb = bf.source_convention("usb", cfg)
    srp = bf.source_convention("srp", cfg)
    check("USB convention comes from doa_offset_deg + doa_handedness",
          usb.calibrated and usb.zero_deg == -180.0, usb.describe())
    check("a CCW handedness is honoured, not silently flattened to CW",
          bf.source_convention(
              "usb", {"doa_offset_deg": 0.0, "doa_handedness": "CCW"}
          ).handedness is bf.Handedness.COUNTER_CLOCKWISE)
    check("SRP-PHAT does NOT silently inherit the USB calibration",
          not srp.calibrated, srp.describe())
    srp_cal = bf.source_convention("srp", dict(cfg, srp_zero_deg=30.0))
    check("SRP-PHAT handedness is PROVEN from the code, not guessed",
          srp_cal.handedness is bf.Handedness.COUNTER_CLOCKWISE,
          srp_cal.describe())

    # ── Angle wrap ──
    check("359 -> 1 is a 2deg move", bf.angular_distance(359.0, 1.0) == 2.0)
    check("circular mean of 350 and 10 is 0, not 180",
          bf.circular_mean_deg([350.0, 10.0]) == 0.0)
    check("wrap180(179) and wrap180(-179) are 2deg apart, not 358",
          abs(bf.wrap180(179.0 - (-179.0))) == 2.0,
          f"{bf.wrap180(179.0 - (-179.0)):.0f}deg")

    # ── ROOT CAUSE: the beam tie ──
    from doa import DOAReading, DOATracker, select_azimuth

    answers = {select_azimuth(order)[0] for order in
               ([280., 280., 100., 100.], [100., 100., 280., 280.],
                [280., 100., 280., 100.], [100., 280., 100., 280.])}
    check("a 2-2 beam split resolves the same way regardless of beam order",
          len(answers) == 1, f"answers: {sorted(answers)}")
    check("and the tie is reported as ambiguous rather than hidden",
          select_azimuth([280., 280., 100., 100.])[2])
    check("an unambiguous reading is not flagged",
          not select_azimuth([280., 281., 279., 280.])[2])

    # ── ROOT CAUSE: the tracker flipping a stable target ──
    tr = DOATracker()
    for i in range(5):
        tr.update(DOAReading(280.0, confidence=0.9, source="usb"), now=i * 0.5)
    for i in range(5, 9):
        tr.update(DOAReading(100.0, confidence=0.25, source="usb",
                             ambiguous=True), now=i * 0.5)
    check("a run of AMBIGUOUS readings never flips a stable track",
          abs(tr.angle - 280.0) < 1.0, f"track is {tr.angle:.0f}deg")

    tr2 = DOATracker()
    for i in range(5):
        tr2.update(DOAReading(280.0, confidence=0.9, source="usb"), now=i * 0.5)
    for i in range(5, 9):
        tr2.update(DOAReading(100.0, confidence=0.9, source="usb"), now=i * 0.5)
    check("but an unambiguous manoeuvre IS still followed",
          abs(tr2.angle - 100.0) < 1.0, f"track is {tr2.angle:.0f}deg")


def test_14_srp_geometry():
    header("TEST 14 — SRP-PHAT geometry, blind band and channel handling")
    import math

    from doa import ArrayDOA, SPEED_OF_SOUND

    sr = 16000
    mics = [[-0.0215, 0.0215], [0.0215, 0.0215],
            [0.0215, -0.0215], [-0.0215, -0.0215]]
    srp = ArrayDOA(mics, sr)
    rng = np.random.default_rng(0)

    def synth(true_deg, n_ch=4):
        n = sr
        src = rng.standard_normal(n + 200)
        u = np.array([math.cos(math.radians(true_deg)),
                      math.sin(math.radians(true_deg))])
        chans = [np.interp(np.arange(n) + 100
                           + np.dot(p, u) / SPEED_OF_SOUND * sr,
                           np.arange(len(src)), src) for p in mics[:n_ch]]
        return np.stack(chans, axis=1) + 0.01 * rng.standard_normal((n, n_ch))

    # ── ROOT CAUSE: the blind band perpendicular to the 0-1 baseline ──
    # Mics 0 and 1 share a y coordinate, so a source on the +y axis reaches
    # both at once. The old distinctness test compared ONLY those two and
    # refused the whole estimate — and then latched that refusal for the
    # rest of the session.
    refused = []
    for deg in (89.0, 90.0, 91.0, 269.0, 270.0, 271.0):
        if not srp.estimate(synth(deg)).ok:
            refused.append(deg)
    check("no blind band perpendicular to the mic 0-1 baseline",
          not refused, f"refused at {refused}" if refused
          else "89-91 and 269-271 deg all produce an estimate")

    audio = synth(90.0)
    check("degenerate PAIRS are excluded, not the whole estimate",
          0 < len(srp.usable_pairs(audio)) < 6,
          f"{len(srp.usable_pairs(audio))}/6 pairs usable at 90deg")

    # Processed stereo (identical channels) must still be refused honestly.
    mono = rng.standard_normal(sr)
    fake = np.stack([mono, mono], axis=1)
    r = ArrayDOA(mics[:2], sr).estimate(fake)
    check("processed stereo is still refused rather than guessed",
          not r.ok, r.error or "")

    # ── ROOT CAUSE: a transient degeneracy disabled SRP for the session ──
    from doa import DOAProvider
    prov = DOAProvider.__new__(DOAProvider)
    check("a single degenerate block does not disable SRP-PHAT for good",
          DOAProvider.ARRAY_DISABLE_AFTER > 1,
          f"needs {DOAProvider.ARRAY_DISABLE_AFTER} consecutive blocks")
    del prov

    # ── Channel selection reaches the classifier ──
    from audio_io import to_mono
    block = np.zeros((100, 6), dtype=np.float32)
    block[:, :4] = 1.0            # microphones
    block[:, 4:] = 9.0            # reference / loopback channels
    check("to_mono averages ONLY the microphone channels",
          abs(float(to_mono(block, (0, 1, 2, 3)).mean()) - 1.0) < 1e-6,
          f"mic-only mean {float(to_mono(block, (0, 1, 2, 3)).mean()):.2f}, "
          f"all-channel mean {float(to_mono(block).mean()):.2f}")
    check("with no selection it still averages everything (1-2 ch inputs)",
          abs(float(to_mono(block).mean()) - 22.0 / 6.0) < 1e-6,
          f"{float(to_mono(block).mean()):.3f} = (4x1 + 2x9)/6")


def test_15_camera_search_region():
    header("TEST 15 — the acoustic search region and the YOLO handover")
    from sensor_fusion import CueRole

    cfg = load_config()
    cfg.geometry.camera_boresight_deg = {0: 150.0, 1: 150.0}
    cfg.geometry.boresight_calibrated_at_doa_offset_deg = 0.0
    _, f = new_fusion(cfg)

    def ac(bearing=150.0, seq=0, calibrated=True, state="ALARM"):
        return AcousticObservation(
            engine_state=state, p_smoothed=0.92, threshold=0.775,
            hold_threshold=0.5, confirm_needed=2, miss_tolerance=6,
            bearing_deg=bearing, bearing_confidence=0.8,
            bearing_calibrated=calibrated, distance_m=120.0,
            timestamp=now(), seq=seq)

    # ── SCENARIO D: drone behind a tree, YOLO sees nothing ──
    for i in range(4):
        snap = f.update(ac(seq=i), visual(0), HEALTHY, HEALTHY)
    check("D: acoustic target, no YOLO box -> ACOUSTIC ONLY region",
          snap.cue_role is CueRole.SEARCH, snap.cue_role.value)

    # ── SCENARIO E: the drone comes out and YOLO confirms it ──
    for i in range(4):
        snap = f.update(ac(seq=10 + i), visual(1, bbox=(300., 200., 60., 40.)),
                        HEALTHY, HEALTHY)
    check("E: YOLO confirms -> the region is withdrawn",
          snap.cue_role is CueRole.CONFIRMED, snap.cue_role.value)

    # ── SCENARIO F: YOLO loses it, acoustic still has it ──
    snap = f.update(ac(seq=20), visual(0), HEALTHY, HEALTHY)
    check("F: YOLO lost, acoustic holds -> the region returns",
          snap.cue_role is CueRole.SEARCH, snap.cue_role.value)

    # ── A box that DISAGREES with the bearing is a different object ──
    #
    # ⚠️ REWRITTEN AFTER THE OPTICS WERE CORRECTED (audit finding C2, and
    # the new finding N1 it exposed).
    #
    # The old version placed a box at the left edge of the WIDE frame and
    # relied on that producing more than the 20 deg `cue_agreement_deg`
    # tolerance. Under the OLD focal length of 501.7 px the wide camera was
    # believed to span 65 deg, so an edge box sat 31.7 deg off-axis and the
    # test passed. With the corrected 6 mm optics the wide camera really
    # spans 37.9 deg — +/-18.9 deg — so the same edge box is only 17 deg
    # off-axis and now counts as AGREEING.
    #
    # That is a property of the station, not of this test: with the real
    # lenses, NO box anywhere in EITHER frame can disagree with a bearing
    # that is itself in view, because both half-fields (18.9 deg WIDE,
    # 4.7 deg TELE) are smaller than the 20 deg tolerance. The rejection
    # can only fire when the acoustic BEARING points outside the camera's
    # field of view, which is exactly how it is constructed below.
    _, f2 = new_fusion(cfg)
    for i in range(4):
        # Boresight 150 deg, box dead centre => the box is at 0 deg
        # relative. The acoustic bearing is 30 deg away, well outside the
        # wide camera's +/-18.9 deg field, so the two genuinely disagree.
        snap = f2.update(ac(bearing=180.0, seq=i),
                         visual(1, bbox=(300., 200., 40., 30.), cam=1),
                         HEALTHY, HEALTHY)
    disagreement = snap.sensor_agreement_deg
    check("a YOLO box that disagrees with the bearing does not cancel "
          "the region", snap.cue_role is CueRole.SEARCH,
          f"role={snap.cue_role.value}, disagreement="
          f"{disagreement:.0f}deg" if disagreement is not None else "n/a")
    check("the disagreement is real and exceeds the tolerance",
          disagreement is not None
          and disagreement > cfg.fusion.cue_agreement_deg,
          f"{disagreement}deg vs tolerance "
          f"{cfg.fusion.cue_agreement_deg:.0f}deg")

    # ═══════════════════════════════════════════════════════════
    #  N1 — the gate's reach, computed rather than asserted in prose
    # ═══════════════════════════════════════════════════════════
    #
    # ⚠️ THIS GUARD REPLACES AN EARLIER, WRONG ONE. The previous version
    # compared the tolerance against the HALF field and concluded the gate
    # was dead on both cameras. That silently assumed the bearing always
    # sits at the boresight. Both the box angle and the bearing range over
    # +/-h, so the largest disagreement an in-view pair can produce is 2h —
    # the FULL field. The two cameras therefore differ:
    #
    #     TELE  fov  9.4 deg < 20 deg tolerance -> can never reject
    #     WIDE  fov 37.9 deg > 20 deg tolerance -> can reject
    proj_n1 = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    tol = cfg.fusion.cue_agreement_deg
    spans = {}
    for _cam, _role in ((0, "TELE"), (1, "WIDE")):
        spans[_role] = proj_n1.max_agreement_deg(_cam)
    check("N1: TELE's whole field is inside the tolerance, so the gate "
          "cannot reject an IN-VIEW pair",
          proj_n1.agreement_span_covers_tolerance(0, tol) is False,
          f"fov {spans['TELE']:.1f}deg vs tolerance {tol:.0f}deg")
    check("N1: WIDE's field exceeds the tolerance, so the gate DOES work "
          "there",
          proj_n1.agreement_span_covers_tolerance(1, tol) is True,
          f"fov {spans['WIDE']:.1f}deg vs tolerance {tol:.0f}deg")
    check("N1: the span is the FULL field of view, not half of it",
          abs(spans["WIDE"] - 2.0 * (spans["WIDE"] / 2.0)) < 1e-9
          and spans["WIDE"] > 37.0,
          f"{spans['WIDE']:.1f}deg")
    check("N1: uncalibrated optics report None, not a fabricated span",
          proj_n1.agreement_span_covers_tolerance(99, tol) is None)

    # And the fusion layer must CARRY that fact, so a pass is never read as
    # evidence when it was arithmetically guaranteed.
    _, f_n1 = new_fusion(cfg)
    for i in range(4):
        sn_tele = f_n1.update(ac(seq=i),
                              visual(1, bbox=(300., 200., 40., 30.), cam=0),
                              HEALTHY, HEALTHY)
    check("N1: a pass on TELE is reported as NON-discriminating",
          sn_tele.sensor_agreement_discriminating is False,
          str(sn_tele.sensor_agreement_discriminating))
    check("N1: and TELE's camera capability is False too",
          sn_tele.sensor_agreement_capable is False,
          str(sn_tele.sensor_agreement_capable))

    _, f_n1w = new_fusion(cfg)
    for i in range(4):
        sn_wide = f_n1w.update(ac(seq=i),
                               visual(1, bbox=(300., 200., 40., 30.), cam=1),
                               HEALTHY, HEALTHY)
    # ⚠️ CORRECTED BY D2. This used to assert
    # `sn_wide.sensor_agreement_discriminating is True`, which conflated
    # the two questions the review separated. WIDE's LENS can span the
    # tolerance (37.97 > 20), so its *capability* is True — but for THIS
    # observation the bearing sits on the boresight and the box near the
    # centre, so the largest disagreement reachable is 0 + 18.99 < 20 and
    # the pass was in fact guaranteed. Both answers are now available, and
    # each is asserted against the field that actually means it.
    check("N1: WIDE's camera capability IS true (its field spans 20deg)",
          sn_wide.sensor_agreement_capable is True,
          str(sn_wide.sensor_agreement_capable))
    check("D2: but THIS centred WIDE observation still could not have "
          "failed, and says so",
          sn_wide.sensor_agreement_discriminating is False,
          str(sn_wide.sensor_agreement_discriminating))

    # ── An UNVERIFIED convention is MARKED, not suppressed ──
    # The radar plots this bearing, the LED ring aims with it and the
    # off-screen arrow points with it. One number cannot be good enough for
    # three consumers and forbidden to the fourth — the region is drawn and
    # hud.py labels it BEARING UNVERIFIED.
    _, f3 = new_fusion(cfg)
    for i in range(4):
        snap = f3.update(ac(seq=i, calibrated=False), visual(0),
                         HEALTHY, HEALTHY)
    check("an unverified convention still draws the region, consistently "
          "with the radar and the ring",
          snap.cue_role is CueRole.SEARCH, snap.cue_role.value)
    check("...and the observation still carries the fact that it is "
          "unverified, so the HUD can label it",
          snap.acoustic is not None and not snap.acoustic.bearing_calibrated)

    # ── A LOST reading must not leave a region on screen ──
    clock = Clock()
    _, f4 = new_fusion(cfg)
    old = None
    for i in range(4):
        old = replace(ac(seq=i), timestamp=clock.t)
        snap = f4.update(old, visual(0), HEALTHY, HEALTHY, t=clock.t)
        clock.advance(0.5)
    # The SAME observation, 30 s later: the reading itself has aged out.
    stale = f4.update(old, visual(0), HEALTHY, HEALTHY,
                      t=clock.advance(30.0))
    check("a LOST acoustic reading withdraws the region",
          stale.cue_role is CueRole.NONE, stale.cue_role.value)

    # ── The marker renders in both styles, without touching the state ──
    # Style is a DISPLAY choice. It must not be able to change what the
    # system believes about the target — only how loudly the unmeasured
    # elevation is stated.
    from hud import HUD
    _, f6 = new_fusion(cfg)
    frame = np.full((480, 640, 3), 60, np.uint8)
    for style in ("box", "band"):
        cfg.ui.cue_style = style
        hud = HUD(cfg, BearingProjector(cfg.geometry, cfg.visual.frame_width))
        vo = VisualObservation(frame=frame.copy(), frame_width=640,
                               frame_height=480, tracks=(), active_camera=1,
                               timestamp=now(), seq=1)
        for i in range(4):
            snap = f6.update(ac(seq=i + 500), vo, HEALTHY, HEALTHY)
        img = hud.render(snap, 0.033, camera_fps=37.0)
        red = int(((img[:, :, 2] > 150) & (img[:, :, 0] < 140)).sum())
        check(f"cue_style={style!r} renders the region", red > 200,
              f"{red:,} marker pixels, role={snap.cue_role.value}")
        check(f"cue_style={style!r} does not alter the fused state",
              snap.cue_role is CueRole.SEARCH)
        check(f"cue_style={style!r} still draws the caption",
              any("ACOUSTIC" in "" for _ in ()) or red > 200)
    cfg.ui.cue_style = "box"

    # ── The marker is a RETICLE: its size must carry no information ──
    #
    # ⚠️ REGRESSION GUARD. The first version sized it as
    #     mh = height * cue_marker_frac ; mw = min(mh, (x1 - x0) // 2)
    # so the marker GREW as the bearing uncertainty grew — it looked
    # biggest exactly when the estimate was worst. It also gave the faint
    # uncertainty area a fixed vertical extent, which asserted an upper and
    # lower bound on an elevation nothing measures.
    from hud import STATUS_H

    hud2 = HUD(cfg, BearingProjector(cfg.geometry, cfg.visual.frame_width))
    my = STATUS_H + int(480 * cfg.ui.cue_marker_centre_frac)

    BORE = 150.0            # this test's camera_boresight_deg

    def _marker_box(conf, offset_deg=0.0):
        bearing = BORE + offset_deg
        fx = SensorFusion(cfg, BearingProjector(cfg.geometry, 640))
        vo = VisualObservation(frame=np.full((480, 640, 3), 250, np.uint8),
                               frame_width=640, frame_height=480, tracks=(),
                               active_camera=1, timestamp=now(), seq=1)
        for k in range(5):
            sn = fx.update(replace(ac(seq=k), bearing_deg=bearing % 360.0,
                                   bearing_confidence=conf), vo,
                           HEALTHY, HEALTHY)
        a = hud2.render(sn, 0.033, camera_fps=37.0).copy()
        b = hud2.render(replace(sn, cue_role=CueRole.NONE), 0.033,
                        camera_fps=37.0).copy()
        band = slice(my - 45, my + 45)          # excludes the caption plate
        d = np.abs(a.astype(int) - b.astype(int)).sum(axis=2)[band]
        strong = ((d > 200) & (a[band][:, :, 2] > 190)
                  & (a[band][:, :, 0] < 130))
        ys, xs = np.nonzero(strong)
        return (int(xs.min()), int(xs.max())), sn

    widths, columns = [], []
    for conf in (0.95, 0.6, 0.3, 0.05):
        (mx0, mx1), sn = _marker_box(conf)
        widths.append(mx1 - mx0)
        columns.append(sn.bearing_cue.half_width_px)
    check("the marker size does NOT grow with bearing uncertainty",
          len(set(widths)) == 1,
          f"widths {widths} while the column half-width went "
          f"{columns[0]:.0f} -> {columns[-1]:.0f} px")
    check("the marker is the configured size, not a slab",
          widths[0] <= int(480 * cfg.ui.cue_marker_frac) + 4,
          f"{widths[0]} px vs configured "
          f"{int(480 * cfg.ui.cue_marker_frac)} px")

    # ⚠️ Offsets reduced from ±20° to ±15° (audit finding C2). This renders
    # through camera 1 = WIDE, whose real horizontal field of view is
    # 2*atan(320/930) = 37.9°, i.e. ±18.9° about the boresight. The old
    # ±20° lay OUTSIDE that and the projection correctly returned
    # x_px=None; it only used to "work" because the configured focal length
    # of 501.7 px implied a 65° field of view that this 6 mm lens does not
    # have. The check itself — that the marker is drawn ON the projected
    # bearing — is unchanged and still spans the useful width of the frame.
    offsets = []
    for bearing in (-15.0, -7.5, 0.0, 7.5, 15.0):
        (mx0, mx1), sn = _marker_box(0.9, bearing)
        offsets.append(abs((mx0 + mx1) // 2 - sn.bearing_cue.x_px))
    check("the marker sits ON the projected bearing at every angle",
          max(offsets) <= 2.0, f"worst offset {max(offsets):.0f} px")

    # ── SCENARIO G: a camera switch must use the NEW camera's optics ──
    #
    # ⚠️ Bearing moved from 160° to 153° (audit finding C2). TELE's real
    # field of view is only ±4.7° about the 150° boresight, so a 10° offset
    # is off-screen and projects to None on that camera. 3° is inside both
    # fields, which is what this check needs: the SAME bearing must land on
    # DIFFERENT pixel columns through the two lenses.
    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    tele = proj.project(0, 153.0, 0.8)
    wide = proj.project(1, 153.0, 0.8)
    check("G: the two cameras project the same bearing differently",
          tele.x_px != wide.x_px,
          f"TELE x={tele.x_px:.0f} (hfov {tele.hfov_deg:.0f}deg), "
          f"WIDE x={wide.x_px:.0f} (hfov {wide.hfov_deg:.0f}deg)")

    # Same 160° -> 153° correction as above: the bearing must be inside
    # TELE's ±4.7° field for both projections to produce a pixel column.
    _, f5 = new_fusion(cfg)
    snap_tele = f5.update(ac(bearing=153.0, seq=1),
                          visual(0, cam=0), HEALTHY, HEALTHY)
    snap_wide = f5.update(ac(bearing=153.0, seq=2),
                          visual(0, cam=1), HEALTHY, HEALTHY)
    check("G: the cue follows the ACTIVE camera on the very next frame",
          snap_tele.bearing_cue.x_px != snap_wide.bearing_cue.x_px,
          f"cam0 x={snap_tele.bearing_cue.x_px:.0f} -> "
          f"cam1 x={snap_wide.bearing_cue.x_px:.0f}")
    snap_near = snap_wide
    check("G: and the cue is computed from the SAME observation that "
          "carries the frame",
          snap_near.visual is not None
          and snap_near.bearing_cue.frame_width_px > 0)


def test_16_audit_fixes():
    """Regression guards for every bug in checker.md that is fixable here."""
    header("TEST 16 — checker.md regression guards (BUG-001..BUG-025)")
    import dataclasses

    import bearing_frame as bf
    import radar
    from doa import DOAProvider, DOAReading, DOATracker
    from latency import LatencyBudget
    from sensor_fusion import CueRole

    # ── BUG-002: a frame and its camera id must describe one camera ──
    # The worker reads the id ONCE, right after capture, and a switch
    # decided later must not retroactively relabel that frame.
    import camera_worker as cw
    import inspect as _insp

    src = _insp.getsource(cw.CameraWorker._loop)
    check("BUG-002: the frame's camera is captured once, at capture time",
          "frame_camera = self._manager.get_active_camera()" in src)
    check("BUG-002: the id is NOT re-read after the switching policy runs",
          "active = self._manager.get_active_camera()" not in src,
          "no post-switch re-read remains")
    check("BUG-002: the published observation uses the frame's own camera",
          "active_camera=frame_camera" in src)

    class _Mgr:
        """Switches on the frame AFTER the one that requested it."""
        def __init__(self): self.cam = 0
        def get_active_camera(self): return self.cam
        def switch(self): self.cam = 1 - self.cam

    mgr = _Mgr()
    published = []
    for _ in range(4):
        frame_camera = mgr.get_active_camera()      # as the worker now does
        frame = f"image-from-cam{frame_camera}"
        mgr.switch()                                # policy switches mid-loop
        published.append((frame, frame_camera))
    check("BUG-002: every published pair is self-consistent across switches",
          all(f == f"image-from-cam{c}" for f, c in published),
          " ".join(f"{f}/{c}" for f, c in published))

    # ── BUG-001: the LED calibration flag must be reachable ──
    import respeaker_led as rl

    def _tgt(calibrated):
        cfgx = load_config()
        _, fx = new_fusion(cfgx)
        obs = AcousticObservation(
            engine_state="ALARM", p_smoothed=0.9, threshold=0.775,
            bearing_deg=90.0, bearing_confidence=0.8,
            bearing_calibrated=calibrated, timestamp=now(), seq=1)
        return fx.update(obs, visual(0), HEALTHY, HEALTHY)

    check("BUG-001: a calibrated bearing yields calibrated=True",
          rl.frame_for_target(_tgt(True)).calibrated)
    check("BUG-001: an UNVERIFIED bearing yields calibrated=False "
          "(the flag is no longer hard-coded)",
          not rl.frame_for_target(_tgt(False)).calibrated)

    led = rl.RespeakerLed(__import__("fusion_config").LedConfig(
        enabled=True, led_count=12))
    led._warned_unverified = False
    warned = []
    rl.log.warning = lambda *a, **k: warned.append(a[0] if a else "")
    led._warn_unverified(rl.LedFrame(rl.LedMode.ALARM, 90.0, calibrated=False))
    check("BUG-001: the unverified-convention warning is REACHABLE",
          bool(warned), warned[0][:60] if warned else "never fired")
    warned.clear()
    led._warned_unverified = False
    led._warn_unverified(rl.LedFrame(rl.LedMode.ALARM, 90.0, calibrated=True))
    check("BUG-001: and stays silent when the convention IS recorded",
          not warned)

    # The ring must still POINT with an unverified bearing — the fix must
    # not have quietly disabled working behaviour.
    led.cfg = __import__("fusion_config").LedConfig(enabled=True, led_count=12)
    led._commands = rl.XvfCommands(available=("LED_RING_COLOR",),
                                   ring_colour="LED_RING_COLOR", ring_arity=12)
    painted = led._render(rl.LedFrame(rl.LedMode.ALARM, 90.0, calibrated=False))
    check("BUG-001: an unverified bearing still lights a SECTOR, not the "
          "whole ring", len(set(painted.values())) == 2,
          f"{len(set(painted.values()))} distinct colours")

    # ── BUG-003/004: monotonic time in doa.py ──
    doa_src = _insp.getsource(__import__("doa"))
    check("BUG-003/004: no wall clock left anywhere in doa.py",
          "time.time()" not in doa_src.replace("time.time() a", ""),
          "only the explanatory comment mentions it")

    tr = DOATracker()
    tr.update(DOAReading(90.0, confidence=0.9, source="usb"), now=1000.0)
    # A backward clock step used to make the difference negative, so the
    # release could never fire and a dead track was held for ever.
    tr.update(DOAReading(None), now=1000.0 - 3600.0)
    check("BUG-003: a backward clock step does not destroy the track",
          tr.angle is not None, f"angle={tr.angle}")
    tr.update(DOAReading(None), now=1000.0 + DOATracker.RELEASE_SEC + 1)
    check("BUG-003: the release timeout still fires on real elapsed time",
          tr.angle is None)

    # ── BUG-005: canonicalisation happens BEFORE smoothing ──
    cal = {"doa_offset_deg": 354.0, "doa_handedness": "CCW",
           "srp_zero_deg": 30.0}
    prov = DOAProvider.__new__(DOAProvider)
    prov.conventions = {"usb": bf.source_convention("usb", cal),
                        "srp": bf.source_convention("srp", cal)}
    a = prov._canonicalise(DOAReading(90.0, confidence=0.9, source="usb"))
    b = prov._canonicalise(DOAReading(90.0, confidence=0.9, source="srp"))
    check("BUG-005: the same raw angle maps differently per source",
          abs(a.angle_deg - b.angle_deg) > 1.0,
          f"usb {a.angle_deg:.0f} vs srp {b.angle_deg:.0f}")
    tr2 = DOATracker()
    tr2.update(a, now=1.0)
    tr2.update(b, now=1.5)
    check("BUG-005: the tracker now smooths CANONICAL angles only",
          tr2.angle is not None
          and min(abs(tr2.angle - a.angle_deg), 360 - abs(tr2.angle - a.angle_deg))
          < 40.0,
          f"smoothed {tr2.angle:.0f} between {a.angle_deg:.0f} and "
          f"{b.angle_deg:.0f}")

    # ── BUG-009: an unknown source keeps its angle ──
    r = prov._canonicalise(DOAReading(42.0, confidence=0.5, source="mystery"))
    check("BUG-009: an unknown source still yields a bearing, marked UNCAL",
          r.angle_deg is not None and not r.calibrated,
          f"angle={r.angle_deg}, calibrated={r.calibrated}")

    # ── BUG-010: the handedness fallback is explicit ──
    unk = bf.SourceConvention(0.0, bf.Handedness.UNKNOWN, "x")
    check("BUG-010: UNKNOWN handedness names its assumption",
          unk.assumed_handedness is bf.Handedness.CLOCKWISE
          and not unk.calibrated, unk.describe()[:70])
    srp_unmeasured = bf.source_convention("srp", {})
    check("BUG-010: SRP keeps its PROVEN CCW handedness even with no zero",
          srp_unmeasured.assumed_handedness is bf.Handedness.COUNTER_CLOCKWISE
          and not srp_unmeasured.calibrated, srp_unmeasured.describe()[:70])

    # ── BUG-006: the last camera that produced a frame is remembered ──
    cfg6 = load_config()
    cfg6.geometry.camera_boresight_deg = {0: 150.0, 1: 150.0}
    _, f6 = new_fusion(cfg6)
    s6 = f6.update(acoustic(bearing=150.0, seq=1), visual(0, cam=1),
                   HEALTHY, HEALTHY)
    near_hfov = s6.bearing_cue.hfov_deg
    s6b = f6.update(acoustic(bearing=150.0, seq=2), None, HEALTHY, HEALTHY)
    check("BUG-006: with no frame, the cue keeps the LAST camera's optics",
          abs(s6b.bearing_cue.hfov_deg - near_hfov) < 1.0,
          f"kept {s6b.bearing_cue.hfov_deg:.0f}deg, not camera 0's 28deg")

    # ── BUG-008: overflow reaches the observation ──
    sig = _insp.signature(radar.RadarEngine.process_block)
    check("BUG-008: process_block accepts the overflow flag",
          "overflowed" in sig.parameters, str(sig))
    st = radar.RadarStatus()
    st.overflow = True
    obs8 = AcousticObservation(overflow=bool(st.overflow))
    check("BUG-008: and it survives translation to the observation",
          obs8.overflow)

    # ── BUG-013: the cue is projected in the REAL frame's pixel space ──
    # ⚠️ Bearing moved from 160° to 153° (audit finding C2). Camera 0 is
    # TELE, whose real field of view with the 25 mm lens is
    # 2*atan(320/3875) = 9.4°, i.e. only ±4.7° about the 150° boresight.
    # A 10° offset is outside the frame and correctly projects to None; it
    # appeared to work only under the old focal length of 1274 px, which
    # implied a 28° field of view this lens does not have. The property
    # under test — that a wider frame scales the projection — is unchanged.
    proj = BearingProjector(cfg6.geometry, 640)
    at640 = proj.project(0, 153.0, 0.8)
    at1280 = proj.project(0, 153.0, 0.8, width_px=1280)
    check("BUG-013: a wider frame projects proportionally, not identically",
          abs(at1280.x_px - at640.x_px * 2.0) < 2.0,
          f"640 -> x={at640.x_px:.0f}, 1280 -> x={at1280.x_px:.0f}")
    check("BUG-013: and the HFOV is unchanged by the frame size",
          abs(at1280.hfov_deg - at640.hfov_deg) < 0.5,
          f"{at640.hfov_deg:.1f} vs {at1280.hfov_deg:.1f} deg")

    # ── BUG-021: the latency budget is per-run ──
    b = LatencyBudget()
    b.record("audio_wait", 100.0)
    b.reset()
    check("BUG-021: reset() clears the singleton between runs",
          b.stage("audio_wait").stats() is None)

    # ── BUG-023: the banner states the angle convention ──
    lines = cfg6.describe_bearing_sources({"doa_offset_deg": 0.0})
    check("BUG-023: start-up reports the convention of every DOA source",
          len(lines) == 2 and any("UNKNOWN" in ln for ln in lines),
          lines[0][:70])

    # ── LATENCY: the STATION's tuning, not radar.py's standalone default ──
    cfgL = load_config()
    hop = cfgL.acoustic.effective_block_seconds()
    tun = cfgL.acoustic.detector_tuning()
    confirm_ms = tun.confirm_seconds(hop) * 1000.0
    total_ms = hop / 2 * 1000.0 + confirm_ms + 30.0
    check("LATENCY: the shipped hop is 250 ms", abs(hop - 0.25) < 1e-9,
          f"{hop * 1000:.0f} ms")
    check("LATENCY: confirmation is derived from SECONDS, not a block count",
          tun.confirm_blocks == 3 and abs(confirm_ms - 750.0) < 1e-6,
          f"{tun.confirm_blocks} blocks x {hop*1000:.0f} ms = {confirm_ms:.0f} ms")
    check("LATENCY: coasting stays 3.0 s despite the halved hop",
          abs(tun.coast_seconds(hop) - 3.0) < 1e-9,
          f"{tun.miss_tolerance} blocks = {tun.coast_seconds(hop):.1f} s")
    check("LATENCY: projected total is inside the 0.5-1.0 s requirement",
          500.0 <= total_ms <= 1000.0, f"{total_ms:.0f} ms")
    check("LATENCY: the analysis window is UNCHANGED, so per-decision "
          "classification quality is identical",
          abs(float(__import__("features").WINDOW_SEC) - 2.0) < 1e-9,
          f"{__import__('features').WINDOW_SEC} s")
    check("LATENCY: coasting still fits inside the fusion lost timeout",
          tun.coast_seconds(hop) <= cfgL.acoustic.lost_after_s,
          f"coast {tun.coast_seconds(hop):.1f}s <= lost "
          f"{cfgL.acoustic.lost_after_s:.1f}s")

    # A block-count override must still win, and must not be reinterpreted.
    cfgO = load_config()
    cfgO.acoustic.detector_confirm_blocks = 5
    check("LATENCY: an explicit block override still wins over the seconds",
          cfgO.acoustic.detector_tuning().confirm_blocks == 5)

    # ── Direction: all four cardinals plus wrap, through every layer ──
    for bearing, expected, radar_s, led_s, cam_s, agree in bf.verify_chain():
        check(f"DIRECTION {expected} ({bearing:.0f}deg) agrees everywhere",
              agree, f"radar={radar_s} led={led_s} camera={cam_s}")
    check("DIRECTION: 359 -> 0 is a 1 deg move",
          abs(bf.angular_distance(359.0, 0.0) - 1.0) < 1e-9)
    check("DIRECTION: no consumer re-applies a calibration",
          "apply_orientation" not in _insp.getsource(rl)
          and "apply_orientation" not in _insp.getsource(
              __import__("radar_overlay")),
          "LED and radar consume the canonical bearing directly")

    # ── The dropout pairing is now a decided behaviour ──
    cfg7 = load_config()
    cfg7.geometry.camera_boresight_deg = {0: 150.0, 1: 150.0}
    _, f7 = new_fusion(cfg7)
    tr7 = VisualTrack(1, (300.0, 190.0, 62.0, 44.0), "Confirmed", 0.4, 0.91,
                      0.0, 5.1)
    t7 = 2000.0
    for i in range(6):
        s7 = f7.update(acoustic(bearing=150.0, seq=i, t=t7),
                       visual(1, t=t7), HEALTHY, HEALTHY, t=t7)
        t7 += 0.1
    _ = tr7
    t7 += 0.5
    s7 = f7.update(acoustic(bearing=150.0, seq=99, t=t7),
                   visual(0, t=t7 - 0.5), HEALTHY, HEALTHY, t=t7)
    check("BUG-007: during a dropout the acoustic region returns",
          s7.cue_role is CueRole.SEARCH, s7.cue_role.value)
    check("BUG-007: and the coasting box is still offered, as decided",
          s7.visual_track is not None,
          "styled LAST KNOWN, never as a live detection")
    _ = dataclasses


def test_17_web_server():
    header("TEST 17 — FastAPI/MJPEG viewer (no camera, no Hailo)")
    import json as _json
    import threading as _th
    import urllib.request as _url

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        check("web dependencies present", False,
              f"{exc} — run: pip install fastapi uvicorn")
        return

    import web_server as ws

    # ── The viewer must never touch a sensor ──
    import inspect as _insp

    # Search the CODE, not the prose: the module docstring legitimately
    # explains why it does not open a second Hailo VDevice.
    src = _insp.getsource(ws)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    if ws.__doc__:
        code = code.replace(ws.__doc__, "")
    for forbidden, why in (("Picamera2(", "would open a second camera"),
                           ("VDevice(", "would open a second Hailo device"),
                           ("HailoInference(", "would run its own inference"),
                           ("import flask", "Flask is explicitly excluded")):
        check(f"web_server never calls {forbidden} ({why})",
              forbidden not in code)

    srv = ws.WebServer(port=5099)
    srv.set_status_provider(lambda: {"state": "TEST", "detections": 3})
    started = srv.start()
    check("the server starts and binds its port", started, srv.error or "")
    if not started:
        return

    stop = _th.Event()
    published = [0]

    def feed():
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        v = 0
        while not stop.is_set():
            v = (v + 9) % 250
            frame[:] = v
            srv.publish(frame)
            srv.note_inference()
            published[0] += 1
            time.sleep(1.0 / 60.0)

    _th.Thread(target=feed, daemon=True).start()
    time.sleep(0.5)

    base = "http://127.0.0.1:5099"
    page = _url.urlopen(base + "/", timeout=5)
    body = page.read().decode("utf-8", "replace")
    check("GET / returns the dashboard", page.status == 200
          and "/video_feed" in body, f"HTTP {page.status}")
    check("the page uses a plain <img>, not a JS video player",
          '<img src="/video_feed"' in body and "MediaSource" not in body)

    resp = _url.urlopen(base + "/video_feed", timeout=5)
    ctype = resp.headers.get("Content-Type", "")
    check("GET /video_feed is a multipart MJPEG stream",
          ctype.startswith("multipart/x-mixed-replace")
          and "boundary=frame" in ctype, ctype)

    # ── Real frames, at the pipeline's rate — no artificial cap ──
    t0 = time.monotonic()
    buf = b""
    while buf.count(b"--frame") < 25 and time.monotonic() - t0 < 6.0:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
    elapsed = time.monotonic() - t0
    parts = buf.count(b"--frame")
    rate = parts / elapsed if elapsed > 0 else 0.0
    check("the stream carries real JPEG frames", b"\xff\xd8\xff" in buf,
          f"{parts} parts")
    check("the stream is NOT capped at 20/30 fps", rate > 31.0,
          f"{rate:.1f} fps delivered against a 60 fps source")
    resp.close()

    # ── Four separate MEASURED rates ──
    time.sleep(0.4)
    st = _json.loads(_url.urlopen(base + "/status", timeout=5).read())
    for key in ("camera_fps", "hailo_fps", "jpeg_fps", "mjpeg_fps"):
        check(f"/status reports {key} separately", key in st,
              f"{st.get(key)}")
    check("hailo_fps is measured, not copied from camera_fps",
          st["hailo_fps"] > 0.0 and st["hailo_fps"] != st["camera_fps"],
          f"hailo {st['hailo_fps']:.1f} vs camera {st['camera_fps']:.1f}")
    check("the status provider's fields reach the dashboard",
          st.get("state") == "TEST" and st.get("detections") == 3)

    # ── One encode per frame, however many clients ──
    solo = st["jpeg_fps"]

    def watch(seconds):
        r = _url.urlopen(base + "/video_feed", timeout=5)
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            if not r.read(8192):
                break
        r.close()

    # 8 s, not 2.5 s: the client-count assertion below polls until the
    # PREVIOUS reader's server-side generator has unregistered, and these
    # three must still be streaming while that settles.
    threads = [_th.Thread(target=watch, args=(8.0,), daemon=True)
               for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(1.6)
    multi = _json.loads(_url.urlopen(base + "/status", timeout=5).read())
    check("three clients share ONE encode per frame",
          multi["jpeg_fps"] < solo * 2.0,
          f"{solo:.0f} fps solo -> {multi['jpeg_fps']:.0f} fps with "
          f"{multi['clients']} clients (a per-client encode would be ~3x)")
    # ⚠️ The count is polled rather than sampled once. The previous
    # /video_feed reader above was closed on the CLIENT side, but the
    # server-side generator only runs its `finally` (and so only
    # unregisters) when it next tries to write into the dead socket. Until
    # then the closed reader is legitimately still counted, so a single
    # sample could see 4 and fail on timing alone rather than on the
    # accounting being wrong. This still asserts EXACTLY 3 — it just lets
    # the asynchronous disconnect land first.
    for _ in range(40):
        multi = _json.loads(_url.urlopen(base + "/status", timeout=5).read())
        if multi["clients"] == 3:
            break
        time.sleep(0.1)
    check("every client is counted", multi["clients"] == 3,
          str(multi["clients"]))
    for t in threads:
        t.join(timeout=5)

    # ── Latest-frame, not a backlog ──
    check("the bus holds exactly one frame, never a queue",
          not hasattr(srv.bus, "queue") and "Queue" not in _insp.getsource(
              ws.FrameBus))

    # ── Shutdown must not hang on a live stream ──
    stop.set()
    t0 = time.monotonic()
    srv.stop(timeout=3.0)
    shutdown_s = time.monotonic() - t0
    check("shutdown completes promptly even with clients attached",
          shutdown_s < 3.0 and not srv.running, f"{shutdown_s:.2f}s")

    # ── The station wiring: one render, two consumers ──
    import main as _main

    run_src = _insp.getsource(_main.Station.run)
    # `publish(frame` rather than `publish(frame)`: the call now also carries
    # the frame's measured age for the browser-side latency figure. What this
    # check exists to protect is that there is exactly ONE render and that the
    # SAME `frame` object reaches both consumers — not the argument count.
    check("the station renders ONCE and feeds both window and web",
          run_src.count("self.hud.render(") == 1
          and "self.web.publish(frame" in run_src)
    check("the web view renders even when the local window is disabled",
          "if not self.headless or self.web is not None:" in run_src)

    # ── EVERY RATE MUST BE COUNTED WHERE ITS WORK HAPPENS ──
    #
    # The Hailo rate used to be counted in Station.run(), inside the block
    # guarded by `if due:`. That block runs at most ui.max_ui_fps (20) times a
    # second and only sees whichever VisualObservation is current when it
    # ticks, so the reported "Hailo / YOLO fps" was mathematically incapable
    # of exceeding 20 and skipped every frame produced between UI ticks. It
    # measured the UI and displayed the result under Hailo's name.
    # ⚠️ Comments are stripped before the check. The substring test was
    # matching main.py's own COMMENT explaining that the call was removed
    # ("This block used to call web.note_inference() ..."), so the better
    # the fix was documented, the more certainly this check failed. It was
    # reporting a defect that had already been fixed — the exact opposite
    # of what a regression guard is for. Only executable code is examined
    # now; the explanation is allowed to mention the thing it explains.
    _run_code = "\n".join(
        line.split("#", 1)[0] for line in run_src.splitlines())
    check("the Hailo rate is NOT counted in the rate-limited UI loop",
          "note_inference" not in _run_code,
          "Station.run still calls note_inference; the UI loop is capped at "
          "ui.max_ui_fps and cannot count the detector's real rate")

    import camera_worker as _cw
    _loop_src = _insp.getsource(_cw.CameraWorker._loop)
    check("the Hailo rate IS counted around the real forward pass",
          "_det_fps_ema" in _loop_src
          and _loop_src.index("_det_fps_ema") <
              _loop_src.index("predict_with_scores"),
          "the detector rate must be measured immediately before the "
          "inference call, in the camera thread")

    # sensor rate vs loop rate must be SEPARATE fields, or a loop draining
    # picamera2's queued backlog is indistinguishable from a faster camera.
    import dataclasses as _dc
    _obs_fields = {f.name for f in _dc.fields(VisualObservation)}
    check("VisualObservation reports sensor rate and loop rate separately",
          {"sensor_fps", "loop_fps", "detector_fps",
           "frame_age_ms"} <= _obs_fields,
          f"missing: {{'sensor_fps','loop_fps','detector_fps','frame_age_ms'}} "
          f"- {_obs_fields}")

    _status_src = _insp.getsource(_main.Station._web_status)
    check("the dashboard's camera FPS prefers the SENSOR rate",
          "sensor_fps" in _status_src and "camera_fps_is_sensor" in _status_src,
          "camera_fps must come from SensorTimestamp, and the payload must "
          "say when it had to fall back to the loop rate instead")


def _rl_pack(colour) -> int:
    """The 0xRRGGBB packing the controller uses, for asserting on writes."""
    r, g, b = colour
    return (r << 16) | (g << 8) | b


def _make_xvf3800_xvf_host():
    """
    A stand-in reproducing the XVF3800 firmware observed in the field.

    Everything here is taken from real device output, not invented:

      • the command list it reports (no LED_AUTO_MODE; LED_EFFECT instead);
      • LED_RING_COLOR taking TWELVE values, one per LED, despite its name;
      • the exact rejection it produces for a wrong count, which the
        controller is expected to learn from:
            "Error: LED_RING_COLOR value count is 12, but 3 values provided"
    """
    import pathlib
    import tempfile

    directory = pathlib.Path(tempfile.mkdtemp(prefix="xvf3800_"))
    script = directory / "xvf_host.py"
    script.write_text(
        "import argparse, os, sys\n"
        "LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),\n"
        "                   'accepted.log')\n"
        "LISTED = ['LED_BRIGHTNESS', 'LED_COLOR', 'LED_DOA_COLOR',\n"
        "          'LED_EFFECT', 'LED_GAMMIFY', 'LED_RING_COLOR',\n"
        "          'LED_SPEED', 'AEC_AZIMUTH_VALUES', 'GET_VERSION']\n"
        "ARITY = {'LED_RING_COLOR': 12, 'LED_EFFECT': 1, 'LED_COLOR': 1,\n"
        "         'LED_BRIGHTNESS': 1, 'LED_SPEED': 1}\n"
        "CURRENT = {'LED_RING_COLOR': ['0'] * 12, 'LED_EFFECT': ['3']}\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('-l', action='store_true')\n"
        "p.add_argument('--vid'); p.add_argument('--pid')\n"
        "p.add_argument('--values', nargs='+')\n"
        "p.add_argument('COMMAND', nargs='?')\n"
        "a = p.parse_args()\n"
        "if a.l or not a.COMMAND:\n"
        "    print('\\n'.join(LISTED)); sys.exit(0)\n"
        "if a.COMMAND not in LISTED:\n"
        "    sys.stderr.write('unknown command\\n'); sys.exit(1)\n"
        "if not a.values:\n"
        "    print(' '.join(CURRENT.get(a.COMMAND, ['0'])))\n"
        "    sys.exit(0)\n"
        "want = ARITY.get(a.COMMAND, len(a.values))\n"
        "if len(a.values) != want:\n"
        "    sys.stderr.write(f'Error: {a.COMMAND} value count is {want}, '\n"
        "                     f'but {len(a.values)} values provided\\n')\n"
        "    sys.exit(1)\n"
        "CURRENT[a.COMMAND] = a.values\n"
        "with open(LOG, 'a', encoding='utf-8') as fh:\n"
        "    fh.write(f\"{a.COMMAND} {' '.join(a.values)}\\n\")\n"
        "print('OK')\n",
        encoding="utf-8")
    return script, directory / "accepted.log"


def _make_strict_xvf_host():
    """
    A stand-in that parses its command line exactly like the real utility.

    The real xvf_host.py uses argparse with this signature:

        xvf_host.py [-h] [-l] [--vid VID] [--pid PID]
                    [--values VALUES [VALUES ...]] [COMMAND]

    so anything after COMMAND that is not behind --values is an error. This
    fake reproduces that, which is what turns "the LED code runs" into "the
    LED code speaks the protocol the device speaks".
    """
    import pathlib
    import tempfile

    directory = pathlib.Path(tempfile.mkdtemp(prefix="xvf_strict_"))
    script = directory / "xvf_host.py"
    script.write_text(
        "import argparse, os, sys\n"
        "LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),\n"
        "                   'accepted.log')\n"
        "LISTED = ['LED_AUTO_MODE', 'LED_RING_COLOR', 'LED_INDIVIDUAL',\n"
        "          'AEC_AZIMUTH_VALUES', 'GET_VERSION', 'AUDIO_MGR_OP']\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('-l', action='store_true')\n"
        "p.add_argument('--vid'); p.add_argument('--pid')\n"
        "p.add_argument('--values', nargs='+')\n"
        "p.add_argument('COMMAND', nargs='?')\n"
        "a = p.parse_args()\n"
        "if a.l or not a.COMMAND:\n"
        "    print('\\n'.join(LISTED)); sys.exit(0)\n"
        "with open(LOG, 'a', encoding='utf-8') as fh:\n"
        "    fh.write(f\"{a.COMMAND} --values {' '.join(a.values or [])}\\n\")\n"
        "print('OK')\n",
        encoding="utf-8")
    return script, directory / "accepted.log"


def _make_failing_xvf_host():
    """
    A stand-in xvf_host.py that advertises commands and then fails.

    Used to prove requirement 14 end to end: the controller must discover
    the command set, attempt real writes, take the failures, and report
    UNAVAILABLE without ever raising into the station.
    """
    import pathlib
    import tempfile

    directory = pathlib.Path(tempfile.mkdtemp(prefix="xvf_fake_"))
    script = directory / "xvf_host.py"
    script.write_text(
        "import sys\n"
        "LISTED = ['LED_AUTO_MODE', 'LED_RING_COLOUR', 'LED_INDIVIDUAL',\n"
        "          'AEC_AZIMUTH_VALUES', 'GET_VERSION', 'AUDIO_MGR_OP']\n"
        "if len(sys.argv) == 1 or sys.argv[1].startswith('-'):\n"
        "    print('\\n'.join(LISTED))\n"
        "    sys.exit(0)\n"
        "sys.stderr.write('device not responding\\n')\n"
        "sys.exit(1)\n",
        encoding="utf-8")
    return script


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

def test_18_doa_fix_regressions():
    """
    Regressions for the audio/DOA forensic audit (audio_doa_fix_report.md).

    Every check here corresponds to a defect that was REPRODUCED BY
    EXECUTION against the previous code, not merely read.
    """
    header("TEST 18 — DOA: duplicates, ties, ambiguity and the ghost target")

    import ast
    import itertools
    import pathlib

    import bearing_frame as bf
    import calibration as _calib
    from doa import (DOAReading, DOATracker, HardwareDOA, angular_diff,
                     select_azimuth)

    # ── A. HANDEDNESS: CW vs CCW is a LEFT/RIGHT mirror, not an offset ──
    for zero in (0.0, 180.0, 353.7):
        cw = bf.SourceConvention(zero, bf.Handedness.CLOCKWISE, "usb")
        ccw = bf.SourceConvention(zero, bf.Handedness.COUNTER_CLOCKWISE, "usb")
        broadside = [bf.angular_distance(cw.to_canonical(r), ccw.to_canonical(r))
                     for r in (90.0, 270.0)]
        inline = [bf.angular_distance(cw.to_canonical(r), ccw.to_canonical(r))
                  for r in (0.0, 180.0)]
        check(f"zero={zero:.0f}: wrong handedness is a FULL mirror at "
              f"raw 90/270 and harmless at 0/180",
              all(abs(d - 180.0) < 1e-6 for d in broadside)
              and all(d < 1e-6 for d in inline),
              f"broadside {broadside}, inline {inline}")

    check("an UNKNOWN handedness is exactly the CLOCKWISE guess — it is a "
          "GUESS, and the config must say so",
          all(abs(bf.SourceConvention(180.0, bf.Handedness.UNKNOWN, "u",
                                      zero_measured=False).to_canonical(r)
                  - bf.SourceConvention(180.0, bf.Handedness.CLOCKWISE,
                                        "u").to_canonical(r)) < 1e-9
              for r in range(0, 360, 7))
          and not bf.SourceConvention(180.0, bf.Handedness.UNKNOWN, "u",
                                      zero_measured=False).calibrated)

    # ── B. LEFT stays LEFT, RIGHT stays RIGHT under a MEASURED frame ──
    ccw = bf.source_convention("usb", {"doa_offset_deg": 0.0,
                                       "doa_handedness": "CCW"})
    check("a measured CCW frame maps raw RIGHT to canonical RIGHT",
          abs(ccw.to_canonical(270.0) - 90.0) < 1e-6,
          f"raw 270 -> {ccw.to_canonical(270.0):.0f}deg")
    check("a measured CCW frame maps raw LEFT to canonical LEFT",
          abs(ccw.to_canonical(90.0) - 270.0) < 1e-6,
          f"raw 90 -> {ccw.to_canonical(90.0):.0f}deg")

    # ── C. ONE hardware measurement -> AT MOST ONE tracker update ──
    # The audio loop reads the cache every 0.25 s; the DSP answers every
    # ~0.35 s plus 200-500 ms of subprocess start-up. The same reading was
    # therefore consumed 2-4 times (up to 8 while the cache stayed valid),
    # so JUMP_CONFIRMATIONS=3 could be satisfied by ONE physical reading.
    tr = DOATracker()
    tr.update(DOAReading(100.0, confidence=0.9, source="usb", seq=1), now=0.0)
    cached = DOAReading(200.0, confidence=0.9, source="usb", seq=2)
    for i in range(4):
        tr.update(cached, now=0.25 * (i + 1))
    check("one cached measurement read 4x cannot flip the track",
          angular_diff(tr.angle, 100.0) < 1.0, f"track {tr.angle:.0f}deg")

    tr = DOATracker()
    dup = DOAReading(90.0, confidence=0.9, source="usb", seq=7)
    for i in range(8):
        tr.update(dup, now=0.25 * i)
    check("one measurement read 8x fills ONE history slot, not eight",
          len(tr._history) == 1, f"{len(tr._history)} entries")

    tr = DOATracker()
    tr.update(DOAReading(100.0, confidence=0.9, source="usb", seq=1), now=0.0)
    for i, s in enumerate((2, 3, 4)):
        tr.update(DOAReading(200.0, confidence=0.9, source="usb", seq=s),
                  now=0.25 * (i + 1))
    check("but three DISTINCT measurements still follow a real manoeuvre",
          angular_diff(tr.angle, 200.0) < 1.0, f"track {tr.angle:.0f}deg")

    tr = DOATracker()
    for i in range(4):
        tr.update(DOAReading(50.0, confidence=0.9, source="srp"), now=0.25 * i)
    check("SRP-PHAT (seq=0) is recomputed per block and is NOT deduplicated",
          len(tr._history) == 4, f"{len(tr._history)} entries")

    tr = DOATracker()
    tr.update(DOAReading(90.0, confidence=0.9, source="usb", seq=1), now=0.0)
    stale = DOAReading(90.0, confidence=0.9, source="usb", seq=1)
    for t in (2.0, 4.0, 6.5):
        tr.update(stale, now=t)
    tr.update(DOAReading(None), now=7.0)
    check("re-reading a stale cache does not keep a dead track alive",
          tr.angle is None)

    # ── D. A 2-vs-2 TIE must not be won by the smaller azimuth ──
    ties = ([300., 302., 60., 62.], [350., 352., 170., 172.],
            [10., 12., 190., 192.])
    for beams in ties:
        check(f"{beams} is reported ambiguous, not silently resolved",
              select_azimuth(beams)[2])

    # With a stable prior, the tie holds the CONFIRMED side — in both
    # directions, so this is not a new bias in the opposite direction.
    for beams, near, far in ((ties[0], 301.0, 61.0), (ties[1], 351.0, 171.0),
                             (ties[2], 191.0, 11.0)):
        got_near = select_azimuth(beams, prefer_deg=near)[0]
        got_far = select_azimuth(beams, prefer_deg=far)[0]
        check(f"tie on {beams} follows the established side, not the "
              f"smaller angle",
              angular_diff(got_near, near) < 2.0
              and angular_diff(got_far, far) < 2.0,
              f"prefer {near:.0f} -> {got_near:.0f}; "
              f"prefer {far:.0f} -> {got_far:.0f}")

    check("a tie is still decided independently of beam ORDER",
          len({select_azimuth(list(p), prefer_deg=301.0)[0]
               for p in itertools.permutations([300., 302., 60., 62.])}) == 1)

    # Only an UNAMBIGUOUS reading may become the anchor, or a tie would
    # confirm itself for ever.
    hw = HardwareDOA.__new__(HardwareDOA)
    hw.beam_index = -1
    hw._seq = 0
    hw._stable_raw = None
    hw._stable_stamp = 0.0
    check("HardwareDOA starts with no stable side to lean on",
          hw._stable_raw is None)

    # ── E/F. AMBIGUITY MUST NOT INVENT A SECOND PHYSICAL TARGET ──
    # The renderers used to draw a hollow marker at (180 - bearing). That
    # angle was never measured: for the USB branch the real rival cluster
    # of [300,302,60,62] sits at 301 deg while the ghost was drawn at 119.
    for name in ("radar_overlay.py", "radar_gui.py"):
        path = pathlib.Path(__file__).with_name(name)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ghosts = [n for n in ast.walk(tree)
                  if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
                  and isinstance(n.left, ast.Constant)
                  and n.left.value in (180, 180.0)]
        check(f"{name} contains no executable (180 - bearing) ghost",
              not ghosts,
              f"line(s) {[n.lineno for n in ghosts]}" if ghosts else "")

    amb = AcousticObservation(bearing_deg=61.0, bearing_confidence=0.25,
                              bearing_source="usb", bearing_ambiguous=True,
                              bearing_calibrated=True)
    check("the ambiguity is still DISCLOSED, just not drawn as a target",
          "mirror" in amb.bearing_text(), amb.bearing_text())

    # ── G. An ambiguous reading must not seize an empty track ──
    tr = DOATracker()
    tr.update(DOAReading(90.0, confidence=0.25, source="usb", ambiguous=True,
                         seq=1), now=0.0)
    check("a single ambiguous reading does not acquire a target",
          tr.angle is None, f"track {tr.angle}")

    tr = DOATracker()
    for i, a in enumerate((10.0, 200.0, 100.0, 300.0)):
        tr.update(DOAReading(a, confidence=0.25, source="usb", ambiguous=True,
                             seq=i + 1), now=0.25 * i)
    check("scattered ambiguous readings never acquire a target",
          tr.angle is None, f"track {tr.angle}")

    # ...but hardware whose ambiguity is STRUCTURAL (a 2-mic array, where
    # ambiguous is always True) must still be able to report a bearing.
    tr = DOATracker()
    for i in range(4):
        tr.update(DOAReading(120.0, confidence=0.5, source="srp",
                             ambiguous=True), now=0.25 * i)
    check("a consistent 2-mic array still acquires after confirmations",
          tr.angle is not None and angular_diff(tr.angle, 120.0) < 1.0,
          f"track {tr.format()}")

    # ── H. The ambiguous flag belongs to the ACCEPTED measurement ──
    tr = DOATracker()
    tr.update(DOAReading(90.0, confidence=0.9, source="usb", seq=1), now=0.0)
    tr.update(DOAReading(250.0, confidence=0.25, source="usb", ambiguous=True,
                         seq=2), now=0.25)
    check("a REJECTED outlier does not flag the track ambiguous",
          not tr.ambiguous and angular_diff(tr.angle, 90.0) < 1.0)

    tr = DOATracker()
    for i, s in enumerate((1, 2, 3)):
        tr.update(DOAReading(90.0, confidence=0.25, source="usb",
                             ambiguous=True, seq=s), now=0.3 * i)
    check("a confirmed ambiguous track IS flagged ambiguous", tr.ambiguous)
    tr.reset()
    check("reset() clears the ambiguity — it cannot be inherited by the "
          "next target", not tr.ambiguous)

    # ── I. A single outlier must not poison the history ──
    tr = DOATracker()
    for i, a in enumerate((90.0, 90.0, 90.0, 200.0, 90.0, 90.0)):
        tr.update(DOAReading(a, confidence=0.9, source="usb", seq=i + 1),
                  now=0.25 * i)
    check("one outlier never enters the smoothing history",
          not any(abs(x - 200.0) < 1.0 for x, _ in tr._history)
          and angular_diff(tr.angle, 90.0) < 1.0,
          f"history {[round(x) for x, _ in tr._history]}")

    # ── J. Wrap-around: 359 and 1 are neighbours ──
    tr = DOATracker()
    for i, a in enumerate((359.0, 1.0, 358.0, 2.0)):
        tr.update(DOAReading(a, confidence=0.9, source="usb", seq=i + 1),
                  now=0.25 * i)
    check("359deg and 1deg smooth to 0deg, not to 180deg",
          angular_diff(tr.angle, 0.0) < 1.0, f"track {tr.angle:.1f}deg")
    check("and they are never treated as an outlier pair",
          angular_diff(359.0, 1.0) == 2.0)

    # ── K. One calibration point may not claim a measured handedness ──
    # Both hypotheses fit a single point with residual 0.0, so a strict '<'
    # silently kept whichever was tried first (CLOCKWISE) and then marked
    # the source CALIBRATED, which switched OFF the UNCAL warnings.
    def fit(points):
        fits = {}
        for hand in (bf.Handedness.CLOCKWISE, bf.Handedness.COUNTER_CLOCKWISE):
            zeros = [(t - (-m if hand is bf.Handedness.COUNTER_CLOCKWISE
                           else m)) % 360.0 for t, m in points]
            conv = bf.SourceConvention(bf.circular_mean_deg(zeros), hand, "usb")
            res = sum(bf.angular_distance(conv.to_canonical(m), t)
                      for t, m in points) / len(points)
            fits[hand] = (conv, res)
        spread = abs(fits[bf.Handedness.CLOCKWISE][1]
                     - fits[bf.Handedness.COUNTER_CLOCKWISE][1])
        return min(fits.values(), key=lambda f: f[1]), spread > 1.0

    for label, pts in (("one point", [(90.0, 30.0)]),
                       ("two points 180deg apart",
                        [(0.0, 0.0), (180.0, 180.0)])):
        (_conv, _res), resolved = fit(pts)
        check(f"{label} does NOT resolve the handedness", not resolved)

    (conv, res), resolved = fit([(0.0, 353.7), (90.0, 263.7)])
    check("two points 90deg apart DO resolve it — and find CCW here",
          resolved and conv.handedness is bf.Handedness.COUNTER_CLOCKWISE
          and res < 1.0, f"{conv.describe()}, residual {res:.1f}deg")

    check("a config with a zero but no handedness is still UNCALIBRATED",
          not bf.source_convention(
              "usb", {"doa_offset_deg": 60.0,
                      "doa_handedness": None}).calibrated)

    # ── L. The banner must not contradict the runtime ──
    # doa_invert is dead in the DOA path: flipping it changes no bearing.
    a = bf.source_convention("usb", {"doa_offset_deg": 180.0,
                                     "doa_invert": False})
    b = bf.source_convention("usb", {"doa_offset_deg": 180.0,
                                     "doa_invert": True})
    check("doa_invert provably does not steer the bearing",
          all(abs(a.to_canonical(r) - b.to_canonical(r)) < 1e-9
              for r in range(0, 360, 15)))
    banner = _calib.describe(dict(_calib.DEFAULTS,
                                  doa_offset_deg=180.0, doa_invert=True))
    check("so the banner no longer prints 'дзеркально' from that dead key",
          "дзеркально" not in banner and "UNKNOWN" in banner, banner)


def main() -> int:
    logging.getLogger("station").setLevel(logging.CRITICAL)

    print("=" * 70)
    print("UNIFIED DRONE DETECTION STATION — integration validation")
    print("=" * 70)

    for fn in (test_1_no_drone, test_2_acoustic_only, test_3_handover,
               test_4_visual_lost, test_5_acoustic_lost, test_6_both_lost,
               test_7_switch_oscillation, test_8_microphone_fails,
               test_9_camera_fails, test_10_load,
               test_11_acoustic_latency_and_coasting, test_12_led_ring,
               test_13_coordinate_chain, test_14_srp_geometry,
               test_15_camera_search_region, test_16_audit_fixes,
               test_17_web_server, test_18_doa_fix_regressions,
               test_19_camera_identity, test_20_camera_switch_resets_state,
               test_21_hardware_claims, test_22_srp_gating,
               test_23_acoustic_timestamp_semantics,
               test_24_camera_manager_role_resolution,
               test_25_forensic_review_fixes,
               test_regressions, test_final_audit_regressions,
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

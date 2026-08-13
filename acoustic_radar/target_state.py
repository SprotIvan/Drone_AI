#!/usr/bin/env python3
"""
target_state.py — Central target representation and thread-safe exchange.

This module is the ONLY place where the acoustic and visual subsystems meet.
Neither worker touches the other's variables; each publishes an immutable
observation, and the fusion layer combines them.

═══════════════════════════════════════════════════════════════════
DESIGN RULES ENFORCED HERE
═══════════════════════════════════════════════════════════════════

1. Every observation carries a monotonic timestamp. `time.monotonic()` is
   used, never `time.time()`: wall clock can jump (NTP step, manual set) and
   a backwards jump would make fresh data look stale, or stale data fresh.

2. Absent measurements are None. There is no "0 means unknown" and no
   sentinel like -1. A field that is None renders as N/A in the UI and is
   never substituted with a plausible-looking number.

3. Observations are frozen dataclasses. A worker builds one and hands it
   over; nothing can mutate it afterwards, so a reader can never observe a
   half-updated record without holding a lock during use.

4. The exchange (`LatestValue`) holds a single latest snapshot, not a queue.
   A slow consumer therefore cannot make a producer block or accumulate an
   unbounded backlog — it simply misses intermediate frames, which is the
   correct behaviour for a live sensor display.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Deque, Generic, List, Optional, Tuple, TypeVar

import numpy as np


def now() -> float:
    """Single time source for the whole application. Monotonic by design."""
    return time.monotonic()


# ═══════════════════════════════════════════════════════════════
#  Freshness
# ═══════════════════════════════════════════════════════════════

class Freshness(Enum):
    """How much a reading can be trusted, given its age."""
    FRESH = "FRESH"       # within the stale timeout
    STALE = "STALE"       # older than stale, younger than lost
    LOST = "LOST"         # older than the lost timeout
    NEVER = "NEVER"       # this sensor has never produced a reading

    @property
    def usable(self) -> bool:
        """FRESH data may drive decisions. STALE may be displayed, not trusted."""
        return self is Freshness.FRESH


def classify_age(age_s: Optional[float], stale_s: float,
                 lost_s: float) -> Freshness:
    if age_s is None:
        return Freshness.NEVER
    if age_s <= stale_s:
        return Freshness.FRESH
    if age_s <= lost_s:
        return Freshness.STALE
    return Freshness.LOST


# ═══════════════════════════════════════════════════════════════
#  Acoustic observation
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AcousticObservation:
    """
    One acoustic decision, i.e. one 0.5 s block processed by RadarEngine.

    Field-by-field this mirrors what radar.RadarStatus actually provides —
    no field was invented to match a nicer-looking interface. In particular:

      • there is NO acoustic elevation: the array reports azimuth only;
      • there is NO native velocity: the engine does not estimate closing
        speed at all. Velocity is derived separately by AcousticTrackHistory
        from the distance series, and is clearly labelled as derived;
      • distance is None unless `calibrate.py range` has been run.
    """

    # ── Classification ──
    #: SLEEP | LISTEN | TRACK | ALARM | ALARM_COASTING
    engine_state: str = "SLEEP"
    p_drone: float = 0.0               # instantaneous P(drone) this block
    p_smoothed: float = 0.0            # EMA-smoothed, this is what decides
    threshold: float = 0.0             # P_START — engine's entry threshold
    hold_threshold: float = 0.0        # P_HOLD — threshold to KEEP an alarm
    confirmations: int = 0
    confirm_needed: int = 0            # confirmations required for an alarm
    misses: int = 0
    miss_tolerance: int = 0            # missed blocks before the target drops
    gated: bool = True                 # below the noise gate = digital silence

    # ── Bearing (azimuth only) ──
    #: THE CANONICAL BEARING (see bearing_frame): 0 = the direction the
    #: installation faces, increasing CLOCKWISE. Already normalised out of
    #: whichever sensor produced it, exactly once, by DOAProvider. The
    #: radar, the LED ring and the camera cue all consume THIS and apply
    #: only their own output transform to it.
    bearing_deg: Optional[float] = None
    bearing_confidence: float = 0.0
    bearing_source: str = "none"       # usb | srp | none
    bearing_ambiguous: bool = False    # tie between opposite directions
    #: The source's convention was actually measured. False means the
    #: direction may be mirrored, so nothing may POINT with it.
    bearing_calibrated: bool = False
    bearing_reason: str = ""           # why, when not calibrated

    # ── Range (estimate, with interval) ──
    distance_m: Optional[float] = None
    distance_lo_m: Optional[float] = None
    distance_hi_m: Optional[float] = None
    distance_reason: str = ""          # why distance is None, when it is
    level_dbfs: float = -120.0
    noise_dbfs: float = -120.0
    excess_db: float = 0.0

    # ── Bookkeeping ──
    timestamp: float = field(default_factory=now)
    seq: int = 0
    overflow: bool = False             # audio buffer overrun on this block

    @property
    def detected(self) -> bool:
        """Engine has a target under track (suspected, confirmed or held)."""
        return self.engine_state in ("TRACK", "ALARM", "ALARM_COASTING")

    @property
    def confirmed(self) -> bool:
        """
        Engine has raised the alarm — enough confirmations.

        ALARM_COASTING counts as confirmed on purpose. It is the SAME
        target, still held by radar.Detector's miss tolerance; the signal is
        merely absent for a moment. Excluding it here would make the fusion
        layer drop and re-acquire the target every time the drone turned or
        the wind gusted, which is exactly the flicker the coasting state
        exists to prevent. Use `coasting` to tell the two apart for display.
        """
        return self.engine_state in ("ALARM", "ALARM_COASTING")

    @property
    def coasting(self) -> bool:
        """
        Alarm is held, but there is NO current signal.

        Everything positional in this observation (bearing, distance) is the
        LAST VALID measurement, not a fresh one. Any UI that shows those
        numbers while this is True must say so.
        """
        return self.engine_state == "ALARM_COASTING"

    @property
    def has_bearing(self) -> bool:
        return self.bearing_deg is not None

    @property
    def has_distance(self) -> bool:
        return self.distance_m is not None

    def distance_text(self) -> str:
        """
        Distance for display. Never presented as an exact figure: acoustic
        ranging with a single small array is a level-based estimate whose
        real accuracy is ±40–60%.
        """
        if self.distance_m is None:
            return "N/A"
        if self.distance_lo_m is not None and self.distance_hi_m is not None:
            return (f"~{self.distance_m:.0f} m "
                    f"({self.distance_lo_m:.0f}-{self.distance_hi_m:.0f})")
        return f"~{self.distance_m:.0f} m"

    def bearing_text(self) -> str:
        if self.bearing_deg is None:
            return "N/A"
        text = f"{self.bearing_deg:.0f}°"
        if not self.bearing_calibrated:
            # An uncalibrated direction may be mirrored. Showing it as a
            # bare number would assert a physical direction the system has
            # no basis for.
            text += " (UNCAL)"
        if self.bearing_ambiguous:
            text += " (±mirror)"
        return text


# ═══════════════════════════════════════════════════════════════
#  Visual observation
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VisualTrack:
    """One tracked object as seen by the camera."""
    track_id: int
    bbox: Tuple[float, float, float, float]        # x, y, w, h in frame px
    state: str                                     # Tentative|Confirmed|Lost
    quality_score: float                           # tracker health, NOT conf
    detection_score: Optional[float]               # real detector confidence
    detection_age_s: Optional[float]               # since last real detection
    distance_m: Optional[float]                    # pinhole, None if uncalib.

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def confirmed(self) -> bool:
        return self.state == "Confirmed"


@dataclass(frozen=True)
class VisualObservation:
    """
    One processed camera frame.

    `frame` is a reference, not a copy. The camera worker guarantees it will
    never write into a frame it has already published (picamera2 returns a
    fresh array per capture), so the UI may draw on it without a copy. This
    is the single largest per-frame cost avoided in the integration.
    """

    frame: Optional[np.ndarray] = None            # BGR, ready for display
    frame_width: int = 0
    frame_height: int = 0
    tracks: Tuple[VisualTrack, ...] = ()
    detector_ran: bool = False                    # YOLO ran on this frame
    detection_count: int = 0
    active_camera: int = 0
    camera_name: str = "?"
    available_cameras: Tuple[int, ...] = ()

    # ── Timing ──
    inference_ms: Optional[float] = None
    tracker_ms: Optional[float] = None
    loop_fps: float = 0.0

    timestamp: float = field(default_factory=now)
    seq: int = 0

    @property
    def confirmed_tracks(self) -> List[VisualTrack]:
        return [t for t in self.tracks if t.confirmed]

    def best_track(self, min_conf: float) -> Optional[VisualTrack]:
        """
        The track most worth calling "the target".

        Chosen by largest box among confirmed tracks whose most recent
        detector confidence clears `min_conf`. Largest box = closest target,
        which is the one an operator cares about. Tracks coasting without a
        detection score are eligible only if they were confirmed, because a
        confirmed track has already been seen at least min_hits times.
        """
        best: Optional[VisualTrack] = None
        best_area = 0.0
        for t in self.tracks:
            if not t.confirmed:
                continue
            if t.detection_score is not None and t.detection_score < min_conf:
                continue
            area = t.bbox[2] * t.bbox[3]
            if area > best_area:
                best, best_area = t, area
        return best


# ═══════════════════════════════════════════════════════════════
#  Derived acoustic kinematics
# ═══════════════════════════════════════════════════════════════

class Approach(Enum):
    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    STEADY = "STEADY"
    UNKNOWN = "UNKNOWN"        # not enough data — never guessed


@dataclass(frozen=True)
class Kinematics:
    """
    Closing speed derived from the acoustic distance series.

    ⚠️ This is DERIVED, not measured. The acoustic subsystem provides no
    velocity of its own; this is the slope of a least-squares line through
    recent distance estimates, each of which carries ±40–60% uncertainty.
    It is reported only when there are enough samples and the slope is
    larger than the configured deadband; otherwise `Approach.UNKNOWN`.
    """
    approach: Approach = Approach.UNKNOWN
    speed_mps: Optional[float] = None       # signed: negative = closing
    samples: int = 0

    def text(self) -> str:
        if self.approach is Approach.UNKNOWN or self.speed_mps is None:
            return "UNKNOWN"
        return f"{self.approach.value} ({abs(self.speed_mps):.0f} m/s)"


class AcousticTrackHistory:
    """
    Rolling history of acoustic bearing/distance, used for the radar trail
    and for the derived closing speed.

    Single-threaded by contract: only the fusion layer touches it.
    """

    def __init__(self, window_s: float, min_samples: int,
                 deadband_mps: float, trail_s: float):
        self.window_s = window_s
        self.min_samples = min_samples
        self.deadband_mps = deadband_mps
        self.trail_s = trail_s
        # (timestamp, distance_m) — distance only, for the velocity fit
        self._range_hist: Deque[Tuple[float, float]] = deque(maxlen=64)
        # (timestamp, bearing_deg, distance_m|None) — for the radar trail
        self._trail: Deque[Tuple[float, float, Optional[float]]] = deque(maxlen=256)

    def clear(self) -> None:
        self._range_hist.clear()
        self._trail.clear()

    def add(self, obs: AcousticObservation) -> None:
        if obs.has_bearing:
            self._trail.append((obs.timestamp, float(obs.bearing_deg),
                                obs.distance_m))
        if obs.has_distance:
            self._range_hist.append((obs.timestamp, float(obs.distance_m)))

    def trail(self, t_now: float) -> List[Tuple[float, float, Optional[float]]]:
        """Recent (age_frac, bearing, distance) for the radar, newest last."""
        if self.trail_s <= 0:
            return []
        out = []
        for ts, bearing, dist in self._trail:
            age = t_now - ts
            if age <= self.trail_s:
                out.append((age / self.trail_s, bearing, dist))
        return out

    def kinematics(self, t_now: float) -> Kinematics:
        """
        Least-squares slope of distance vs time over the recent window.

        Returns UNKNOWN unless there are at least `min_samples` points
        spanning a usable time base. A slope inside the deadband is reported
        as STEADY rather than as a small approach speed, because the
        underlying range estimate is not precise enough to distinguish them.
        """
        pts = [(ts, d) for ts, d in self._range_hist
               if t_now - ts <= self.window_s]
        if len(pts) < self.min_samples:
            return Kinematics(Approach.UNKNOWN, None, len(pts))

        t = np.array([p[0] for p in pts], dtype=np.float64)
        d = np.array([p[1] for p in pts], dtype=np.float64)
        t -= t.mean()
        denom = float(np.dot(t, t))
        if denom < 1e-9:                     # all samples at the same instant
            return Kinematics(Approach.UNKNOWN, None, len(pts))

        slope = float(np.dot(t, d - d.mean()) / denom)   # m/s, + = receding

        if abs(slope) < self.deadband_mps:
            return Kinematics(Approach.STEADY, slope, len(pts))
        if slope < 0:
            return Kinematics(Approach.APPROACHING, slope, len(pts))
        return Kinematics(Approach.RECEDING, slope, len(pts))


# ═══════════════════════════════════════════════════════════════
#  Thread-safe latest-value exchange
# ═══════════════════════════════════════════════════════════════

T = TypeVar("T")


class LatestValue(Generic[T]):
    """
    A one-slot mailbox holding the most recent value published by a producer.

    Why not a Queue: a queue either grows without bound when the consumer is
    slow (memory leak, and the consumer falls further behind reality) or
    blocks the producer when full (the camera thread would stall on the UI).
    A live sensor display wants neither — it wants the newest value and does
    not care about the ones it missed.

    The lock is held only for a reference assignment/read, so contention
    between the audio thread, the camera thread and the UI is negligible.
    """

    __slots__ = ("_value", "_lock", "_count")

    def __init__(self, initial: Optional[T] = None):
        self._value: Optional[T] = initial
        self._lock = threading.Lock()
        self._count = 0

    def publish(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._count += 1

    def get(self) -> Optional[T]:
        with self._lock:
            return self._value

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# ═══════════════════════════════════════════════════════════════
#  Worker health
# ═══════════════════════════════════════════════════════════════

class SubsystemState(Enum):
    STARTING = "STARTING"
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"      # running, but something is wrong
    OFFLINE = "OFFLINE"        # failed to start or has stopped
    DISABLED = "DISABLED"      # intentionally not started


@dataclass(frozen=True)
class SubsystemHealth:
    """Status of one worker, so the UI can show partial failure honestly."""
    state: SubsystemState = SubsystemState.STARTING
    detail: str = ""
    error: Optional[str] = None
    restarts: int = 0
    timestamp: float = field(default_factory=now)

    def with_state(self, state: SubsystemState, detail: str = "",
                   error: Optional[str] = None) -> "SubsystemHealth":
        return replace(self, state=state, detail=detail, error=error,
                       timestamp=now())

    @property
    def ok(self) -> bool:
        return self.state in (SubsystemState.ONLINE, SubsystemState.DEGRADED)


if __name__ == "__main__":
    print("=" * 66)
    print("target_state.py — self-test")
    print("=" * 66)

    # 1. Freshness classification
    print("\n1. Freshness (stale=1.5 s, lost=3.0 s):")
    for age in (0.0, 1.0, 2.0, 5.0, None):
        f = classify_age(age, 1.5, 3.0)
        print(f"   age={str(age):>5} -> {f.value:<6} usable={f.usable}")

    # 2. Honest formatting of absent values
    print("\n2. Absent measurements are never invented:")
    empty = AcousticObservation()
    print(f"   bearing  -> {empty.bearing_text()}")
    print(f"   distance -> {empty.distance_text()}")
    filled = AcousticObservation(distance_m=87.0, distance_lo_m=52.0,
                                 distance_hi_m=145.0, bearing_deg=142.0)
    print(f"   bearing  -> {filled.bearing_text()}")
    print(f"   distance -> {filled.distance_text()}")

    # 3. Derived kinematics
    print("\n3. Derived closing speed (deadband 1.5 m/s):")
    t0 = now()
    for label, series in (
            ("approaching 10 m/s", [200, 195, 190, 185, 180, 175]),
            ("hovering",           [100, 101,  99, 100, 101,  99]),
            ("receding 6 m/s",     [100, 103, 106, 109, 112, 115]),
            ("too few samples",    [100, 90]),
    ):
        hist = AcousticTrackHistory(window_s=3.0, min_samples=4,
                                    deadband_mps=1.5, trail_s=8.0)
        n = len(series)
        for i, d in enumerate(series):
            ts = t0 - (n - 1 - i) * 0.5
            hist.add(AcousticObservation(distance_m=float(d), bearing_deg=142.0,
                                         timestamp=ts))
        k = hist.kinematics(t0)
        print(f"   {label:<20} -> {k.text():<28} (n={k.samples})")

    # 4. LatestValue never blocks and never backs up
    print("\n4. LatestValue keeps only the newest of 1000 published values:")
    box: LatestValue[int] = LatestValue()
    for i in range(1000):
        box.publish(i)
    print(f"   published={box.count}  held={box.get()}")
    print()

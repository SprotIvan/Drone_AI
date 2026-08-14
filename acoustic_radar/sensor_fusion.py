#!/usr/bin/env python3
"""
sensor_fusion.py — Explicit sensor-priority state machine and fused target.

═══════════════════════════════════════════════════════════════════
THE STATE MACHINE
═══════════════════════════════════════════════════════════════════

                          ┌──────────────┐
             ┌───────────►│  SEARCHING   │◄────────────┐
             │            └──────┬───────┘             │
             │                   │ acoustic contact    │ hold expires
             │                   ▼                     │
             │          ┌──────────────────┐    ┌──────┴──────┐
             │          │ ACOUSTIC_DETECTED│    │ TARGET_LOST │
             │          └────────┬─────────┘    └──────▲──────┘
             │                   │ engine confirms     │ both sensors
             │                   ▼                     │ lost
             │          ┌──────────────────┐           │
             │  ┌──────►│ ACOUSTIC_TRACKING├───────────┤
             │  │       └────────┬─────────┘           │
             │  │                │ in camera range     │
             │  │                │ + stable N updates  │
             │  │                ▼                     │
             │  │       ┌──────────────────┐           │
             │  │       │VISUAL_ACQUISITION├───────────┤
             │  │       └────────┬─────────┘           │
             │  │                │ YOLO confirms       │
             │  │                ▼                     │
             │  │       ┌──────────────────┐           │
             │  │       │ VISUAL_TRACKING  ├───────────┤
             │  │       └────────┬─────────┘           │
             │  │                │ visual lost         │
             │  │                ▼                     │
             │  │  ┌───────────────────────────┐       │
             │  └──┤VISUAL_LOST_ACOUSTIC_TRACK ├───────┘
             │     └───────────────────────────┘
             │                   │ visual reacquired
             │                   └──────► VISUAL_TRACKING
             │
             └── (visual-only path: SEARCHING ──► VISUAL_TRACKING when the
                 camera finds a target with no acoustic contact at all)

Every transition is implemented in exactly one place (`_next_state`) and
logged once when it fires. There is no priority logic scattered through
booleans elsewhere in the codebase.

═══════════════════════════════════════════════════════════════════
SENSOR PRIORITY
═══════════════════════════════════════════════════════════════════

    MICROPHONE PRIMARY   SEARCHING, ACOUSTIC_DETECTED, ACOUSTIC_TRACKING,
                         VISUAL_ACQUISITION, VISUAL_LOST_ACOUSTIC_TRACKING
    CAMERA PRIMARY       VISUAL_TRACKING

The microphone is never stopped when the camera takes over. It keeps
running and is used for confirmation, for the closing-speed estimate, for
cross-checking, and to re-cue the camera the moment visual tracking breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from camera_cue import BearingCue, BearingProjector
from fusion_config import StationConfig
from target_state import (
    AcousticObservation, AcousticTrackHistory, Freshness, Kinematics,
    SubsystemHealth, SubsystemState, VisualObservation, VisualTrack,
    classify_age, now,
)

log = logging.getLogger("station.fusion")


# ═══════════════════════════════════════════════════════════════
#  States
# ═══════════════════════════════════════════════════════════════

class SystemState(Enum):
    SEARCHING = "SEARCHING"
    ACOUSTIC_DETECTED = "ACOUSTIC_DETECTED"
    ACOUSTIC_TRACKING = "ACOUSTIC_TRACKING"
    VISUAL_ACQUISITION = "VISUAL_ACQUISITION"
    VISUAL_TRACKING = "VISUAL_TRACKING"
    VISUAL_LOST_ACOUSTIC_TRACKING = "VISUAL_LOST_ACOUSTIC_TRACKING"
    TARGET_LOST = "TARGET_LOST"

    @property
    def has_target(self) -> bool:
        return self not in (SystemState.SEARCHING, SystemState.TARGET_LOST)

    @property
    def label(self) -> str:
        """Short operator-facing name."""
        return {
            SystemState.SEARCHING: "SEARCHING",
            SystemState.ACOUSTIC_DETECTED: "ACOUSTIC CONTACT",
            SystemState.ACOUSTIC_TRACKING: "ACOUSTIC TRACKING",
            SystemState.VISUAL_ACQUISITION: "VISUAL ACQUISITION",
            SystemState.VISUAL_TRACKING: "VISUAL TRACKING",
            SystemState.VISUAL_LOST_ACOUSTIC_TRACKING: "VISUAL LOST - ACOUSTIC",
            SystemState.TARGET_LOST: "TARGET LOST",
        }[self]


class Priority(Enum):
    """Which sensor currently owns the target solution."""
    NONE = "NONE"
    ACOUSTIC = "ACOUSTIC"
    CAMERA = "CAMERA"


class CueRole(Enum):
    """
    What the acoustic direction should be doing on the CAMERA image.

    This is the whole of the "drone behind a tree" feature, decided in one
    place so the overlay can never contradict the state machine.

        NONE       nothing to draw — no acoustic bearing at all, the
                   reading is old enough to be LOST, or the projection is
                   impossible because the camera has no boresight/focal
                   calibration. An UNVERIFIED angle convention is NOT in
                   this list: it is drawn and labelled, exactly as the
                   radar and the ring do with the same number.

        SEARCH     the microphone has a target the CAMERA HAS NOT
                   CONFIRMED — either YOLO sees nothing there, or what it
                   sees is at a different bearing. This is the occluded
                   case: tree, building, roof, car. The image gets a
                   translucent red search region labelled ACOUSTIC ONLY.

        CONFIRMED  YOLO has a LIVE box that agrees with the acoustic
                   bearing. The box becomes the primary indication and the
                   search region is withdrawn, so the display never shows
                   two competing LIVE positions for one drone.

    Losing the visual track moves CONFIRMED back to SEARCH on the very next
    update, which is the reacquisition behaviour: the region reappears
    exactly where the microphone still hears the target.

    ⚠️ BUG-007 — WHAT HAPPENS DURING A DROPOUT, DECIDED EXPLICITLY.
    While the visual track is coasting, the HUD draws BOTH the last-known
    box and the acoustic region. That is deliberate and it is not a
    contradiction, because the two are making different statements:

        amber, DASHED, "LAST KNOWN"   where the camera last SAW it
        red, bracketed, "ACOUSTIC ONLY" where the microphone hears it NOW

    An earlier version of this docstring claimed the display "never" shows
    two positions, which was simply false — `hud._draw_boxes` appends the
    coasting track explicitly. The guarantee that actually holds, and that
    the styling enforces, is that never more than ONE of them is presented
    as a live visual detection.
    """
    NONE = "NONE"
    SEARCH = "ACOUSTIC ONLY"
    CONFIRMED = "VISUAL CONFIRMED"


#: Priority implied by each state. Single source of truth — nothing else
#: is allowed to decide who is primary.
_PRIORITY_OF_STATE = {
    SystemState.SEARCHING: Priority.NONE,
    SystemState.TARGET_LOST: Priority.NONE,
    SystemState.ACOUSTIC_DETECTED: Priority.ACOUSTIC,
    SystemState.ACOUSTIC_TRACKING: Priority.ACOUSTIC,
    SystemState.VISUAL_ACQUISITION: Priority.ACOUSTIC,
    SystemState.VISUAL_LOST_ACOUSTIC_TRACKING: Priority.ACOUSTIC,
    SystemState.VISUAL_TRACKING: Priority.CAMERA,
}


# ═══════════════════════════════════════════════════════════════
#  Fused snapshot — what the UI renders
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FusedTarget:
    """
    The unified target state at one instant.

    Every optional field is None when the corresponding sensor cannot
    currently supply it. The UI renders None as N/A. Nothing here is ever
    filled in with a plausible substitute.
    """

    # ── System ──
    state: SystemState = SystemState.SEARCHING
    priority: Priority = Priority.NONE
    state_age_s: float = 0.0
    target_id: int = 0

    # ── Acoustic ──
    acoustic: Optional[AcousticObservation] = None
    acoustic_freshness: Freshness = Freshness.NEVER
    acoustic_age_s: Optional[float] = None
    kinematics: Kinematics = field(default_factory=Kinematics)

    # ── Visual ──
    visual: Optional[VisualObservation] = None
    visual_track: Optional[VisualTrack] = None
    visual_freshness: Freshness = Freshness.NEVER
    visual_age_s: Optional[float] = None

    # ── Cross-sensor ──
    bearing_cue: BearingCue = field(default_factory=BearingCue)
    #: What the acoustic direction should be doing on the camera image.
    cue_role: CueRole = CueRole.NONE
    #: |visual bearing − acoustic bearing| in degrees, or None when the
    #: mount geometry is uncalibrated and the comparison is impossible.
    sensor_agreement_deg: Optional[float] = None
    #: True when the acoustic range says the target should be visible.
    in_camera_range: Optional[bool] = None
    camera_range_gate_m: Optional[float] = None

    #: Recent acoustic positions for the radar trail, as
    #: (age_fraction, bearing_deg, distance_m|None), oldest first.
    trail: Tuple[Tuple[float, float, Optional[float]], ...] = ()

    # ── Health ──
    acoustic_health: SubsystemHealth = field(default_factory=SubsystemHealth)
    visual_health: SubsystemHealth = field(default_factory=SubsystemHealth)

    timestamp: float = field(default_factory=now)

    # ── Convenience for the UI ─────────────────────────────────

    @property
    def acoustic_online(self) -> bool:
        return self.acoustic_health.ok

    @property
    def visual_online(self) -> bool:
        return self.visual_health.ok

    def bearing_text(self) -> str:
        if self.acoustic is None:
            return "N/A"
        return self.acoustic.bearing_text()

    def distance_text(self) -> str:
        """
        Preferred distance for the headline readout.

        The VISUAL distance is preferred when the camera owns the target,
        because a pinhole measurement from a tight bounding box is far more
        precise than a level-based acoustic estimate. Both are shown
        separately in the sensor bar; this is only the headline.
        """
        if (self.priority is Priority.CAMERA and self.visual_track is not None
                and self.visual_track.distance_m is not None):
            return f"{self.visual_track.distance_m:.1f} m (visual)"
        if self.acoustic is not None and self.acoustic.has_distance:
            return self.acoustic.distance_text() + " (acoustic)"
        return "N/A"

    def visual_confidence_text(self) -> str:
        """
        Real detector confidence, or an explicit absence.

        When the camera has lost the track, the number that was last
        measured is prefixed "LAST" rather than shown bare. It was true a
        moment ago and is worth showing, but presenting a past measurement
        in the same form as a live one would be exactly the kind of quiet
        staleness this system is required not to produce.
        """
        if self.visual_track is None:
            return "N/A"
        if self.visual_track.detection_score is None:
            return "COASTING"      # tracked by prediction, no fresh detection
        text = f"{self.visual_track.detection_score:.0%}"
        if (self.state is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING
                or self.visual_freshness is not Freshness.FRESH):
            return f"LAST {text}"
        return text


# ═══════════════════════════════════════════════════════════════
#  The fusion engine
# ═══════════════════════════════════════════════════════════════

class SensorFusion:
    """
    Combines the latest acoustic and visual observations into one target
    state, and drives the priority state machine.

    Threading contract: `update()` is called from exactly one thread (the
    UI thread). The observations it reads are immutable snapshots published
    by the worker threads, so no lock is needed here beyond the one already
    inside LatestValue.
    """

    def __init__(self, config: StationConfig, projector: BearingProjector):
        self.config = config
        self.projector = projector

        self.state = SystemState.SEARCHING
        self.priority = Priority.NONE
        self._state_since = now()
        self._target_id = 0

        self.history = AcousticTrackHistory(
            window_s=config.acoustic.velocity_window_s,
            min_samples=config.acoustic.velocity_min_samples,
            deadband_mps=config.acoustic.velocity_deadband_mps,
            trail_s=config.ui.radar_trail_s)

        # Gate counters — a target must satisfy a condition for N
        # consecutive updates, not once. This is what stops the system
        # flip-flopping on a single noisy estimate.
        self._in_range_streak = 0
        self._visual_confirm_streak = 0

        #: The camera that most recently produced a frame. Used when the
        #: camera worker has published nothing, so the cue and the range
        #: gate keep using the optics that were actually in play.
        self._last_active_camera = 0

        # Last-known-good visual position, kept across a visual dropout so
        # reacquisition has somewhere to look (STATE 7).
        self._last_visual_track: Optional[VisualTrack] = None
        self._last_visual_time: Optional[float] = None

        #: When EITHER sensor last had usable contact with the target.
        #: The target-lost timeout is measured from here, not from the last
        #: state change: a target held in ACOUSTIC_TRACKING for 60 s and
        #: then lost would otherwise be declared lost instantly, because
        #: 60 s already exceeds the 5 s timeout.
        self._last_contact_time: Optional[float] = None

        self._last_acoustic_seq = -1
        self.transitions: List[Tuple[float, SystemState, SystemState, str]] = []

    def reset(self) -> None:
        """
        Return to SEARCHING and forget every accumulated target.

        Exposed so the operator's 'r' key does not have to re-run __init__
        on a live object (which would silently drop any state added to the
        class later).
        """
        self.state = SystemState.SEARCHING
        self.priority = Priority.NONE
        self._state_since = now()
        self._in_range_streak = 0
        self._visual_confirm_streak = 0
        self._last_visual_track = None
        self._last_visual_time = None
        self._last_contact_time = None
        self._last_acoustic_seq = -1
        self.history.clear()
        self.transitions.clear()
        log.info("fusion state machine reset")

    # ── Threshold helpers ──────────────────────────────────────

    def acoustic_confidence_threshold(self,
                                      obs: Optional[AcousticObservation]) -> float:
        """
        Optional EXTRA acoustic confidence gate. Disabled by default.

        ⚠️ Do not re-enable this without a specific reason. It defaulted to
        the engine's own trained threshold (0.775), which sounds harmless
        and is not: radar.Detector applies that threshold to ENTER a track
        and then deliberately holds the track at threshold x HOLD_FACTOR
        (0.70), i.e. ~0.54. So an engine legitimately in ALARM routinely
        reports p_smoothed between 0.54 and 0.775 — and this gate then
        refused to promote it, leaving the target stuck in
        ACOUSTIC_DETECTED and never handing it to the camera.

        That is second-guessing a subsystem that already decided, using a
        threshold fitted on real validation data for a 2% false-alarm rate.
        The engine's verdict is now taken as-is, exactly as radar_gui.py
        always did. 0.0 = no additional gate.
        """
        configured = self.config.acoustic.confidence_threshold
        if configured is not None:
            return float(configured)
        return 0.0

    # ── Main entry point ───────────────────────────────────────

    def update(self, acoustic: Optional[AcousticObservation],
               visual: Optional[VisualObservation],
               acoustic_health: SubsystemHealth,
               visual_health: SubsystemHealth,
               t: Optional[float] = None) -> FusedTarget:
        """
        Fuse the latest observations into a target state.

        Args:
            t: the current time on the same monotonic base the observations
               are stamped with. Defaults to now(). It is injectable so
               timeout behaviour can be tested deterministically instead of
               by sleeping — and so every age in one update is computed
               against a SINGLE instant rather than several slightly
               different readings of the clock.
        """
        t = now() if t is None else float(t)

        # ── Freshness ──
        a_age = None if acoustic is None else t - acoustic.timestamp
        v_age = None if visual is None else t - visual.timestamp
        a_fresh = classify_age(a_age, self.config.acoustic.stale_after_s,
                               self.config.acoustic.lost_after_s)
        v_fresh = classify_age(v_age, self.config.visual.stale_after_s,
                               self.config.visual.lost_after_s)

        # ── Feed the history only with genuinely new acoustic frames ──
        # Without the sequence check the UI's higher tick rate would insert
        # the same acoustic reading many times, which would corrupt the
        # least-squares closing-speed fit by making one sample look like ten.
        if (acoustic is not None and acoustic.seq != self._last_acoustic_seq
                and a_fresh is Freshness.FRESH):
            self._last_acoustic_seq = acoustic.seq
            # ⚠️ A COASTING observation carries the last valid bearing and
            # distance re-stamped with a new time. Feeding those into the
            # history would insert a run of identical distances at advancing
            # timestamps — a perfectly flat least-squares fit — and the UI
            # would confidently report a manoeuvring drone as STEADY. The
            # history is left untouched instead: it is a record of
            # measurements, and during coasting no measurement is made.
            if acoustic.coasting:
                pass
            elif acoustic.detected:
                self.history.add(acoustic)
            else:
                self.history.clear()

        kinematics = self.history.kinematics(t)

        # ── Per-sensor booleans, computed once, used everywhere ──
        conf_gate = self.acoustic_confidence_threshold(acoustic)
        acoustic_live = (acoustic is not None and a_fresh is Freshness.FRESH
                         and acoustic.detected and acoustic_health.ok)
        acoustic_confident = (acoustic_live
                              and acoustic.p_smoothed >= conf_gate)
        acoustic_confirmed = acoustic_live and acoustic.confirmed

        # ⚠️ BUG-006. Which camera's OPTICS apply.
        #
        # `0 if visual is None` silently assumed the FAR camera whenever the
        # camera worker had not published — while the hardware might well be
        # on NEAR. That is not only a display question: the same id chooses
        # activation_distance_m()/release_distance_m(), so the camera-RANGE
        # GATE, a fusion decision, was evaluated with the wrong optics
        # (~20 m for FAR against ~8 m for NEAR).
        #
        # The last camera that actually produced a frame is remembered
        # instead. It is the best available answer, and it is a measurement
        # rather than an assumption.
        if visual is not None:
            active_camera = visual.active_camera
            self._last_active_camera = active_camera
        else:
            active_camera = self._last_active_camera
        visual_track = None
        if visual is not None and v_fresh is Freshness.FRESH:
            visual_track = visual.best_track(
                self.config.visual.confirm_conf_threshold)
        visual_live = visual_track is not None and visual_health.ok

        if visual_live:
            self._last_visual_track = visual_track
            self._last_visual_time = t

        if acoustic_live or visual_live:
            self._last_contact_time = t

        # ── Camera-range gate ──
        gate_m = self.config.activation_distance_m(active_camera)
        release_m = self.config.release_distance_m(active_camera)
        in_range = self._evaluate_range_gate(acoustic, acoustic_live,
                                             gate_m, release_m)

        # ── Streaks ──
        if in_range is True and acoustic_confident:
            self._in_range_streak += 1
        elif in_range is False:
            self._in_range_streak = 0
        # in_range is None (unknown distance) leaves the streak untouched:
        # absence of a measurement is not evidence either way.

        self._visual_confirm_streak = (self._visual_confirm_streak + 1
                                       if visual_live else 0)

        # ── Cross-sensor checks ──
        bearing = acoustic.bearing_deg if acoustic is not None else None
        bearing_conf = acoustic.bearing_confidence if acoustic is not None else 0.0
        # BUG-013: project into the pixel space of the frame that will
        # actually carry the overlay.
        frame_w = (visual.frame_width if visual is not None
                   and visual.frame_width else None)
        cue = self.projector.project(active_camera, bearing, bearing_conf,
                                     width_px=frame_w)

        agreement = None
        if visual_track is not None:
            agreement = self.projector.agreement_deg(
                active_camera, visual_track.center[0], bearing,
                width_px=frame_w)

        cue_role = self._cue_role(acoustic, a_fresh, visual_track, cue,
                                  agreement)

        # ── State machine ──
        new_state, reason = self._next_state(
            acoustic_live=acoustic_live,
            acoustic_confident=acoustic_confident,
            acoustic_confirmed=acoustic_confirmed,
            acoustic_freshness=a_fresh,
            in_range=in_range,
            visual_live=visual_live,
            visual_freshness=v_fresh,
            visual_health_ok=visual_health.ok,
            t=t)

        if new_state is not self.state:
            self._transition(new_state, reason, t)

        # ── Assemble the snapshot ──
        return FusedTarget(
            state=self.state,
            priority=self.priority,
            state_age_s=t - self._state_since,
            target_id=self._target_id,
            acoustic=acoustic,
            acoustic_freshness=a_fresh,
            acoustic_age_s=a_age,
            kinematics=kinematics,
            visual=visual,
            visual_track=visual_track or self._coasting_track(t),
            visual_freshness=v_fresh,
            visual_age_s=v_age,
            bearing_cue=cue,
            cue_role=cue_role,
            sensor_agreement_deg=agreement,
            in_camera_range=in_range,
            camera_range_gate_m=gate_m,
            trail=tuple(self.history.trail(t)),
            acoustic_health=acoustic_health,
            visual_health=visual_health,
            timestamp=t)

    # ── The camera's acoustic search region ────────────────────

    def _cue_role(self, acoustic: Optional[AcousticObservation],
                  freshness: Freshness, visual_track: Optional[VisualTrack],
                  cue: BearingCue,
                  agreement: Optional[float]) -> CueRole:
        """
        Decide, once, what the acoustic direction does on the camera image.

        Every condition below is a refusal to draw something misleading:

          • no bearing, or a reading old enough to be LOST -> nothing. A
            region drawn from a stale bearing tells an operator to search
            where the drone WAS.
          • an UNCALIBRATED bearing -> nothing. Its source convention was
            never measured, so it may be mirrored; a confident red region
            on the wrong side of the image is worse than no region.
          • the projection itself unavailable (no boresight, no focal
            length) -> nothing, and camera_cue already says why.
          • YOLO has a box that AGREES with the bearing -> the box is the
            truth and the region withdraws, so the operator is never shown
            two positions for one drone.
          • YOLO has a box that DISAGREES beyond the tolerance -> that is a
            different object, and the acoustic region stays up.
        """
        if acoustic is None or not acoustic.has_bearing:
            return CueRole.NONE
        if not acoustic.detected or freshness is Freshness.LOST:
            return CueRole.NONE
        if not cue.available:
            return CueRole.NONE

        # ⚠️ AN UNVERIFIED CONVENTION IS MARKED, NOT SUPPRESSED.
        #
        # An earlier version returned NONE here when `bearing_calibrated`
        # was False, and that was wrong in two ways.
        #
        # It was INCONSISTENT: the radar plots this very bearing, the LED
        # ring lights on it and the off-screen arrow points with it. Only
        # the camera region refused. One number cannot be trustworthy
        # enough for three consumers and too dangerous for the fourth.
        #
        # And it OVERRODE THE OPERATOR. `doa_handedness` records a
        # calibration; its absence means nobody wrote the file, not that
        # the direction is wrong. An operator who has watched the radar
        # track a source across the field has verified the direction more
        # convincingly than any config key — and refusing to draw for them
        # removes a working feature to enforce paperwork.
        #
        # So the region is drawn, and hud.py labels it BEARING UNVERIFIED.
        # That is this project's standing rule everywhere else: show the
        # measurement, state exactly what is not known about it.

        if visual_track is not None:
            tolerance = self.config.fusion.cue_agreement_deg
            # agreement None = the comparison is impossible (uncalibrated
            # optics). A box we cannot cross-check is still a visual
            # confirmation, which is how the system behaved before the
            # region existed.
            if agreement is None or agreement <= tolerance:
                return CueRole.CONFIRMED
        return CueRole.SEARCH

    # ── Range gate with hysteresis ─────────────────────────────

    def _evaluate_range_gate(self, acoustic: Optional[AcousticObservation],
                             acoustic_live: bool, gate_m: Optional[float],
                             release_m: Optional[float]) -> Optional[bool]:
        """
        Is the target inside the camera's useful detection range?

        Returns True / False / None, where None means "cannot be determined"
        — the acoustic range is not calibrated, or the optics are not
        calibrated, so there is no honest answer. None is NOT treated as
        False; the caller decides what to do with an unknown.

        Hysteresis: once inside, the target must exceed `release_m` (further
        out than `gate_m`) to be considered outside again.
        """
        if not acoustic_live or acoustic is None or not acoustic.has_distance:
            return None
        if gate_m is None:
            return None

        distance = float(acoustic.distance_m)
        currently_in = self.state in (SystemState.VISUAL_ACQUISITION,
                                      SystemState.VISUAL_TRACKING,
                                      SystemState.VISUAL_LOST_ACOUSTIC_TRACKING)
        threshold = (release_m if currently_in and release_m else gate_m)
        return distance <= threshold

    def _coasting_track(self, t: float) -> Optional[VisualTrack]:
        """
        The last confirmed visual track, while it is still recent enough to
        be worth showing as a "last known position" during a dropout
        (STATE 7). Beyond the visual lost timeout it is dropped entirely
        rather than displayed as if it were current.
        """
        if self._last_visual_track is None or self._last_visual_time is None:
            return None
        if t - self._last_visual_time > self.config.visual.lost_after_s:
            return None
        return self._last_visual_track

    # ── Transition table ───────────────────────────────────────

    def _next_state(self, *, acoustic_live: bool, acoustic_confident: bool,
                    acoustic_confirmed: bool, acoustic_freshness: Freshness,
                    in_range: Optional[bool],
                    visual_live: bool, visual_freshness: Freshness,
                    visual_health_ok: bool, t: float
                    ) -> Tuple[SystemState, str]:
        """
        The complete transition table. Returns (next_state, reason).

        Ordering rule: visual confirmation is evaluated before acoustic
        loss, so a target the camera can see is never dropped merely
        because the microphone lost it (TEST 5).
        """
        s = self.state
        cfg = self.config
        visual_confirmed = (visual_live
                            and self._visual_confirm_streak
                            >= cfg.visual.confirm_frames)
        range_ready = (self._in_range_streak
                       >= cfg.acoustic.acquisition_updates())

        # ── Camera confirmation wins from any state ──
        # This is the "CAMERA BECOMES PRIMARY" rule, and it applies even
        # with no acoustic contact when visual-only targets are allowed.
        if visual_confirmed:
            if s is not SystemState.VISUAL_TRACKING:
                if s in (SystemState.SEARCHING, SystemState.TARGET_LOST):
                    if not cfg.fusion.allow_visual_only_targets:
                        return s, ""
                    return SystemState.VISUAL_TRACKING, "visual-only acquisition"
                return SystemState.VISUAL_TRACKING, "YOLO confirmed the target"
            return SystemState.VISUAL_TRACKING, ""

        # ── SEARCHING ──
        if s is SystemState.SEARCHING:
            if acoustic_live:
                return SystemState.ACOUSTIC_DETECTED, "acoustic contact"
            return s, ""

        # ── TARGET_LOST → SEARCHING after the display hold ──
        if s is SystemState.TARGET_LOST:
            if acoustic_live:
                return SystemState.ACOUSTIC_DETECTED, "new acoustic contact"
            if t - self._state_since >= cfg.fusion.target_lost_display_s:
                return SystemState.SEARCHING, "lost-target hold expired"
            return s, ""

        # ── ACOUSTIC_DETECTED — suspected, not yet confirmed ──
        if s is SystemState.ACOUSTIC_DETECTED:
            if not acoustic_live:
                return self._acoustic_gone_state(acoustic_freshness, t)
            # Two independent gates must both pass: the engine's own state
            # machine must have raised the alarm (5 consecutive
            # confirmations), AND the smoothed probability must clear the
            # fusion-level confidence threshold.
            if acoustic_confirmed and acoustic_confident:
                return SystemState.ACOUSTIC_TRACKING, "acoustic alarm confirmed"
            return s, ""

        # ── ACOUSTIC_TRACKING — confirmed, camera cannot see it yet ──
        if s is SystemState.ACOUSTIC_TRACKING:
            if not acoustic_live:
                return self._acoustic_gone_state(acoustic_freshness, t)
            if range_ready:
                return (SystemState.VISUAL_ACQUISITION,
                        f"target inside camera range for "
                        f"{self._in_range_streak} acoustic updates")
            # ⚠️ `in_range is None` means the distance could not be
            # measured; `in_range is False` means it WAS measured and the
            # target is out of camera range. Conflating the two (an earlier
            # version tested only `_in_range_streak == 0`) made a target at
            # 200 m arm visual acquisition against a camera whose derived
            # range is ~20 m. Only genuine absence of a measurement may
            # fall through to the policy default here.
            if (in_range is None
                    and not cfg.acoustic.require_distance_for_acquisition
                    and acoustic_confirmed):
                return (SystemState.VISUAL_ACQUISITION,
                        "acoustic confirmed; range unknown, acquiring anyway")
            return s, ""

        # ── VISUAL_ACQUISITION — camera hunting, acoustic still primary ──
        if s is SystemState.VISUAL_ACQUISITION:
            if not visual_health_ok:
                return (SystemState.ACOUSTIC_TRACKING,
                        "camera offline, reverting to acoustic")
            if not acoustic_live:
                return self._acoustic_gone_state(acoustic_freshness, t)
            # Target has flown back out of camera range. `in_range` here is
            # already evaluated against the RELEASE distance (the wider of
            # the two thresholds), so this cannot chatter against the
            # activation gate — see _evaluate_range_gate.
            if in_range is False:
                self._in_range_streak = 0
                return (SystemState.ACOUSTIC_TRACKING,
                        "target left camera range")
            return s, ""

        # ── VISUAL_TRACKING — camera primary ──
        if s is SystemState.VISUAL_TRACKING:
            # visual_confirmed was already handled above, so getting here
            # means the visual track has dropped below confirmation.
            if visual_freshness is Freshness.FRESH and visual_live:
                return s, ""            # still seen, just not yet re-confirmed
            if acoustic_live:
                return (SystemState.VISUAL_LOST_ACOUSTIC_TRACKING,
                        "visual track lost, acoustic still holds the target")
            if t - (self._last_visual_time or t) >= cfg.visual.lost_after_s:
                return SystemState.TARGET_LOST, "visual lost, no acoustic contact"
            return s, ""

        # ── VISUAL_LOST_ACOUSTIC_TRACKING — fall back and try again ──
        if s is SystemState.VISUAL_LOST_ACOUSTIC_TRACKING:
            if not acoustic_live:
                return self._acoustic_gone_state(acoustic_freshness, t)
            # Acoustic still has it: go straight back to trying to reacquire.
            return (SystemState.VISUAL_ACQUISITION,
                    "re-cueing camera from acoustic bearing")

        return s, ""

    def _acoustic_gone_state(self, freshness: Freshness,
                             t: float) -> Tuple[SystemState, str]:
        """
        Where to go when the acoustic contact has dropped.

        A merely STALE reading is tolerated — the target may just be between
        two 0.5 s updates. Only a fully LOST reading, held past
        target_lost_after_s, declares the target lost.
        """
        if freshness is Freshness.STALE:
            return self.state, ""
        since_contact = t - (self._last_contact_time
                             if self._last_contact_time is not None
                             else self._state_since)
        if since_contact >= self.config.fusion.target_lost_after_s:
            return (SystemState.TARGET_LOST,
                    f"both sensors lost the target ({since_contact:.1f}s "
                    f"without contact)")
        # Not long enough yet — hold the current state and keep waiting.
        return self.state, ""

    def _transition(self, new_state: SystemState, reason: str,
                    t: float) -> None:
        old = self.state
        self.state = new_state
        self.priority = _PRIORITY_OF_STATE[new_state]
        self._state_since = t

        # A brand-new target gets a new id when we come out of a
        # no-target state, so the operator can tell "the same drone came
        # back" from "a second drone arrived".
        if not old.has_target and new_state.has_target:
            self._target_id += 1

        if new_state in (SystemState.SEARCHING, SystemState.TARGET_LOST):
            self._in_range_streak = 0
            self._visual_confirm_streak = 0
            if new_state is SystemState.SEARCHING:
                self.history.clear()
                self._last_visual_track = None
                self._last_visual_time = None
                self._last_contact_time = None

        self.transitions.append((t, old, new_state, reason))
        if len(self.transitions) > 200:
            del self.transitions[:100]

        log.info("STATE %s -> %s  (%s)  priority=%s",
                 old.value, new_state.value, reason or "-", self.priority.value)


if __name__ == "__main__":
    # Scenario-level self-test. Drives the machine with synthetic
    # observations and prints the resulting state sequence.
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    from fusion_config import load as load_config

    cfg = load_config()
    proj = BearingProjector(cfg.geometry, cfg.visual.frame_width)
    fusion = SensorFusion(cfg, proj)

    healthy = SubsystemHealth(SubsystemState.ONLINE)

    def acoustic(state="ALARM", p=0.95, dist=None, bearing=142.0, seq=0):
        return AcousticObservation(
            engine_state=state, p_smoothed=p, threshold=0.775,
            bearing_deg=bearing, bearing_confidence=0.8,
            distance_m=dist, timestamp=now(), seq=seq)

    def visual(n_tracks=0, score=0.9):
        tracks = tuple(
            VisualTrack(track_id=1, bbox=(300.0, 200.0, 60.0, 40.0),
                        state="Confirmed", quality_score=0.5,
                        detection_score=score, detection_age_s=0.0,
                        distance_m=5.3)
            for _ in range(n_tracks))
        return VisualObservation(tracks=tracks, timestamp=now(), seq=0)

    print("=" * 66)
    print("sensor_fusion.py — smoke test")
    print("=" * 66)
    print("(full scenario coverage lives in test_integration.py)")

    print("\nA drone at 200 m — far outside the camera's derived ~20 m range.")
    print("The camera must NOT be armed; acoustic stays primary.\n")
    fusion.update(None, visual(0), healthy, healthy)
    print(f"   start                -> {fusion.state.value}")
    for i in range(3):
        fusion.update(acoustic("TRACK", 0.9, 200.0, seq=i), visual(0),
                      healthy, healthy)
    print(f"   acoustic suspicion   -> {fusion.state.value}")
    for i in range(3, 12):
        fusion.update(acoustic("ALARM", 0.95, 200.0, seq=i), visual(0),
                      healthy, healthy)
    print(f"   acoustic confirmed   -> {fusion.state.value} "
          f"(priority {fusion.priority.value})")

    print("\nSame drone closing to 15 m — now inside camera range.\n")
    for i in range(12, 20):
        fusion.update(acoustic("ALARM", 0.95, 15.0, seq=i), visual(0),
                      healthy, healthy)
    print(f"   closed to 15 m       -> {fusion.state.value}")
    for i in range(20, 26):
        fusion.update(acoustic("ALARM", 0.95, 15.0, seq=i), visual(1),
                      healthy, healthy)
    print(f"   YOLO sees it         -> {fusion.state.value} "
          f"(priority {fusion.priority.value})")

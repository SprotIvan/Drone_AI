#!/usr/bin/env python3
"""
fusion_config.py — Central configuration for the unified detection station.

Every tunable number used by the sensor-fusion layer lives here. Nothing in
the fusion/UI code is allowed to contain a bare magic number.

═══════════════════════════════════════════════════════════════════
HOW TO READ THIS FILE
═══════════════════════════════════════════════════════════════════

Each value is tagged:

    [DERIVED]     computed from the existing subsystems' own parameters
                  (audio block rate, tracker miss tolerances, camera
                  optics). Changing the underlying subsystem changes this
                  automatically or is documented below.

    [MEASURED]    taken from a real measurement already present in the
                  project (e.g. the FAR camera focal length, which two
                  independent calibrations agreed on to 2.8%).

    [CALIBRATE]   ⚠️ NOT measured. A placeholder that is either honest-None
                  (the feature disables itself and the UI says so) or a
                  documented guess that MUST be validated in the field
                  before the number displayed to an operator means anything.

    [POLICY]      an operational choice, not a physical quantity.

Overrides: put any subset of these keys in `fusion_config.json` next to this
file. Values there win over the defaults. Unknown keys are reported rather
than silently ignored, because a typo'd key that is silently dropped is
indistinguishable from a setting that does not work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_PATH = Path(__file__).with_name("fusion_config.json")


# ═══════════════════════════════════════════════════════════════
#  Acoustic subsystem
# ═══════════════════════════════════════════════════════════════

@dataclass
class AcousticConfig:
    """
    Parameters of the acoustic early-warning sensor.

    The acoustic engine (radar.RadarEngine) decides at a fixed rate of
    1 / radar.BLOCK_SEC = 2 Hz. All timeouts below are expressed in seconds
    and were chosen as whole multiples of that 0.5 s period so that they
    correspond to an exact number of missed updates.
    """

    # [POLICY] OPTIONAL extra gate on smoothed P(drone). None = disabled,
    # which is the correct default: radar.Detector has already applied its
    # trained threshold AND its hysteresis, and re-applying a threshold on
    # top of a held track breaks it (see
    # SensorFusion.acoustic_confidence_threshold for the full explanation).
    # Set a number here only to make the station DELIBERATELY more
    # conservative than the detector it is built on.
    confidence_threshold: Optional[float] = None

    # [DERIVED] 3 missed updates (3 x 0.5 s). Beyond this the acoustic
    # reading is displayed as STALE: it is still shown, but explicitly
    # marked as not current.
    stale_after_s: float = 1.5

    # [DERIVED] radar.MISS_TOLERANCE = 6 blocks = 3.0 s is when the acoustic
    # detector itself gives up on a target. Matching that here means the
    # fusion layer and the engine declare acoustic loss at the same moment
    # instead of disagreeing.
    lost_after_s: float = 3.0

    # [DERIVED] Number of consecutive acoustic updates a target must remain
    # inside the camera-range gate before visual acquisition is armed.
    # 4 updates = 2.0 s of agreement. Prevents a single noisy distance
    # estimate from flipping the system into VISUAL_ACQUISITION.
    stable_updates_for_acquisition: int = 4

    # [POLICY] If the acoustic ranging is not calibrated there is no
    # distance at all, so the "is it in camera range?" gate cannot be
    # evaluated. False = still allow visual acquisition on a confirmed
    # acoustic contact alone (an uncalibrated system remains useful).
    # True = require a real distance, i.e. refuse to guess.
    require_distance_for_acquisition: bool = False

    # [DERIVED] Window used to estimate closing speed from the distance
    # history, in seconds. 3.0 s = 6 acoustic updates. Shorter windows are
    # dominated by the ±40–60% noise of acoustic ranging.
    velocity_window_s: float = 3.0

    # [DERIVED] Minimum number of distance samples inside that window before
    # any closing speed is reported at all. Below this the UI shows UNKNOWN.
    velocity_min_samples: int = 4

    # [CALIBRATE] Speed magnitude below which motion is reported as STEADY
    # rather than APPROACHING/RECEDING. 1.5 m/s is a deadband chosen to sit
    # above the drift of the smoothed acoustic range estimate, not a
    # measured figure. Validate by logging a stationary hovering drone and
    # checking that it is not reported as approaching.
    velocity_deadband_mps: float = 1.5

    # ── Detector tuning (overrides radar.DetectorTuning) ───────
    #
    # ⚠️ These are the ONLY place the station may retune the acoustic
    # detector. radar.py holds the defaults so that `python radar.py`
    # standalone behaves identically; None here means "use radar.py's
    # default" rather than a second, competing set of numbers.
    #
    # Every count below is in BLOCKS of radar.BLOCK_SEC (0.5 s). Do not
    # write seconds here — `detector_tuning()` is the only converter, and
    # it multiplies by the real block period so the two can never drift.

    # [POLICY] Blocks above P_START required to raise the alarm.
    # radar.py default 2 -> 2 x 0.5 s = 1.0 s confirmation latency.
    # Raising this is the correct response to too many false alarms; do NOT
    # raise the probability threshold, which was fitted on validation data.
    detector_confirm_blocks: Optional[int] = None

    # [POLICY] Blocks the alarm is HELD after the signal disappears before
    # the target is dropped (ALARM_COASTING). radar.py default 6 -> 3.0 s.
    # Must stay <= lost_after_s below, or the fusion layer declares the
    # sensor lost while the engine is still holding the target. The
    # acoustic worker logs a warning if that invariant is broken.
    detector_miss_tolerance: Optional[int] = None

    # [POLICY] Absolute P_HOLD — smoothed probability required to KEEP an
    # already-raised alarm. radar.py default 0.50 against a P_START of
    # 0.775 (P_START itself comes from training and is NOT settable here).
    # None = keep radar.py's 0.50. To revert to the older proportional
    # rule (P_START x HOLD_FACTOR = 0.542) set radar.HOLD_THRESHOLD = None.
    detector_hold_threshold: Optional[float] = None

    # [POLICY] EMA weight on the instantaneous probability. 1.0 = no
    # smoothing at all. radar.py default 0.5.
    detector_prob_ema: Optional[float] = None

    # [POLICY] Audio hop, in seconds. None = radar.BLOCK_SEC (0.5).
    #
    # ⚠️ THE LAST REMAINING LATENCY LEVER, AND IT IS NOT FREE.
    #
    # Measured budget from audible to indication (see latency.py):
    #     read wait     = block_seconds / 2   (mean quantisation delay)
    #     confirmation  = confirm_blocks x block_seconds
    #     everything else measured at ~30 ms combined
    #
    # Shortening the hop cuts the READ WAIT without weakening the decision,
    # because the 2 s analysis window is unchanged — only how often it is
    # re-evaluated. Shortening the CONFIRMATION cuts the evidence itself.
    #
    #     0.50 s hop, 2 confirmations -> ~1.28 s   (default, measured)
    #     0.25 s hop, 4 confirmations -> ~1.15 s   same evidence, 2x CPU
    #     0.25 s hop, 3 confirmations -> ~0.90 s   25% less evidence, 2x CPU
    #
    # DEFAULT LEFT UNCHANGED ON PURPOSE. Halving the hop doubles the
    # inference rate, and on the Raspberry Pi 5 the acoustic thread shares
    # its cores with the camera pipeline and Hailo. Measure block_total on
    # the Pi before spending that CPU: if it is not comfortably under the
    # new hop, the audio buffer overruns and detection gets WORSE.
    block_seconds: Optional[float] = None

    def detector_tuning(self):
        """
        Build a radar.DetectorTuning, leaving unset fields at radar.py's
        defaults.

        Imported lazily: fusion_config must stay importable on a machine
        with no numpy/onnxruntime (the config self-test, the docs build),
        and radar.py pulls in the whole DSP stack.
        """
        from radar import DetectorTuning

        tuning = DetectorTuning()
        for attr, value in (
                ("confirm_blocks", self.detector_confirm_blocks),
                ("miss_tolerance", self.detector_miss_tolerance),
                ("hold_threshold", self.detector_hold_threshold),
                ("prob_ema", self.detector_prob_ema)):
            if value is not None:
                setattr(tuning, attr, value)
        return tuning


# ═══════════════════════════════════════════════════════════════
#  Camera / visual subsystem
# ═══════════════════════════════════════════════════════════════

@dataclass
class VisualConfig:
    """Parameters of the visual confirmation sensor."""

    # [MEASURED] Detector confidence threshold actually used by the HEF
    # inference in TWO_CAMERAS_FIXED.HailoInference (conf_thresh=0.4).
    # Kept here so one file drives it.
    detector_conf_threshold: float = 0.4
    detector_iou_threshold: float = 0.45

    # [POLICY] Minimum detector confidence for the fusion layer to accept a
    # visual contact as a CONFIRMATION (i.e. hand priority to the camera).
    # Higher than the detector threshold on purpose: 0.4 is the level at
    # which a box is worth tracking, 0.55 is the level at which we are
    # willing to tell the operator "visually confirmed".
    confirm_conf_threshold: float = 0.55

    # [DERIVED] The tracker deletes a Confirmed track after
    # confirmed_miss_tolerance(3) + max_age(10) = 13 frames without a
    # detection; at the sensor's locked 30 fps that is ~0.43 s. The fusion
    # layer must not declare visual loss BEFORE the tracker has given up, or
    # it would fight its own tracker.
    stale_after_s: float = 0.30
    lost_after_s: float = 1.00

    # [POLICY] Consecutive frames a visual track must be Confirmed before
    # the fusion layer promotes the camera to primary. The IMM tracker
    # already requires min_hits=3 to confirm; this is a second, independent
    # guard at the fusion level.
    confirm_frames: int = 3

    # [MEASURED] Frame geometry actually configured in TWO_CAMERAS_FIXED.
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    max_fps: Optional[int] = None      # None = original hard 30 fps lock
    buffer_count: int = 4
    warmup_frames: int = 3

    # [MEASURED] Detector duty cycle from the original main loop: run YOLO
    # every Nth frame while a confirmed track exists, every frame otherwise.
    frame_skip: int = 2

    # [POLICY] Feed the detector the frame in RGB order.
    #
    # ⚠️ picamera2's "RGB888" format delivers B,G,R byte order in the numpy
    # array (a well-known libcamera naming quirk), and the original code
    # passed that array straight to the detector while showing a
    # channel-swapped copy. Whether that is correct depends on the channel
    # order the model was trained with — which cannot be determined from
    # this repository.
    #
    # False reproduces the original behaviour EXACTLY and is the default, so
    # this integration cannot change detection quality. Set True only after
    # A/B testing both settings against a known drone. See report section 5.
    swap_detector_channels: bool = False


# ═══════════════════════════════════════════════════════════════
#  Camera switching (near/far optics selection)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CameraSwitchConfig:
    """
    Which of the two cameras to use.

    ⚠️ SCALE NOTE — this is the single most misunderstood part of the
    system. These thresholds are in METRES and they are SMALL (1.5 / 2.0 m).
    They do NOT describe how far the system can see. They decide which
    OPTICS to use for a drone that is already very close:

        FAR  camera = IMX477, ~28° HFOV, focal 1274 px — the long lens
        NEAR camera = IMX708, ~66° HFOV, focal ~502 px — the wide lens

    Inside ~1.5 m the drone overflows the narrow lens's field of view, so
    the system swaps to the wide one. The acoustic sensor meanwhile works
    out to hundreds of metres. The two ranges are not comparable, which is
    exactly why acoustic distance must NOT drive this switch — see
    `allow_acoustic_fallback` below.
    """

    # [MEASURED/POLICY] Original values from TWO_CAMERAS_FIXED, preserved.
    # The 0.5 m gap is the hysteresis band: switch in at 1.5 m, back out
    # only past 2.0 m, hold whatever you have in between.
    switch_to_near_below_m: float = 1.5
    switch_to_far_above_m: float = 2.0

    # [POLICY] NEW — temporal confirmation on top of the distance
    # hysteresis. The original code switched on a SINGLE frame's estimate,
    # so one inflated bounding box could trigger a swap. The drone must now
    # be on the same side of the threshold for this many consecutive frames.
    # 5 frames at 30 fps = 0.17 s, which is short enough to be invisible to
    # an operator and long enough to reject per-frame box jitter.
    confirm_frames: int = 5

    # [MEASURED] Hard debounce inside CameraManager, in seconds. A switch
    # physically restarts the pipeline on the other sensor, so it must not
    # happen more often than this regardless of what the policy wants.
    debounce_interval_s: float = 0.5

    # [MEASURED] Fallback pixel-height thresholds, used only when the active
    # camera has no focal calibration and therefore no distance in metres.
    fallback_near_height_px: float = 120.0
    fallback_far_height_px: float = 80.0

    # [POLICY] Deliberately False. Acoustic range CANNOT drive this switch:
    # its floor (ranging.RangeEstimator.MIN_DISTANCE_M = 2.0 m) is already
    # above switch_to_near_below_m, and its stated accuracy is ±40–60%, so
    # it can never resolve a 1.5 m/2.0 m decision. Enabling this would make
    # the system permanently select FAR. Left as a flag only to document
    # that the option was considered and rejected.
    allow_acoustic_fallback: bool = False


# ═══════════════════════════════════════════════════════════════
#  Optics / geometry — acoustic bearing to camera pixel
# ═══════════════════════════════════════════════════════════════

@dataclass
class GeometryConfig:
    """
    Mapping between the acoustic frame of reference and the camera frame.

    ⚠️ READ THIS BEFORE TRUSTING THE ON-SCREEN BEARING CUE.

    An acoustic bearing and a camera pixel column are NOT the same thing and
    one cannot be converted into the other without knowing how the two
    sensors are physically mounted relative to each other. This project
    contains no such measurement. Rather than invent one, the cue is
    DISABLED by default: `camera_boresight_deg = None` makes the UI display
    "BEARING→VIEW: NOT CALIBRATED" and draw nothing.

    To enable it, perform the calibration in report section 12 and set
    camera_boresight_deg to the acoustic-frame azimuth that the camera's
    optical axis points at.
    """

    # [CALIBRATE] Azimuth (in the same 0–360° frame the acoustic subsystem
    # reports, after radar_calibration.json's doa_offset_deg has been
    # applied) that each camera's optical centre looks along.
    # None = unknown = cue disabled. NEVER guess these.
    camera_boresight_deg: Dict[int, Optional[float]] = field(
        default_factory=lambda: {0: None, 1: None})

    # [CALIBRATE] The value radar_calibration.json's `doa_offset_deg` had
    # when the boresight above was measured.
    #
    # ⚠️ THIS EXISTS BECAUSE THE BORESIGHT SILENTLY GOES STALE. The
    # boresight is an azimuth in the INSTALLATION frame, and doa_offset_deg
    # is what defines that frame. Change the offset — say, by -180 to stop
    # the radar showing the wrong side — and the frame rotates underneath
    # every boresight already recorded, by exactly the same amount. The
    # numbers still look plausible, nothing errors, and the camera cue just
    # points the wrong way.
    #
    # Recording the offset the measurement was taken under lets the station
    # notice and say so, instead of leaving it to be discovered by a drone
    # arrow pointing at the wrong horizon. None = never recorded, no check.
    boresight_calibrated_at_doa_offset_deg: Optional[float] = None

    # [CALIBRATE] The array's handedness ("CW"/"CCW") when the boresight
    # above was measured.
    #
    # ⚠️ Recorded SEPARATELY from the offset because the two break the
    # boresight in different ways. A changed offset ROTATES the frame and
    # every boresight can be corrected by adding the same delta. A changed
    # handedness MIRRORS it, and no delta can undo a mirror — the boresight
    # has to be measured again. Without this field the station would
    # cheerfully compute a delta after a mirror and hand out a number that
    # is wrong by twice the bearing.
    boresight_calibrated_handedness: Optional[str] = None

    # [MEASURED] Horizontal focal length in pixels per camera, copied from
    # TWO_CAMERAS_FIXED.CAMERA_FOCAL_PX so that one file drives both.
    # Camera 0 (IMX477 FAR): two independent calibrations at 5.0 m and 1.5 m
    #   agreed to 2.8% -> 1274.0 is trustworthy.
    # Camera 1 (IMX708 NEAR): UNVERIFIED, two candidate values (502 vs 1003)
    #   remain unresolved in the source comments. Any distance shown on the
    #   NEAR camera inherits that uncertainty.
    camera_focal_px: Dict[int, Optional[float]] = field(
        default_factory=lambda: {0: 1274.0, 1: 501.7})

    # [CALIBRATE] Real width of the drone being tracked, in metres. The
    # pinhole distance estimate scales linearly with this, so a 2x error
    # here is a 2x error in every visual distance.
    drone_real_width_m: Optional[float] = 0.25

    # [POLICY] Vertical tolerance for the bearing cue. The acoustic array
    # reports azimuth only — it has no elevation — so the cue can only be a
    # vertical band, never a point. Drawing a point would imply an
    # elevation measurement that does not exist.
    cue_is_azimuth_only: bool = True


# ═══════════════════════════════════════════════════════════════
#  Sensor-fusion state machine
# ═══════════════════════════════════════════════════════════════

@dataclass
class FusionConfig:
    """Timeouts and gates of the priority state machine."""

    # [DERIVED] Distance at or below which the target is considered to be
    # inside the camera's useful detection range, and above which it is not.
    #
    # None = compute it from the optics (see derive_visual_range_m). That is
    # the correct default because the answer follows from the lens, the
    # target size and the smallest box the detector can resolve — it is NOT
    # a free parameter, and it is certainly not 200 m.
    #
    # With the measured FAR focal length (1274 px), a 0.25 m drone and a
    # 16 px minimum box, the derived value is ~20 m. That is the honest
    # order of magnitude of this camera's drone-detection range.
    camera_activation_distance_m: Optional[float] = None
    camera_release_distance_m: Optional[float] = None

    # [CALIBRATE] Smallest bounding-box width, in pixels, at which the
    # detector still finds the drone reliably. 16 px is a conservative
    # engineering assumption for a 640-px-wide input, NOT a measurement of
    # this particular model. Measure it: walk the drone out until detection
    # becomes intermittent, read the box width, put that number here.
    min_detectable_box_px: float = 16.0

    # [POLICY] Hysteresis ratio between activation and release range. The
    # release distance is activation x this factor, so the system does not
    # drop out of visual acquisition the instant the noisy acoustic range
    # wobbles above the threshold. 1.5 = release 50% further out.
    range_hysteresis_factor: float = 1.5

    # [DERIVED] Both sensors silent for this long -> TARGET_LOST. Must be
    # longer than acoustic lost_after_s (3.0 s) so that a target which is
    # merely between acoustic updates is never declared lost. 5.0 s gives
    # 2 s of margin.
    target_lost_after_s: float = 5.0

    # [POLICY] How long TARGET_LOST is displayed before falling back to
    # SEARCHING. Purely so a human sees that the target was lost rather
    # than the banner vanishing instantly.
    target_lost_display_s: float = 3.0

    # [POLICY] Allow the camera to establish a target with no acoustic
    # contact at all (a silent or downwind drone that is nonetheless
    # visible). True = the camera is a fully independent detector, not just
    # a confirmer of acoustic contacts.
    allow_visual_only_targets: bool = True

    # [DERIVED] How closely a YOLO box must agree with the acoustic bearing
    # before the box is accepted as a visual confirmation OF THAT target
    # and the acoustic search region is withdrawn.
    #
    # 20 deg sits just above the DOA's own stated accuracy (+/-15 deg for
    # SRP-PHAT on a 43 mm array, doa.py's figure). Tighter and a correct
    # box gets rejected by DOA noise alone; looser and an unrelated object
    # elsewhere in the frame silently cancels the region for a drone that
    # is still hidden behind a tree.
    cue_agreement_deg: float = 20.0


# ═══════════════════════════════════════════════════════════════
#  User interface
# ═══════════════════════════════════════════════════════════════

@dataclass
class UIConfig:
    """Appearance and refresh policy of the single unified window."""

    window_name: str = "DRONE DETECTION STATION"
    fullscreen: bool = False

    # [POLICY] Upscale factor for the displayed frame.
    #
    # ⚠️ 1.0 (native 640x480), NOT 1.5. Upscaling costs a full-frame
    # interpolation pass plus a 2.25x larger buffer to composite and blit
    # every refresh — measured at more than double the HUD cost, and on a
    # Raspberry Pi 5 that is CPU taken directly from the camera and audio
    # threads. Detection is unaffected either way (the detector always runs
    # on the native frame), so this is pure display cost.
    #
    # Raise it only on a machine with CPU to spare, or drag the window
    # larger instead — the window is resizable and the compositor scales it
    # for free.
    display_scale: float = 1.0

    # ── Acoustic search region on the camera image ─────────────

    # [POLICY] "box" or "band".
    #
    # box   a compact rectangle marking where to look. Its WIDTH is the
    #       real azimuth uncertainty; its top and bottom edges are drawn
    #       OPEN, because the array measures azimuth only and the drone's
    #       height is genuinely unknown. This is the default: it is what an
    #       operator can find instantly on a busy frame.
    # band  the same region extended over the full image height. Makes the
    #       missing elevation impossible to overlook, at the cost of
    #       covering much more of the picture.
    #
    # Both are equally honest — the difference is how loudly the unknown
    # elevation is stated, not whether it is stated.
    cue_style: str = "box"

    # [POLICY] Height of the box, as a fraction of the image height, and
    # where its centre sits (0 = top, 1 = bottom).
    #
    # ⚠️ THE VERTICAL POSITION IS NOT A MEASUREMENT. Nothing in this system
    # measures elevation. The box is placed slightly above the middle
    # because that is where an airborne target usually appears in a level
    # camera, and its open top and bottom edges plus the ELEV NOT MEASURED
    # caption say so on the picture. Do not read the box's height as a
    # claim about the drone's altitude.
    cue_box_height_frac: float = 0.34
    cue_box_centre_frac: float = 0.42

    # [POLICY] Size of the compact MARKER at the centre of the region, as a
    # fraction of the image height.
    #
    # The marker and the uncertainty area are two different statements and
    # are sized independently on purpose: the marker says "the bearing
    # points HERE", the surrounding area says "and it cannot be narrower
    # than THIS". Sizing the marker to the uncertainty buries the first
    # message in a slab of colour; sizing the uncertainty to the marker
    # would claim a precision the array does not have.
    cue_marker_frac: float = 0.16

    # [POLICY] Radar widget size in pixels, and its inset from the frame
    # corner. Sized as a fraction of the DISPLAYED frame so it scales with
    # display_scale.
    radar_size_frac: float = 0.42
    radar_margin_px: int = 12

    # [DERIVED] The radar's static artwork (rings, spokes, cardinal marks)
    # is rendered once and cached; only the target blip and sweep are drawn
    # per frame. This is what keeps the overlay off the critical path.
    # Redraw of the cache is triggered only by a range-scale change.
    radar_sweep_deg_per_s: float = 90.0

    # [POLICY] Fade time of the target trail on the radar, in seconds.
    # 0 disables the trail.
    radar_trail_s: float = 8.0

    # [POLICY] Range rings, metres. The radar picks the smallest scale that
    # contains the current target and shows the ring labels for it.
    radar_scales_m: tuple = (25.0, 50.0, 100.0, 250.0, 500.0)

    # [POLICY] Redraw the whole HUD at most this often, in Hz.
    #
    # ⚠️ This is a HARD CPU BUDGET, not a target. The display is the only
    # part of the station that can be made cheaper without hurting
    # detection, so it is capped well below the camera rate on purpose: the
    # camera keeps capturing and running YOLO at full speed, and only the
    # picture on screen refreshes at 20 Hz — which is visually smooth and
    # leaves the sensor threads the CPU they need.
    #
    # The station also skips the redraw entirely when no new camera frame
    # has arrived, so this is a ceiling, not a floor.
    #
    # Raising this trades camera FPS for display smoothness. On a Pi 5,
    # don't.
    max_ui_fps: float = 20.0


# ═══════════════════════════════════════════════════════════════
#  ReSpeaker LED ring
# ═══════════════════════════════════════════════════════════════

@dataclass
class LedConfig:
    """
    Hardware indicator on the ReSpeaker XVF3800 microphone array.

    ⚠️ READ THIS BEFORE EXPECTING THE RING TO LIGHT UP.

    This repository contains NO LED control code and NO USB protocol
    implementation. The only verified control path to the array is the one
    doa.py already uses: running Seeed's external `xvf_host.py` utility as a
    subprocess with a control-command name (`xvf_host.py AEC_AZIMUTH_VALUES`).
    That utility is NOT part of this project — it ships with the array.

    Consequently the LED command NAMES cannot be taken from this repository,
    and they are not guessed here. `respeaker_led.py` asks the installed
    xvf_host.py for its own command list at startup and uses only names that
    the installed firmware actually reports. If nothing matches, the LED
    subsystem reports UNAVAILABLE, logs why, and the detection pipeline runs
    exactly as it does today.

    To see what your array supports:   python respeaker_led.py --probe
    Then put the confirmed names in fusion_config.json under "led".
    """

    # [POLICY] Master switch. True is safe: with nothing configured and no
    # array attached the controller reports UNAVAILABLE and costs one
    # idle thread that never writes anything.
    enabled: bool = True

    # [CALIBRATE] Number of LEDs on the physical ring. None = unknown, and
    # the ring is then NOT driven, because "light up the LED at 142°"
    # has no answer without it. NEVER guess this — count them, or read it
    # from `--probe` output if the firmware reports it.
    led_count: Optional[int] = None

    # [POLICY] How many LEDs the red target sector spans. Kept odd so the
    # sector is symmetric about the bearing. Clamped to led_count.
    sector_leds: int = 3

    # [CALIBRATE] Physical azimuth, in the ARRAY's own frame (before
    # radar_calibration.json's doa_offset_deg), that LED index 0 sits at.
    # 0.0 means "LED 0 is at the array's own 0 degrees".
    led_zero_offset_deg: float = 0.0

    # [CALIBRATE] True if LED indices increase clockwise when the array's
    # azimuth increases counter-clockwise (or vice versa). Verify by
    # driving a known bearing and looking at which LED lights.
    led_index_clockwise: bool = True

    # ── Command names (filled in from --probe) ─────────────────
    #
    # None = not configured. A None here is never substituted with a guess;
    # the controller simply reports which one is missing.

    # Command that takes the ring out of the firmware's own automatic
    # speech/DOA animation and into host control — the "LED_AUTO_MODE = 0"
    # step. Without it the firmware keeps repainting the ring on claps and
    # speech and fights every frame written here.
    #
    # ⚠️ The XVF3800 firmware observed in the field has NO LED_AUTO_MODE.
    # Its animation selector is LED_EFFECT (alongside LED_SPEED and
    # LED_DOA_COLOR), so that is what the controller looks for. The value
    # written is `auto_mode_off_value` and the exact call is logged,
    # because the effect enumeration is not something this project can
    # read — if the ring still animates on its own, try other small
    # integers here.
    cmd_auto_mode: Optional[str] = None        # e.g. LED_EFFECT
    auto_mode_off_value: str = "0"

    # Command that paints the ring, and (if the firmware has one) a command
    # that addresses a single LED.
    #
    # ⚠️ The ring command's ARITY is measured from the device, not assumed
    # from its name. On the XVF3800 firmware observed in the field
    # LED_RING_COLOR takes TWELVE values — one packed colour per LED —
    # despite reading like a single colour for the whole ring. That is the
    # good case: the entire sector paints in one call.
    cmd_ring_colour: Optional[str] = None      # LED_RING_COLOR
    cmd_led_colour: Optional[str] = None       # takes INDEX R G B

    # [CALIBRATE] Byte order inside a packed per-LED colour value.
    #
    # This is the ONE thing the controller cannot read off the device: the
    # firmware reports how many values it wants, but not how each is
    # encoded, and reading a dark ring back returns zeros. 24-bit 0xRRGGBB
    # is the near-universal convention and is the default. If the ring
    # shows blue where red is expected, change this to "bgr" — nothing
    # else about the indication depends on it.
    led_value_order: str = "rgb"               # "rgb" | "bgr"

    # [MEASURED] Flag the utility expects command values behind. Taken from
    # the usage line the installed xvf_host.py prints on a bad call:
    #
    #   xvf_host.py [-h] [-l] [--vid VID] [--pid PID]
    #               [--values VALUES [VALUES ...]] [COMMAND]
    #
    # so a write is `xvf_host.py LED_RING_COLOR --values 0 0 40`. Passing
    # the numbers positionally makes argparse reject them as unrecognized
    # arguments and every LED write fails. Set to null if some other
    # release of the utility really does take values positionally.
    values_flag: Optional[str] = "--values"

    # [POLICY] Colours as 0-255 RGB triples.
    # Dim blue is deliberately dim: this ring sits next to the operator and
    # a bright idle indicator ruins night vision and hides the alarm.
    colour_searching: tuple = (0, 0, 40)
    colour_alarm: tuple = (255, 0, 0)
    colour_coasting: tuple = (255, 0, 0)       # same red — the target is
                                               # still held, see report

    # [POLICY] Minimum seconds between two hardware writes. Each write is a
    # subprocess spawn (200-500 ms of python start-up, measured in doa.py),
    # so this is a hard floor, not a preference. 10 Hz of *requests* is
    # fine; the controller only writes when the effective frame changes.
    min_write_interval_s: float = 0.15

    # [POLICY] Bearing change, in degrees, that is worth a hardware write.
    # Below this the sector would land on the same LED anyway. Set from the
    # ring geometry at runtime if led_count is known.
    bearing_quantum_deg: float = 10.0

    # [POLICY] If no frame is submitted for this long the ring falls back
    # to the searching colour. This is the safety net for requirement 14:
    # a wedged audio thread must not leave the ring stuck showing red.
    watchdog_s: float = 3.0

    # [POLICY] Consecutive failed writes before the controller gives up and
    # reports UNAVAILABLE. It keeps retrying at a slow backoff so a
    # replugged array recovers without restarting the station.
    max_consecutive_failures: int = 5

    # [POLICY] Seconds to wait for one xvf_host.py invocation.
    command_timeout_s: float = 3.0


# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    level: str = "INFO"                    # DEBUG | INFO | WARNING | ERROR
    to_file: bool = True
    log_dir: str = "logs"
    max_bytes: int = 2_000_000
    backup_count: int = 3

    # [POLICY] Minimum seconds between two identical rate-limited messages.
    # Without this, per-frame DEBUG lines produce thousands of lines/second.
    rate_limit_s: float = 2.0

    # [POLICY] Log a target's bearing/distance only when it changes by more
    # than this, to keep the event log readable during a long track.
    bearing_change_deg: float = 10.0
    distance_change_frac: float = 0.25


# ═══════════════════════════════════════════════════════════════
#  Integrations (optional, off by default)
# ═══════════════════════════════════════════════════════════════

@dataclass
class IntegrationConfig:
    """
    Optional side-channels carried over from radar_gui.py.

    ⚠️ SECURITY: radar_gui.py contains a hard-coded Telegram bot token, and
    that file is committed to git — the token is in the repository history
    and must be treated as compromised. Revoke it via @BotFather.
    Nothing here contains a token; supply one via the environment variable
    ACOUSTIC_RADAR_TELEGRAM_TOKEN or via fusion_config.json if wanted.
    """
    telegram_enabled: bool = False
    telegram_token_env: str = "ACOUSTIC_RADAR_TELEGRAM_TOKEN"
    telegram_token: Optional[str] = None       # prefer the env var

    audio_logging_enabled: bool = True         # save WAV around each alarm
    uart_enabled: bool = True                  # bearing out on /dev/serial0


# ═══════════════════════════════════════════════════════════════
#  Root
# ═══════════════════════════════════════════════════════════════

@dataclass
class StationConfig:
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    switching: CameraSwitchConfig = field(default_factory=CameraSwitchConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    led: LedConfig = field(default_factory=LedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)

    # ── Derived quantities ─────────────────────────────────────

    def derive_visual_range_m(self, camera_id: int) -> Optional[float]:
        """
        Useful drone-detection range of one camera, from its own optics.

            box_width_px = focal_px * target_width_m / distance_m
        =>  distance_m   = focal_px * target_width_m / box_width_px

        evaluated at the smallest box the detector can still resolve.

        Returns None when the inputs are not calibrated, because a range
        derived from an unmeasured focal length would be a fabricated
        number wearing a physics costume.
        """
        focal = self.geometry.camera_focal_px.get(camera_id)
        width = self.geometry.drone_real_width_m
        min_px = self.fusion.min_detectable_box_px
        if not focal or not width or min_px <= 0:
            return None
        return float(focal) * float(width) / float(min_px)

    def activation_distance_m(self, camera_id: int) -> Optional[float]:
        """Range at which the target is deemed to have entered camera range."""
        if self.fusion.camera_activation_distance_m is not None:
            return float(self.fusion.camera_activation_distance_m)
        return self.derive_visual_range_m(camera_id)

    def release_distance_m(self, camera_id: int) -> Optional[float]:
        """Range at which it is deemed to have left again (hysteresis)."""
        if self.fusion.camera_release_distance_m is not None:
            return float(self.fusion.camera_release_distance_m)
        activation = self.activation_distance_m(camera_id)
        if activation is None:
            return None
        return activation * self.fusion.range_hysteresis_factor

    # ── Serialisation ──────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe_calibration(self) -> list[str]:
        """Human-readable list of what is and is not calibrated."""
        out: list[str] = []
        for cam_id, focal in sorted(self.geometry.camera_focal_px.items()):
            name = "FAR/IMX477" if cam_id == 0 else "NEAR/IMX708"
            if focal:
                rng = self.derive_visual_range_m(cam_id)
                rng_s = f"{rng:.0f} m" if rng else "n/a"
                out.append(f"camera {cam_id} ({name}): focal {focal:.0f} px, "
                           f"derived visual range ~{rng_s}")
            else:
                out.append(f"camera {cam_id} ({name}): focal NOT CALIBRATED")
        for cam_id, bore in sorted(self.geometry.camera_boresight_deg.items()):
            if bore is None:
                out.append(f"camera {cam_id}: boresight azimuth NOT CALIBRATED "
                           f"-> acoustic bearing cue disabled")
            else:
                out.append(f"camera {cam_id}: boresight {bore:.1f}°")
        if self.geometry.drone_real_width_m is None:
            out.append("drone_real_width_m NOT SET -> no visual distance")
        return out

    def check_bearing_frames(self, calibration_cfg: dict) -> list[str]:
        """
        Warn when the camera boresight was measured in a frame that has
        since been rotated by a change to doa_offset_deg.

        Returns human-readable warnings, empty when everything agrees. The
        corrected value is computed and shown, because "your boresight is
        stale" without the replacement number is only half a warning.
        """
        recorded = self.geometry.boresight_calibrated_at_doa_offset_deg
        if recorded is None:
            return []

        current = float(calibration_cfg.get("doa_offset_deg", 0.0))
        # ⚠️ HANDEDNESS FIRST. The frame can change in two ways and only one
        # of them can be repaired arithmetically.
        #
        # A rotation shifts every boresight by the same delta, so the
        # corrected value can simply be computed. A HANDEDNESS change
        # mirrors the frame, and a mirror is not a shift: telling an
        # operator to add a delta after the array's handedness was
        # re-measured would hand them a confidently wrong number, which is
        # the exact failure this whole check exists to prevent.
        recorded_hand = self.geometry.boresight_calibrated_handedness
        current_hand = calibration_cfg.get("doa_handedness")
        if (recorded_hand is not None and current_hand is not None
                and str(recorded_hand).upper() != str(current_hand).upper()):
            out = [f"the array's angle convention changed from "
                   f"{recorded_hand} to {current_hand} since the camera "
                   f"boresight was measured — the frame is now MIRRORED, "
                   f"not merely rotated"]
            for cam_id, bore in sorted(
                    self.geometry.camera_boresight_deg.items()):
                if bore is not None:
                    out.append(f"  camera {cam_id}: boresight {bore:.0f}deg is "
                               f"INVALID and CANNOT be corrected by adding an "
                               f"offset — it must be RE-MEASURED")
            out.append("  point the camera at a sound source, read the "
                       "bearing the HUD shows, and put that number in "
                       "geometry.camera_boresight_deg")
            return out

        delta = (current - float(recorded) + 180.0) % 360.0 - 180.0
        if abs(delta) < 0.5:
            return []

        out = [f"doa_offset_deg is now {current:+.0f}deg but the camera "
               f"boresight was measured at {float(recorded):+.0f}deg — the "
               f"installation frame has rotated {delta:+.0f}deg underneath it"]
        for cam_id, bore in sorted(self.geometry.camera_boresight_deg.items()):
            if bore is None:
                continue
            out.append(f"  camera {cam_id}: boresight {bore:.0f}deg is STALE, "
                       f"it should be {(float(bore) + delta) % 360.0:.0f}deg "
                       f"(or re-measure it)")
        out.append("  then set geometry.boresight_calibrated_at_doa_offset_deg"
                   f" to {current:+.0f} to silence this")
        return out


# ═══════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════

_SECTIONS = {
    "acoustic": AcousticConfig,
    "visual": VisualConfig,
    "switching": CameraSwitchConfig,
    "geometry": GeometryConfig,
    "fusion": FusionConfig,
    "ui": UIConfig,
    "led": LedConfig,
    "logging": LoggingConfig,
    "integrations": IntegrationConfig,
}


def load(path: Path | str = CONFIG_PATH) -> StationConfig:
    """
    Build the configuration, applying `fusion_config.json` if it exists.

    Unknown keys are reported on stdout instead of being dropped silently:
    a mistyped override that does nothing is one of the harder bugs to
    notice in a field-deployed system.
    """
    cfg = StationConfig()
    path = Path(path)
    if not path.exists():
        return cfg

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] could not read {path}: {exc} — using defaults")
        return cfg

    for section_name, values in raw.items():
        if section_name not in _SECTIONS:
            print(f"[config] unknown section '{section_name}' ignored")
            continue
        section = getattr(cfg, section_name)
        valid = {f.name for f in fields(section)}
        if not isinstance(values, dict):
            print(f"[config] section '{section_name}' must be an object")
            continue
        for key, value in values.items():
            if key not in valid:
                print(f"[config] unknown key '{section_name}.{key}' ignored")
                continue
            # JSON object keys are strings; camera-id maps must be ints.
            if key in ("camera_boresight_deg", "camera_focal_px") \
                    and isinstance(value, dict):
                value = {int(k): v for k, v in value.items()}
            setattr(section, key, value)

    return cfg


if __name__ == "__main__":
    c = load()
    print("=" * 66)
    print("fusion_config.py — effective configuration")
    print("=" * 66)
    print(json.dumps(c.to_dict(), indent=2, default=str))
    print("\nCalibration status:")
    for line in c.describe_calibration():
        print(f"   {line}")
    print("\nDerived camera-range gate (FAR camera):")
    print(f"   activation: {c.activation_distance_m(0)}")
    print(f"   release:    {c.release_distance_m(0)}")

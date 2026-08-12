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

    # [POLICY] Minimum smoothed P(drone) for the FUSION layer to treat an
    # acoustic contact as trustworthy enough to drive the camera.
    #
    # This is deliberately SEPARATE from the engine's own detection
    # threshold (model_config.json -> decision_threshold, currently 0.775,
    # chosen during training for a 2% false-alarm rate). The engine decides
    # "is this a drone"; this decides "am I confident enough to act on it".
    # None = inherit the engine's threshold, which is the sane default
    # because that threshold was fitted to real validation data whereas any
    # number typed here would not be.
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


# ═══════════════════════════════════════════════════════════════
#  User interface
# ═══════════════════════════════════════════════════════════════

@dataclass
class UIConfig:
    """Appearance and refresh policy of the single unified window."""

    window_name: str = "DRONE DETECTION STATION"
    fullscreen: bool = False

    # [POLICY] Upscale factor for the displayed frame. The camera produces
    # 640x480; 1.5 gives a 960x720 window that is readable on a monitor
    # without changing what the detector sees (the detector always runs on
    # the native frame).
    display_scale: float = 1.5

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

    # [POLICY] Redraw the whole HUD at most this often, in Hz. The camera
    # loop is independent of this; a slow UI never slows the sensors.
    max_ui_fps: float = 30.0


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

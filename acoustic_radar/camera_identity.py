#!/usr/bin/env python3
"""
camera_identity.py — Which physical camera is WIDE, and which is TELE.

═══════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS (audit finding C1)
═══════════════════════════════════════════════════════════════════

Before this file, the station told its two cameras apart like this:

    FAR_CAMERA_ID  = 0
    NEAR_CAMERA_ID = 1
    picam = Picamera2(camera_num=camera_id)

That is not an identity. It is a position in whatever order libcamera
happened to enumerate the CSI ports in, and NOTHING binds position 0 to the
25 mm lens. The order can change on a kernel or libcamera update, when a
cable is reseated, or when one sensor fails to probe and the survivor
becomes index 0. Both cameras here are the SAME sensor model (IMX477P), so
no sensor-model check can notice the swap either.

The consequence of a silent swap is not cosmetic. Role decides:

    focal length      -> every visual distance in metres
    boresight         -> where the acoustic cue is drawn
    switch direction  -> which lens is selected as the drone closes

so a swapped pair produces a station that is confidently wrong about all
three, with no error anywhere.

═══════════════════════════════════════════════════════════════════
⚠️ WHAT THIS FILE DELIBERATELY DOES **NOT** DO
═══════════════════════════════════════════════════════════════════

It does not guess. There is no fallback to enumeration order.

That is a deliberate, operator-approved choice, and the reason is specific
to this station: the two available sources of truth CONTRADICT each other.

    the code, before this change   index 0 = FAR  = narrow 28 deg lens
                                            i.e. index 0 = the 25 mm TELE
    the operator's hardware notes  index 0 = WIDE = the 6 mm lens

Both cannot be right, and nothing readable from software can settle it —
identical sensor models, identical tuning file, identical reported mode.
Picking either one would hard-code a 50/50 guess into the optics layer and
present it as fact, which is the exact failure C1 describes.

So when the mapping cannot be established from configuration, resolution
FAILS and the camera subsystem refuses to start. The acoustic subsystem is
unaffected. An operator sees the discovered cameras and the exact JSON to
paste; they never see a role that was assigned by chance.

═══════════════════════════════════════════════════════════════════
HOW A ROLE IS BOUND
═══════════════════════════════════════════════════════════════════

`Picamera2.global_camera_info()` reports, per camera, an `Id` which is the
device-tree path of the sensor, e.g.

    /base/axi/pcie@120000/rp1/i2c@88000/imx477@1a
    /base/axi/pcie@120000/rp1/i2c@80000/imx477@1a

The I2C controller address (`i2c@88000` vs `i2c@80000`) is the CSI port the
camera is physically plugged into. That is stable across reboots and across
enumeration reordering, because it describes the SOCKET, not the discovery
sequence. It changes only if the operator physically moves a cable — which
is precisely when the role SHOULD be re-declared.

So the operator records, once, in fusion_config.json:

    "geometry": {
      "camera_role_id_hint": {
        "WIDE": "i2c@88000",
        "TELE": "i2c@80000"
      }
    }

and this module matches each hint against the reported Ids.

⚠️ Whether the enumeration ORDER is stable on this Pi is still unknown and
still needs a reboot test — but it no longer matters, which is the point.
The hint is matched against the Id, so a reordering changes the index and
the role follows the socket. See HARDWARE_TEST_REQUIRED.md, test HW-1.

This module imports nothing from picamera2 and touches no hardware: it is a
pure function over the list of dicts, so the mapping rules are unit-testable
off the target machine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

# ═══════════════════════════════════════════════════════════════
#  Roles
# ═══════════════════════════════════════════════════════════════

#: The wide-angle camera (6 mm lens on this station). Selected when the
#: drone is close enough to overflow the narrow lens's field of view.
WIDE = "WIDE"

#: The long/telephoto camera (Kowa 25 mm on this station). The default,
#: used for everything beyond the switch threshold.
TELE = "TELE"

ROLES = (WIDE, TELE)

#: Both cameras on this station are Sony IMX477P (Arducam B0240/BO240).
#:
#: ⚠️ This is a CONSISTENCY check, never an identity check. Both cameras
#: report the same model, so a match proves only "this is one of the two
#: expected cameras" — it can never distinguish WIDE from TELE. The value of
#: checking it is catching a THIRD camera, or a swapped-in different module,
#: before its focal length is silently applied to the wrong optics.
EXPECTED_SENSOR_MODEL = "imx477"


class CameraIdentityError(RuntimeError):
    """
    Raised when a role cannot be bound to exactly one physical camera.

    Carries a fully-formed, multi-line operator message: what was found,
    what was expected, and the exact configuration to write. It is raised
    rather than returned so that no caller can accidentally continue with a
    partial mapping.
    """


# ═══════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════

def _camera_line(entry: Mapping[str, Any]) -> str:
    return (f"    index {entry.get('Num', '?')}  "
            f"model={str(entry.get('Model', '?')):10s}  "
            f"Id={entry.get('Id', '?')}")


def describe_cameras(cameras: Sequence[Mapping[str, Any]]) -> str:
    """Human-readable table of what libcamera reported. Used in errors."""
    if not cameras:
        return "    (no cameras reported by libcamera)"
    return "\n".join(_camera_line(c) for c in cameras)


def suggest_hint_config(cameras: Sequence[Mapping[str, Any]]) -> str:
    """
    The JSON block the operator must paste, pre-filled with real Ids.

    ⚠️ The two roles are listed in the order the cameras were enumerated,
    which is NOT a claim about which is which — it is a template. The
    accompanying error text says so explicitly, because a pre-filled
    suggestion is exactly the kind of thing that gets pasted unread.
    """
    ids = unique_fragments(cameras)
    while len(ids) < 2:
        ids.append("<no camera found>")
    return ('  "geometry": {\n'
            '    "camera_role_id_hint": {\n'
            f'      "{WIDE}": "{ids[0]}",\n'
            f'      "{TELE}": "{ids[1]}"\n'
            '    }\n'
            '  }')


def _fragment_candidates(camera_id: str):
    """
    Progressively more specific pieces of a device-tree Id, shortest first.

    The i2c controller node is tried first because it names the physical
    CSI socket and so means something to an operator; the longer forms are
    fallbacks that trade readability for uniqueness.
    """
    text = str(camera_id)
    parts = [p for p in text.split("/") if p]
    out = []
    for part in parts:
        if part.startswith("i2c@"):
            out.append(part)
    # Progressively longer suffixes: "imx477@1a", "i2c@88000/imx477@1a", ...
    for n in range(1, len(parts) + 1):
        out.append("/".join(parts[-n:]))
    out.append(text)
    # De-duplicate, preserving order.
    seen = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def unique_fragments(cameras: Sequence[Mapping[str, Any]]):
    """
    One Id fragment per camera, each matching EXACTLY ONE of them.

    ═══════════════════════════════════════════════════════════════
    ⚠️ FORENSIC REVIEW DEFECT D3 — WHY THIS IS NOT JUST `i2c@...`
    ═══════════════════════════════════════════════════════════════

    The previous version always returned the first `i2c@NNNNN` node. That
    is the right answer on a Pi 5 with two CSI connectors, where the two
    sensors hang off different I2C controllers — but it is NOT universal.
    Put both sensors behind a camera multiplexer, or on one controller at
    different chip addresses, and both fragments come out identical:

        "WIDE": "i2c@88000",  "TELE": "i2c@88000"

    `resolve_roles()` then correctly refuses that config as ambiguous — so
    nothing unsafe happened — but the operator had been handed, by this
    very function, a configuration guaranteed to be rejected on the next
    start-up, with no hint as to why.

    Each fragment is now CHECKED against every camera before being offered,
    falling back to longer and longer pieces of the path until it is
    unique. The last resort is the full Id, which always distinguishes two
    distinct device-tree nodes.
    """
    cameras = list(cameras or [])
    ids = [str(c.get("Id", "")) for c in cameras]
    out = []
    for own in ids:
        chosen = own                      # full Id: the always-works fallback
        for candidate in _fragment_candidates(own):
            matches = sum(1 for other in ids
                          if candidate.lower() in other.lower())
            if matches == 1:
                chosen = candidate
                break
        out.append(chosen)
    return out


# ═══════════════════════════════════════════════════════════════
#  Resolution
# ═══════════════════════════════════════════════════════════════

def validate_cameras(
    cameras: Sequence[Mapping[str, Any]],
    expected_model: Optional[str] = EXPECTED_SENSOR_MODEL,
) -> List[Mapping[str, Any]]:
    """
    Checks that do NOT need a role or a frame: are both cameras present,
    and are they the sensor the optics assume?

    Split out of resolve_roles() so CameraManager can run it BEFORE opening
    any device — there is no point starting two pipelines and grabbing
    probe frames only to discover a camera is missing. resolve_roles()
    still calls it, so a direct caller loses no validation.
    """
    cameras = list(cameras or [])

    if len(cameras) < 2:
        raise CameraIdentityError(
            f"the station needs 2 cameras, libcamera reports "
            f"{len(cameras)}:\n{describe_cameras(cameras)}\n"
            f"  check the CSI cables and `libcamera-hello --list-cameras`")

    # ⚠️ Consistency, never identity: both cameras report the same model, so
    # this can only catch a DIFFERENT module being plugged in — whose pixel
    # pitch would silently invalidate every theoretical focal length.
    if expected_model:
        wrong = [c for c in cameras
                 if expected_model.lower() not in str(
                     c.get("Model", "")).lower()]
        if wrong:
            raise CameraIdentityError(
                f"expected every camera to be {expected_model}, but found:\n"
                f"{describe_cameras(wrong)}\n"
                f"  all cameras reported:\n{describe_cameras(cameras)}\n"
                f"  if the hardware really changed, update "
                f"geometry.expected_sensor_model — the focal lengths and "
                f"the theoretical optics are derived from the IMX477 "
                f"pixel pitch and will be wrong for another sensor")
    return cameras


def resolve_roles(
    cameras: Sequence[Mapping[str, Any]],
    hints: Optional[Mapping[str, Optional[str]]],
    expected_model: Optional[str] = EXPECTED_SENSOR_MODEL,
) -> Dict[str, int]:
    """
    Bind each role to exactly one camera index. Raises, or returns a
    complete mapping — never a partial one.

    Args:
        cameras:  Picamera2.global_camera_info() output. Each entry needs
                  at least 'Id' and 'Num'; 'Model' enables the sensor check.
        hints:    role -> substring matched against that camera's 'Id'.
                  Matching is case-insensitive. None/empty for a role means
                  "not configured", which is a hard failure by design.
        expected_model: substring every camera's 'Model' must contain, or
                  None to skip the check.

    Raises:
        CameraIdentityError with an operator-ready explanation.
    """
    cameras = validate_cameras(cameras, expected_model)

    # ── Hints must exist. No enumeration-order fallback. ──
    hints = dict(hints or {})
    missing = [r for r in ROLES if not hints.get(r)]
    if missing:
        raise CameraIdentityError(
            f"camera role(s) {', '.join(missing)} are not configured, and "
            f"the optical check did not resolve them either.\n"
            f"  WHY NOT BY INDEX: both cameras are the same sensor model, so "
            f"an index is not an identity — if libcamera reorders them, WIDE "
            f"and TELE swap silently and every distance, cue and lens choice "
            f"is wrong with no error.\n"
            f"  cameras found:\n{describe_cameras(cameras)}\n"
            f"  The station normally works this out ITSELF by comparing the "
            f"two images: the 25 mm lens magnifies 4.17x more than the 6 mm "
            f"one, which is unmistakable in the pixels. That check needs "
            f"something with visible structure in view — point the cameras "
            f"at a textured scene (buildings, trees, a wall with detail), "
            f"not at empty sky, and restart.\n"
            f"  To pin the mapping permanently instead, add to "
            f"fusion_config.json:\n{suggest_hint_config(cameras)}\n"
            f"  (see HARDWARE_TEST_REQUIRED.md, test HW-1)")

    # ── Match each hint against exactly one camera ──
    mapping: Dict[str, int] = {}
    for role in ROLES:
        hint = str(hints[role]).lower()
        matched = [c for c in cameras
                   if hint in str(c.get("Id", "")).lower()]
        if not matched:
            raise CameraIdentityError(
                f"camera role {role} is configured as \"{hints[role]}\" but "
                f"no camera's Id contains that string.\n"
                f"  cameras found:\n{describe_cameras(cameras)}\n"
                f"  a camera may be unplugged, or the hint may be stale "
                f"after a cable was moved")
        if len(matched) > 1:
            raise CameraIdentityError(
                f"camera role {role} is configured as \"{hints[role]}\" but "
                f"that matches {len(matched)} cameras — the hint is "
                f"ambiguous:\n{describe_cameras(matched)}\n"
                f"  use a longer, unique fragment of the Id")
        mapping[role] = int(matched[0].get("Num", cameras.index(matched[0])))

    # ── Two roles must not resolve to the same device ──
    if mapping[WIDE] == mapping[TELE]:
        raise CameraIdentityError(
            f"both roles resolved to camera index {mapping[WIDE]} — the two "
            f"hints ({hints[WIDE]!r} and {hints[TELE]!r}) select the same "
            f"camera.\n  cameras found:\n{describe_cameras(cameras)}")

    return mapping


# ═══════════════════════════════════════════════════════════════
#  Theoretical optics
# ═══════════════════════════════════════════════════════════════

#: Sony IMX477 pixel pitch, millimetres. Datasheet value (1.55 um).
IMX477_PIXEL_PITCH_MM = 0.00155

#: Sony IMX477 full active pixel array, in pixels. Datasheet value.
IMX477_PIXEL_ARRAY = (4056, 3040)


#: Minimum normalised cross-correlation for an optical role check to be
#: believed at all. Two cameras a few centimetres apart see slightly
#: different viewpoints, so even a correct assignment never reaches 1.0.
OPTICAL_MIN_SCORE = 0.25

#: How far ahead the winning assignment must score. Both assignments are
#: scored; if they are close, the scene did not distinguish them (a blank
#: sky, a flat wall) and the answer is INCONCLUSIVE rather than a coin toss.
OPTICAL_MIN_MARGIN = 0.10


def _to_gray(frame):
    """[H,W] float32 grey, from a colour or mono frame. numpy in, numpy out."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 3:
        # Plain mean over channels: which of RGB/BGR this is does not
        # matter for a structural correlation, and guessing wrong would
        # only change the weighting slightly.
        arr = arr.mean(axis=2)
    return arr.astype("float32")


#: How far from the frame centre the telephoto view may sit, as a fraction
#: of the wide frame's width/height. The search is BOUNDED on purpose: an
#: unbounded slide would happily match a random distant patch of a repetitive
#: scene, which is a wrong answer rather than an honest refusal. 0.25 of the
#: wide frame is about +/-9.5 deg at the 6 mm lens's 37.9 deg field — far
#: more than any sane mast mounting, and far less than the whole frame.
OPTICAL_MAX_SHIFT_FRAC = 0.25

#: A correlation peak must beat the best peak OUTSIDE its own neighbourhood
#: by this much to count as a unique match. This is what rejects a
#: self-similar scene (a brick wall, a chain-link fence, ripples): such a
#: scene correlates equally well in many places, so no single alignment is
#: evidence of anything.
OPTICAL_MIN_PEAK_MARGIN = 0.15


def _match_peaks(wide_frame, tele_frame, ratio: float,
                 max_shift_frac: float = OPTICAL_MAX_SHIFT_FRAC):
    """
    Best and runner-up normalised correlation of `tele_frame`, shrunk by
    `ratio`, slid over the centre region of `wide_frame`.

    Returns (best, runner_up). (0.0, 0.0) when the comparison is impossible
    or either image is flat.

    ⚠️ THIS REPLACED A FIXED CENTRE CROP (finding C1, forensic review).
    The previous version assumed the telephoto view was the EXACT centre of
    the wide view. Measured, that assumption failed at one degree of
    pointing error: the telephoto field is only 9.4 deg across, so 1 deg
    displaces the image by ~68 output pixels and the correlation collapses
    from 0.90 to 0.02. Two cameras bolted to a mast are misaligned by more
    than that, so the check was returning INCONCLUSIVE in exactly the
    situation it was built for.

    Searching for the alignment instead of assuming it removes the
    dependency on mounting precision entirely, while the BOUND keeps the
    honest-refusal behaviour: a match found 9 degrees off-axis is still
    plausible optics; one found anywhere in the frame is just a coincidence.
    """
    import cv2
    import numpy as np

    wide = _to_gray(wide_frame)
    tele = _to_gray(tele_frame)
    h, w = wide.shape[:2]
    th, tw = tele.shape[:2]
    if min(h, w, th, tw) < 16 or ratio <= 1.0:
        return 0.0, 0.0

    # The telephoto view, reduced to the size it would occupy inside the
    # wide view if the claimed roles are right.
    small_w = max(8, int(round(w / ratio)))
    small_h = max(8, int(round(h / ratio)))
    small = cv2.resize(tele, (small_w, small_h),
                       interpolation=cv2.INTER_AREA)

    # Bounded search window, centred on the frame centre.
    pad_x = int(round(max_shift_frac * w))
    pad_y = int(round(max_shift_frac * h))
    x0 = max(0, (w - small_w) // 2 - pad_x)
    y0 = max(0, (h - small_h) // 2 - pad_y)
    x1 = min(w, (w + small_w) // 2 + pad_x)
    y1 = min(h, (h + small_h) // 2 + pad_y)
    window = wide[y0:y1, x0:x1]
    if window.shape[0] < small_h or window.shape[1] < small_w:
        return 0.0, 0.0

    # ⚠️ A flat image correlates with nothing, and TM_CCOEFF_NORMED divides
    # by the standard deviation — on a constant patch that is a division by
    # zero and yields NaN. Refuse before that happens: an empty sky is the
    # normal condition for this station, not an exceptional one.
    if float(np.std(small)) < 1e-6 or float(np.std(window)) < 1e-6:
        return 0.0, 0.0

    res = cv2.matchTemplate(window.astype(np.float32),
                            small.astype(np.float32),
                            cv2.TM_CCOEFF_NORMED)
    if res.size == 0 or not np.all(np.isfinite(res)):
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        if res.size == 0:
            return 0.0, 0.0

    best = float(res.max())
    by, bx = np.unravel_index(int(np.argmax(res)), res.shape)

    # Runner-up, measured OUTSIDE the winning peak's own neighbourhood.
    # Pixels adjacent to a true match correlate highly too; that is the
    # same peak, not a competing one. Excluding half a template width makes
    # the runner-up a genuinely different alignment.
    masked = res.copy()
    ry = max(1, small_h // 2)
    rx = max(1, small_w // 2)
    masked[max(0, by - ry):by + ry + 1, max(0, bx - rx):bx + rx + 1] = -1.0
    runner = float(masked.max()) if masked.size else -1.0
    return best, runner


def optical_assignment_score(wide_frame, tele_frame, ratio: float,
                             max_shift_frac: float = OPTICAL_MAX_SHIFT_FRAC
                             ) -> float:
    """
    How well `tele_frame` looks like a `ratio`x magnification of some part
    of `wide_frame`, as a normalised cross-correlation in [-1, 1].

    If the roles are as claimed, the telephoto image shows a region near
    the centre of the wide image, magnified. The best alignment within a
    bounded search is that correlation. A correct assignment scores high;
    the swapped assignment compares a magnified crop against a wide view
    and scores near zero.
    """
    return _match_peaks(wide_frame, tele_frame, ratio, max_shift_frac)[0]


def classify_roles_by_optics(frame_a, frame_b, ratio: float,
                             min_score: float = OPTICAL_MIN_SCORE,
                             min_margin: float = OPTICAL_MIN_MARGIN):
    """
    Decide, FROM THE IMAGES, which camera carries the long lens.

    ═══════════════════════════════════════════════════════════════
    WHY THIS EXISTS — it closes the software half of finding C1
    ═══════════════════════════════════════════════════════════════

    Binding a role to a device-tree Id makes the mapping STABLE, but it
    does not make it KNOWN: someone still has to state which socket holds
    which lens, and the two sources available for this station contradict
    each other. That looked like an unavoidable hardware question.

    It is not. The two lenses differ by 25/6 = 4.17x in magnification, and
    that is an enormous, unmistakable difference which is present in the
    pixels themselves. Both cameras point the same way from the same mast,
    so the telephoto view is a region NEAR THE CENTRE of the wide view,
    magnified. The assignment can therefore be MEASURED rather than
    declared.

    ⚠️ "NEAR the centre", not "AT the centre". The alignment is SEARCHED
    for within a bounded window rather than assumed, because the telephoto
    field is only 9.4 deg wide and one degree of mounting error is enough
    to destroy a fixed-centre correlation. See _match_peaks.

    Returns (role_of_a, role_of_b, detail) on a decisive result, or
    (None, None, detail) when the scene could not distinguish them.

    ⚠️ IT REFUSES RATHER THAN GUESSES. Three ways it declines:
      * both assignments score below `min_score` — nothing in view has
        enough structure to correlate (an empty sky, which is exactly what
        this station spends most of its time looking at);
      * the winning alignment is not UNIQUE — some other position in the
        search window correlates almost as well, which is what a
        self-similar scene (a brick wall, a fence, foliage) produces. A
        match that could equally have been found somewhere else is not
        evidence about the optics;
      * the two assignments are within `min_margin` of each other.
    In all three the caller falls back to the configured hint, and the
    hint's absence is still fatal. This never silently overrides a
    declared mapping; CameraManager uses it to CONFIRM one, and treats a
    contradiction as a hard error.

    ⚠️ SYNTHETIC VALIDATION ONLY. The tests behind this exercise modelled
    perturbations — pointing offset, exposure, noise, distortion. They do
    NOT establish that it works on the real optical pair; that is HW-1.
    """
    best_ab, runner_ab = _match_peaks(frame_a, frame_b, ratio)
    best_ba, runner_ba = _match_peaks(frame_b, frame_a, ratio)
    score_a_wide, score_b_wide = best_ab, best_ba
    detail = (f"NCC(A=WIDE,B=TELE)={score_a_wide:+.3f} "
              f"(2nd peak {runner_ab:+.3f}) "
              f"NCC(B=WIDE,A=TELE)={score_b_wide:+.3f} "
              f"(2nd peak {runner_ba:+.3f}) "
              f"(ratio {ratio:.2f}, min score {min_score:.2f}, "
              f"min margin {min_margin:.2f}, "
              f"min peak margin {OPTICAL_MIN_PEAK_MARGIN:.2f})")

    best = max(score_a_wide, score_b_wide)
    margin = abs(score_a_wide - score_b_wide)
    if best < min_score:
        return None, None, f"no structure to match — {detail}"

    # Uniqueness of the WINNING assignment's alignment.
    if score_a_wide >= score_b_wide:
        peak_margin = best_ab - runner_ab
    else:
        peak_margin = best_ba - runner_ba
    if peak_margin < OPTICAL_MIN_PEAK_MARGIN:
        return None, None, (f"the matching alignment is not unique — the "
                            f"scene repeats at this scale — {detail}")

    if margin < min_margin:
        return None, None, f"scene does not separate the two — {detail}"
    if score_a_wide > score_b_wide:
        return WIDE, TELE, detail
    return TELE, WIDE, detail


def theoretical_focal_px(lens_mm: float,
                         sensor_crop_width_px: float,
                         output_width_px: float,
                         pixel_pitch_mm: float = IMX477_PIXEL_PITCH_MM
                         ) -> float:
    """
    Horizontal focal length in PIXELS OF THE OUTPUT IMAGE, from optics.

        f_px = (f_mm / pixel_pitch_mm) * (output_width / sensor_crop_width)

    The first term is the focal length measured in sensor photosites; the
    second rescales it into the output image, because binning, cropping and
    ISP downscaling all change how many sensor pixels one output pixel
    covers.

    ⚠️ THIS IS A THEORETICAL VALUE AND MUST NEVER BE LABELLED CALIBRATED.
    It assumes the lens is exactly its nominal focal length, an undistorted
    pinhole projection, and that `sensor_crop_width_px` really is the
    region the ISP sampled. Real calibration measures the delivered image
    instead and absorbs all three.

    ⚠️ HOW FAR OFF IT IS, IS UNKNOWN. Do not attach an error bound to this
    number — an earlier version of this docstring claimed "right to a few
    percent", which was a guess about lens tolerance that ignored
    distortion and mounting and was never checked against anything. The
    error is UNKNOWN until HARDWARE_TEST_REQUIRED.md test HW-2 is run.

    Args:
        lens_mm:              nominal lens focal length (6.0 or 25.0 here)
        sensor_crop_width_px: width, in FULL-ARRAY sensor pixels, of the
                              region that maps onto the output. Equals the
                              full array width for a full-FOV mode, and the
                              ScalerCrop width for a cropped mode.
        output_width_px:      width of the delivered frame (640 here)
    """
    if lens_mm <= 0 or sensor_crop_width_px <= 0 or output_width_px <= 0:
        raise ValueError("theoretical_focal_px needs positive dimensions")
    focal_in_sensor_px = float(lens_mm) / float(pixel_pitch_mm)
    return focal_in_sensor_px * (float(output_width_px)
                                 / float(sensor_crop_width_px))

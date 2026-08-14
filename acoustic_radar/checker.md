# Acoustic Radar — Full Forensic Audit

Audit date: 2026-08-14
Repository state audited: commit `5ee4935` ("uqqqqqqq"), working tree **clean**
(no uncommitted changes, no staged changes — verified with `git status`).

Method: two independent read passes over the whole repository, plus read-only
execution (compileall, the full test suite, static analysis, module self-tests,
import-graph extraction, ONNX metadata inspection, and targeted probe scripts
run from stdin without creating files).

**No source file, configuration file or calibration file was modified.**
`logs/station.log` is written by the act of running the station and was
restored to its committed content afterwards; `git status` is clean.

---

## 1. Executive Summary

The system is structurally sound and materially better than it was: there is
now one canonical bearing definition, one place that converts a sensor
convention into it, a measured latency budget, and 202 passing checks. The
directional chain (radar / LED / camera cue) is derived from a single value,
and the deterministic chain test proves the three agree for all four cardinal
directions.

However, this audit found **25 distinct issues**, of which **5 are HIGH**. Two
of them are regressions introduced by the most recent changes and are not
covered by any test:

* the LED ring's "unverified convention" warning is **dead code** — it can
  never fire, so the ring aims a sector from a possibly-mirrored bearing and
  says nothing (BUG-001);
* on the frame where the camera switches, the **old camera's image is
  published with the new camera's id**, so the acoustic cue is projected with
  the wrong optics for that frame (BUG-002) — this is precisely the invariant
  the specification says must never be violated.

Three further HIGH items concern the direction subsystem: two wall-clock
timeouts in `doa.py` on a Raspberry Pi that has no RTC, and a tracker that
averages angles from two sources with different conventions into one number.

Latency remains **1279 ms measured** (mean, excluding window fill) against a
0.5–1.0 s requirement. The dominant remaining terms are structural, not bugs.

Nothing in this audit was verified on the real ReSpeaker, the real cameras or
the Hailo accelerator. Every hardware-dependent claim is marked.

---

## 2. Critical Findings

None. No issue found can corrupt data, destroy state, or crash the station
outright. The two most serious items are HIGH: they cause silently wrong
*direction* output, which is dangerous operationally but is contained.

---

## 3. High Findings

### BUG-001 — LED "unverified convention" warning is unreachable dead code

```
ID:            BUG-001
Severity:      HIGH
Category:      SOFTWARE BUG  (regression from the most recent change)
File:          respeaker_led.py
Line/function: 181 (frame_for_target), 747 (_warn_unverified),
               791 (_render), 102 (LedFrame docstring)
```

**Problem.** `frame_for_target()` assigns `calibrated = True` unconditionally:

```python
calibrated = True
if acoustic.coasting:
    return LedFrame(LedMode.COASTING, acoustic.bearing_deg, calibrated)
```

`LedFrame.calibrated` therefore can never be `False` anywhere in the station.

**Root cause.** When the camera cue was changed from "refuse to draw" to "draw
and label", the same relaxation was applied to the LED by hard-coding the
flag, rather than by keeping the flag and moving the response. The compensating
log warning was written but its guard tests the flag that is now always true.

**Evidence.** `grep -n calibrated respeaker_led.py` shows the only producer is
line 181 (`= True`) and line 1188 (`calibrated=True` in the manual bearing
test). Both consumers are therefore unreachable:

* line 747 `if self._warned_unverified or frame.calibrated or ...: return` —
  always returns, so the once-per-session warning **never fires**;
* line 791 `if (frame.bearing_deg is None or not frame.calibrated ...)` — the
  whole-ring fallback for an unverified convention is **never taken**.

**Impact.** With `doa_handedness` absent, the HUD prints `BEARING UNVERIFIED`
but the LED ring lights a confident directional sector with no indication
anywhere that the direction may be mirrored. The operator's most glanceable
indicator is the one with no caveat. The `LedFrame` docstring (line 102)
documents behaviour that cannot occur.

**Recommended fix.** Restore the flag's value from
`acoustic.bearing_calibrated` and decide explicitly what the ring does with
it — either keep aiming and let `_warn_unverified` fire, or fall back to the
whole-ring colour. Do not leave both consumers unreachable.

---

### BUG-002 — Camera switch publishes the old frame with the new camera id

```
ID:            BUG-002
Severity:      HIGH
Category:      SOFTWARE BUG
File:          camera_worker.py
Line/function: ~396-437, _run() main loop
```

**Problem.** Ordering inside one loop iteration:

1. `frame_raw = manager.get_frame()` — captured from the **currently active**
   camera (call it A);
2. `active = self._manager.get_active_camera()` → A;
3. `tracks = self._collect_tracks(active)` → correct for A;
4. `switched = self._policy.evaluate(...)`; **if it switched**,
   `active = self._manager.get_active_camera()` → now **B**;
5. `self.latest.publish(VisualObservation(frame=frame_bgr, active_camera=active, ...))`

The published observation therefore pairs **camera A's image** with
**camera B's id**.

**Root cause.** `active` is reused as both "which camera produced this frame"
and "which camera is now selected". They are the same value on every frame
except the switching frame.

**Evidence.** Read directly from `camera_worker.py`; the reassignment inside
`if switched:` occurs before the single `publish(...)` call, and `frame_bgr`
is derived from `frame_raw` captured before step 4.

**Impact.** `sensor_fusion.update()` reads `visual.active_camera` and uses it
for `projector.project(active_camera, ...)`, `activation_distance_m()` and
`release_distance_m()`. For one frame after every switch the acoustic cue is
projected through the wrong focal length and boresight — FAR is 1274 px /
28° HFOV, NEAR is 501.7 px / 65°, so the cue column can be displaced by
hundreds of pixels. The camera-range gate is likewise evaluated with the wrong
optics for that frame. The specification states: *"There must never be a frame
where camera 1 displays a cue calculated using camera 0 calibration."*
This is that frame, inverted.

**Mitigation already present.** `CameraSwitchConfig.debounce_interval_s = 0.5`
and `confirm_frames = 5` bound how often it can happen.

**Recommended fix.** Capture the camera id that produced the frame *before*
the policy runs, and publish that id with the frame; apply the new id from the
next capture onwards.

---

### BUG-003 — `DOATracker` release timeout uses the wall clock

```
ID:            BUG-003
Severity:      HIGH
Category:      SOFTWARE BUG
File:          doa.py
Line/function: 697, DOATracker.update()
```

**Problem.** `now = time.time() if now is None else now`, then
`if self.angle is not None and now - self._last_update > self.RELEASE_SEC`.

**Root cause.** Wall clock used for an interval measurement. Every caller in
the runtime path (`DOAProvider.update`, `radar.process_block`) omits `now`, so
the default applies.

**Evidence.** `target_state.py` line 13-15 states the project's own rule:
*"`time.monotonic()` is used, never `time.time()`: wall clock can jump (NTP
step, manual set)"*. `camera_manager.py` (line 80-88) and `audio_logger.py`
(line 67) both document having been converted for exactly this reason.
`doa.py` was not.

**Impact.** A Raspberry Pi has no battery-backed RTC and steps its clock when
NTP first syncs after boot — routinely by hours. A **backward** step makes
`now - _last_update` negative, so the 6 s release never fires and a dead track
is held indefinitely, which in turn keeps `ALARM_COASTING` supplied with a
stale bearing. A **forward** step releases a live track instantly.

**Recommended fix.** Use `time.monotonic()` as the default, consistent with the
rest of the project.

---

### BUG-004 — `HardwareDOA` stamps and ages readings with the wall clock

```
ID:            BUG-004
Severity:      HIGH
Category:      SOFTWARE BUG
File:          doa.py
Line/function: 418 (_loop), 464 (read)
```

**Problem.** `self._stamp = time.time()` in the polling thread, and
`if reading.ok and time.time() - stamp > max_age: return DOAReading(None, ...)`
in `read()`.

**Impact.** Same clock-jump exposure as BUG-003, on the primary bearing source.
A forward step makes every cached reading look older than `max_age = 2.0 s`,
so `DOAProvider` silently falls through to SRP-PHAT (or to no bearing at all)
even though the array is answering normally. A backward step serves arbitrarily
stale azimuths as fresh.

**Recommended fix.** `time.monotonic()` for both the stamp and the comparison.

---

### BUG-005 — The tracker mixes angles from two sources with different conventions

```
ID:            BUG-005
Severity:      HIGH
Category:      SOFTWARE BUG / DESIGN RISK
File:          doa.py
Line/function: DOATracker.update()/_accept(), DOAProvider._to_canonical()
```

**Problem.** `DOAProvider.update()` prefers the USB reading and falls back to
SRP-PHAT per block. `DOATracker._accept()` appends whatever arrives into one
`_history` deque and returns their circular mean. `DOATracker.update()` also
sets `self.source = reading.source` on every accepted reading.
`DOAProvider._to_canonical()` then applies `self.conventions[tracker.source]`
— a single convention — to that mixed mean.

**Root cause.** The per-source convention split was introduced at the
*conversion* step but the *smoothing* step upstream of it remained
source-agnostic.

**Evidence.** `_history` is appended in `_accept` with no source field; the
convention lookup uses only the most recent `tracker.source`.

**Impact.** USB and SRP-PHAT are documented as having different (and, for USB,
until recently unknown) zero and handedness. If USB drops out intermittently —
which is exactly what `DOAProvider.update`'s fallback exists to handle — the
smoothed angle becomes a mean of numbers expressed in two frames, and the
convention applied is whichever source happened to answer last. The resulting
bearing is wrong in a way no calibration can correct.

**Practical exposure today.** LOW on the user's current hardware: the array
delivers processed stereo, `channels_are_distinct()` refuses, and SRP-PHAT is
never the source. The defect is latent but real, and it activates the moment a
4-channel raw mode is used.

**Recommended fix.** Either keep one tracker per source and select between
their canonical outputs, or convert each reading to canonical **before**
smoothing so the history holds one frame only.

---

## 4. Medium Findings

### BUG-006 — Cue and range gate assume camera 0 when the camera is absent

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: sensor_fusion.py:425
```

`active_camera = 0 if visual is None else visual.active_camera`.

Verified by probe: with `visual=None` the cue is computed with a 28° HFOV
(camera 0) regardless of which camera the hardware actually has selected. The
same value drives `activation_distance_m()` / `release_distance_m()`, so the
camera-range **gate** — a fusion decision, not just a display — is evaluated
with camera 0's optics whenever the camera worker has not published. Camera 0
and camera 1 derive ranges of ~20 m and ~8 m respectively.

**Fix direction.** Carry the last known active camera across a visual gap, or
make the cue unavailable rather than defaulting.

---

### BUG-007 — Two target indications for one drone during a visual dropout

```
Severity: MEDIUM | Category: DESIGN RISK / documentation contradiction
File: sensor_fusion.py (CueRole docstring) vs hud.py:_draw_boxes
```

`CueRole` documents: *"the box becomes the primary indication and the search
region is withdrawn, so the display never shows two competing positions for
one drone."*

Verified by probe: after a visual dropout the fused snapshot has
`cue_role = ACOUSTIC ONLY` **and** `visual_track` populated (from
`_coasting_track()`), and `hud._draw_boxes` explicitly appends that coasting
track to `draw_list`. Both are drawn simultaneously.

This is arguably the *desired* reacquisition behaviour, and the two are styled
differently (amber dashed "LAST KNOWN" box vs red "ACOUSTIC ONLY" marker). The
defect is that the documented invariant is false and the overlap was never a
deliberate decision.

---

### BUG-008 — `overflow` never reaches the observation layer

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: radar.py (process_block), acoustic_worker.py (_to_observation)
```

`process_block` sets `st.overflow = False` and never sets it `True`.
`acoustic_worker._to_observation` faithfully copies
`overflow=bool(status.overflow)` — always `False`. The audio thread does detect
overflows (`overflowed` from `stream.read`) and rate-logs them, but never
writes the flag into the status. `AcousticObservation.overflow` is dead, and no
UI element can report dropped audio.

---

### BUG-009 — An unknown source string discards a valid tracker angle

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: doa.py, DOAProvider._to_canonical()
```

```python
conv = self.conventions.get(source)
if conv is None:
    return CanonicalBearing(confidence=..., source=source, reason=...)
```

`deg` defaults to `None`, so a perfectly good tracker angle is thrown away
rather than reported as uncalibrated. Only `"usb"` and `"srp"` are registered.
Not reachable in production today (those are the only sources that can set
`tracker.angle`), but it is reachable from `doa.py`'s own self-test, which
uses `source="test"`.

---

### BUG-010 — `UNKNOWN` handedness is silently treated as clockwise

```
Severity: MEDIUM | Category: SOFTWARE BUG (documentation contradicts behaviour)
File: bearing_frame.py:168-173, SourceConvention.to_canonical()
```

The class docstring states: *"UNKNOWN handedness is NOT silently treated as
clockwise."* The arithmetic does exactly that — the `COUNTER_CLOCKWISE` branch
negates, every other value (including `UNKNOWN`) adds. The only protection is
that `calibrated` is `False`, and since the consumers were changed to draw
anyway (see BUG-001), an unmeasured handedness now produces a confident
clockwise guess with only a HUD label to qualify it.

---

### BUG-011 — UART debounce uses the wall clock

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: radar.py:693, UartSender.send()
```

`now = time.time()` with `MIN_INTERVAL = 1.0`. A backward NTP step suppresses
all bearing output on `/dev/serial0` for the duration of the jump. The
integration enables UART by default (`IntegrationConfig.uart_enabled = True`).

---

### BUG-012 — Telegram cooldown uses the wall clock

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: telegram_alert.py:132, _try_send()
```

Same class as BUG-011: `COOLDOWN_SEC` measured against `time.time()`. A
backward step suppresses alerts; a forward step defeats the cooldown.
Telegram is disabled by default, which bounds the exposure.

---

### BUG-013 — One projector width is assumed for both cameras and for the display

```
Severity: MEDIUM | Category: DESIGN RISK
File: main.py:168, camera_cue.py, hud.py:_display_size / _draw_bearing_cue
```

`BearingProjector` is constructed once with `config.visual.frame_width` (640)
and is used for **both** cameras. `cue.x_px` is therefore in a 640-wide space.
`hud._display_size()` sizes the canvas from the **actual**
`visual.frame.shape`, and `_draw_bearing_cue` scales `cue.x_px` by
`ui.display_scale` only.

YOLO boxes follow the real frame size; the acoustic cue does not. If either
camera ever delivers a frame width other than `config.visual.frame_width`, the
cue lands in the wrong column while the boxes stay correct. Nothing asserts the
coupling. Currently benign — both streams are configured 640×480 — but silent.

---

### BUG-014 — YOLO boxes are not clipped to the frame

```
Severity: MEDIUM | Category: SOFTWARE BUG
File: TWO_CAMERAS_FIXED.py, _decode_head() (return path)
```

After un-letterboxing, `x1o/y1o/x2o/y2o` are returned with no clamp to
`[0, W] × [0, H]`. `hud._draw_boxes` clips for drawing, but
`sensor_fusion` passes `visual_track.center[0]` to
`BearingProjector.agreement_deg()` → `bearing_of_pixel()`, which extrapolates
`atan((x - W/2) / focal)` outside the sensor. An out-of-frame centre produces a
bearing beyond the lens's real field of view and can wrongly cancel — or wrongly
sustain — the acoustic search region via `cue_agreement_deg`.

**Verified not a bug:** the grid decode itself is correct —
`gy, gx = np.where(mask)` maps rows to *y* and columns to *x*, and
`cx_grid`/`cy_grid` use them accordingly. No X/Y swap.

---

## 5. Low Findings

### BUG-015 — Dead ternary in `to_canonical`
`bearing_frame.py:169-170`. `signed = (raw_deg if ... COUNTER_CLOCKWISE else raw_deg)`
— both branches identical; the following `if` does the real work. Harmless,
confusing.

### BUG-016 — Inconsistent zero handling between sources
`bearing_frame.py:206-233`. When uncalibrated, the `srp` branch returns
`zero_deg = 0.0` while the `usb` branch preserves `doa_offset_deg`. No
functional consequence today (both are marked uncalibrated) but the asymmetry
is unexplained.

### BUG-017 — Dead code in `telegram_alert.py`
`urllib.error` imported unused (line 17); `text = msg.get("text", "")`
assigned and never used (line 96).

### BUG-018 — Dead code / style in `TWO_CAMERAS_FIXED.py`
`fh, fw = bbox_arr.shape[0], bbox_arr.shape[1]` assigned and unused
(line 1336); ambiguous variable name `l` (line 1189); 93 semicolon-joined
statements. Pre-existing legacy file.

### BUG-019 — Private method called across a module boundary
`radar.py` calls `self.doa._to_canonical(DOAReading(None))` from
`process_block`. Works, but couples `radar` to `DOAProvider`'s internals.

### BUG-020 — Wall clock in `TWO_CAMERAS_FIXED.main()`
Lines 1468-1651 use `time.time()` for frame timing. **Not** in the station's
execution path (`camera_worker` imports classes and helpers, not `main()`), so
this affects only the standalone script.

### BUG-021 — `latency.BUDGET` is a process-wide singleton
Statistics accumulate across multiple `Station` runs in one process (the test
suite does this). Harmless for the deployed single-run process.

### BUG-022 — The repository ships an uncalibrated configuration
`radar_calibration.json` in git still contains
`doa_offset_deg: 0.0, doa_invert: false` and **no** `doa_handedness`; there is
**no** `fusion_config.json` at all. A fresh clone therefore starts uncalibrated
(bearing marked `UNCAL`), and the user's working calibration exists only on the
Pi. Any redeploy from git silently reverts the direction calibration.

### BUG-023 — `describe_calibration()` does not report handedness
`fusion_config.py`. The startup banner lists focal lengths and boresights but
never states whether the DOA convention was measured, so the single most
important calibration fact is absent from the banner (it appears only in the
`DOA:` line from `DOAProvider.describe()`).

### BUG-024 — `_check_camera_manager_is_current()` raises `SystemExit`
`TWO_CAMERAS_FIXED.py`. Raised at import time if `camera_manager` is stale.
`camera_worker` imports the module inside `try/except BaseException`, so it
degrades correctly — but only because that handler catches `BaseException`
rather than `Exception`. Fragile coupling worth noting.

### BUG-025 — Detector confirmations survive a gated block
`radar.py`, `Detector.update()`. In the gated branch the non-alarm path clears
`self.confirmations`, and the tracking path clears it only when not already in
an alarm state. The logic is correct but the two clears are expressed
differently from the ungated branch, making the invariant hard to verify by
reading.

---

## 6. Microphone / Audio Audit

| Aspect | Finding |
|---|---|
| Sample rate | 16 000 Hz, single source `features.SAMPLE_RATE`, matches `model_config.json` (`sample_rate: 16000`). **Consistent.** |
| Sample format | `float32` in `sd.InputStream`. Consistent. |
| Channel count | Discovered at runtime (`resolve_input`): ≥4 channels → open up to 6, else 2. **Measured, not assumed.** |
| Channel ORDER / which are microphones | **HARDWARE VERIFICATION REQUIRED.** `mic_channels` defaults to "the first N channels" — an assumption documented as such in `audio_io.py` and `calibration.py`. |
| Array geometry | `mic_positions_m` is a **DEFAULT** (43 mm square), never measured against the real board. **HARDWARE VERIFICATION REQUIRED.** |
| Preprocessing | `to_mono()` now averages only `mic_channels`; falls back to all channels for 1–2 channel inputs. Correct. |
| Feature extraction | `features.MelFrontend` shared by training and inference — verified: ONNX input `['batch',1,64,'time_frames']`, `N_MELS=64`, `target_frames=126`, forward pass succeeds. |
| Model input | Verified by execution. |
| Class mapping | `model_config.json` `class_names = {0: Фон, 1: ДРОН}`; code takes `softmax(...)[1]`. **Correct.** |
| Sliding window | `buffer[:-n] = buffer[n:]` — verified empirically that numpy handles the overlapping assignment correctly. **Not a bug.** |
| Buffer bounds | `_block_times` capped at 200; `AcousticTrackHistory` deques `maxlen=64/256`; `latency.Stage` capped at 512. **All bounded.** |
| Dropped samples | Detected (`overflowed`) and rate-logged, but **never surfaced** — see BUG-008. |
| Timestamps | `AcousticObservation.timestamp` uses `target_state.now()` = `time.monotonic()`. Correct. |
| Blocked capture | `stream.read()` blocks; the thread is a daemon and `join()` warns rather than hanging. Correct. |
| Cleanup | `_teardown` closes UART and engine, logs overflow count and block times. Correct. |

---

## 7. DOA / Coordinate Audit

Conventions established **by evaluating the code**, not by reading comments:

| Layer | Zero | Positive direction | Units | Wrap | Basis |
|---|---|---|---|---|---|
| Physical mic | UNKNOWN | UNKNOWN | deg | — | `mic_positions_m` is a default |
| SRP-PHAT output | +x axis of `mic_positions_m` | counter-clockwise | deg | 0..360 | **PROVEN** — steering vectors are `(cos θ, sin θ)` ⇒ `atan2` |
| XVF3800 USB | UNKNOWN | UNKNOWN | deg | 0..360 | Seeed firmware; undocumented here |
| Canonical | installation front | **clockwise** | deg | 0..360 | Defined in `bearing_frame` |
| Radar screen | up | clockwise | deg | 0..360 | **PROVEN** — `_polar_to_xy(b) = (cos(b−90), sin(b−90))`, y down |
| LED ring | `led_zero_offset_deg` | `led_index_clockwise` | index | mod N | CALIBRATE |
| Camera cue | `camera_boresight_deg` | right in image | px | ±HFOV/2 | **PROVEN** — `x = W/2 + f·tan(rel)` |
| Screen | top-left | x right, y down | px | — | OpenCV |

**Mirror handling.** `bearing_frame` proves that a handedness error cannot be
corrected by any additive offset (exact at one bearing, 180° out a quarter turn
away). Handedness is a separate parameter per source. The user's own five-point
calibration confirmed CCW for the XVF3800 (mean error 7.9° vs 70.9° for CW) —
but that result lives only on the Pi (BUG-022).

**Wraparound.** `wrap360`, `wrap180`, `angular_distance` and `circular_mean_deg`
are centralised in `bearing_frame`; `camera_cue.wrap_signed_deg` is an alias to
`wrap180`. Verified: `angular_distance(359, 1) = 2`, `circular_mean(350, 10) = 0`,
`wrap180(179 − (−179)) = −2`. **No wrap bugs found.**

**Double transforms.** None found in the current code. `apply_orientation` /
`unapply_orientation` survive in `doa.py` but the only remaining caller is
`calibrate.py`'s import list — verified that `cmd_doa` now fits with
`SourceConvention`. The LED no longer un-applies `doa_offset_deg`.

**Outstanding directional defects:** BUG-003, BUG-004, BUG-005, BUG-010.

---

## 8. Canonical Bearing Analysis — every transformation

```
XVF3800 firmware
  └─ HardwareDOA._query_once()
       parse_azimuth_values()          text → floats
       azimuths_to_degrees()           radians→degrees IF max|v| ≤ 2π+0.2   [heuristic]
       select_azimuth()                4 beams → 1 angle + ambiguity flag
  └─ (fallback) ArrayDOA.estimate()    GCC-PHAT → atan2 angle, CCW about +x

  ↓  DOATracker.update()/_accept()     circular mean over ≤8 readings
                                       ⚠ BUG-005: sources are mixed here

  ↓  DOAProvider._to_canonical()       ← THE ONLY convention conversion
       SourceConvention.to_canonical()   canonical = zero ± raw
                                       ⚠ BUG-010: UNKNOWN behaves as CW

  ↓  RadarStatus.angle_deg  →  AcousticObservation.bearing_deg   (canonical)

  ├─ radar_overlay._polar_to_xy(b)          identity + screen rotation
  ├─ bearing_frame.to_led_index(b, ...)     − led_zero_deg, ± handedness
  └─ camera_cue.project(cam, b, conf)       − camera_boresight_deg, tan projection
```

Each output layer applies exactly one transform, and none of them re-reads the
raw sensor angle. Confirmed by grep: no `apply_orientation` call remains in the
runtime path.

One heuristic worth flagging: `azimuths_to_degrees()` decides radians-vs-degrees
by testing whether every value is ≤ 2π. A firmware genuinely reporting degrees
while a target sits between 0° and 6° would be misread as radians. Low
probability, non-zero. **DESIGN RISK.**

---

## 9. Radar Audit

* Bearing→screen: **PROVEN correct** by evaluation — 0→up, 90→right,
  180→down, 270→left. A physical target on the left produces a target on the
  left, given a correct canonical bearing.
* Consumes `acoustic.bearing_deg` directly (`radar_overlay.py:280`); applies no
  correction of its own. Correct.
* Stale handling: `COL_STALE` when `acoustic_freshness is not FRESH`; header
  reports STALE even after TARGET_LOST (covered by test A2).
* Trail: bounded deque, filtered by `trail_s`.
* Scale selection: picks the smallest ring containing the target.
* Sweep: decorative, explicitly labelled as not a measurement.
* **No radar-specific defect found.** Its correctness is entirely inherited
  from the canonical bearing, so BUG-003/004/005/010 propagate here.

---

## 10. LED Audit

| Aspect | Finding |
|---|---|
| Control path | Subprocess to Seeed's `xvf_host.py` — the only mechanism present in the repo. No invented USB commands. |
| Command discovery | Queries the firmware's own command list; uses only names it reports; refuses ambiguous matches. Correct. |
| Argument form | `--values` flag, taken from the utility's own usage string. Correct (was a real hardware failure, now fixed). |
| Arity | **Measured** by reading the command back; falls back to parsing the device's `value count is N` error. Correct. |
| Auto mode | Searches for `LED_AUTO_MODE`/`LED_EFFECT`; writes `auto_mode_off_value`. The effect enumeration is **HARDWARE-DEPENDENT** and unverifiable here. |
| LED count | Configuration; `None` ⇒ subsystem reports UNAVAILABLE. Correct. |
| Index mapping | `to_led_index(canonical, N, led_zero_deg, clockwise)` — canonical in, no microphone calibration read. Correct. |
| Zero / direction | **CALIBRATION REQUIRED**, unverified on this hardware. |
| Sector | `led_sector()` wraps modulo N. Verified for wrap across index 0. |
| Colour packing | `0xRRGGBB` assumed; `led_value_order` exists to flip it. **HARDWARE VERIFICATION REQUIRED.** |
| RED/BLUE priority | `frame_for_target` is a pure function of the fused state; test proves 50 repaints give one frame. No flicker path. |
| Coasting | `LedMode.COASTING` uses the same red as `ALARM`; test asserts the two colours are equal. Correct. |
| Duplicate writes | Shadow buffer; per-pixel diff; whole-ring shortcut. Measured 8% write/submit ratio. Correct. |
| Update rate | `min_write_interval_s = 0.15`; no busy-spin (the `_wake.set()` that caused one was removed). Correct. |
| Failure handling | Backoff, retry, UNAVAILABLE status, never raises into the station. Correct. |
| Cleanup | `stop()` joins with timeout, restores searching colour, logs write count. Correct. |
| **Defect** | **BUG-001** — the unverified-convention path is dead code. |

**Is `physical LEFT → LED LEFT` guaranteed by the code?** No. It is guaranteed
*given* correct `led_zero_offset_deg`, `led_index_clockwise` and a calibrated
source convention. Two of those three are unverified on this hardware, and
BUG-001 removes the only warning that would say so.

---

## 11. Camera Acoustic Cue Audit

Intended flow — verified present and tested:

```
acoustic bearing → active camera calibration → image column
   → CueRole.SEARCH  ("ACOUSTIC ONLY — OCCLUDED")
YOLO agrees (≤ cue_agreement_deg) → CueRole.CONFIRMED, region withdrawn
YOLO lost → CueRole.SEARCH returns on the next update
```

Verified by test (scenarios D, E, F) and by probe.

* **Projection:** `x = W/2 + f·tan(rel)`, `rel = wrap180(bearing − boresight)`.
  Correct, delegated to `bearing_frame`.
* **Out-of-FOV:** returns `in_view=False` with a side, never a clamped pixel.
  Correct — the HUD draws an edge arrow.
* **Uncertainty:** projected through the same lens; `uncertainty_exceeds_fov`
  reported. The marker is a fixed-size reticle (verified constant at 29 px
  while the column varied 149→320 px) and sits on the projected bearing to
  within 1 px at every angle tested.
* **Elevation:** never asserted — open brackets, arrows, and an
  `ELEV NOT MEASURED` caption.
* **Stale acoustic:** `Freshness.LOST` ⇒ `CueRole.NONE`. Verified.
* **Defects:** BUG-006 (camera-0 fallback), BUG-007 (dual indication),
  BUG-013 (width coupling), BUG-014 (unclipped box centres feed the agreement
  check).
* **`cue_agreement_deg = 20°` has no force on the FAR camera.** Its HFOV is
  28°, so no box inside the frame can disagree with the boresight by 20°.
  The disagreement test is effectively NEAR-only. Documented in the test but
  not in the configuration.

---

## 12. Two-Camera Audit

* `CameraManager.get_frame()` returns `None` on any failure and fails over
  after repeated failures — verified by test C1.
* Published frame is a **fresh buffer** (`cv2.cvtColor` allocates); the tracker
  only reads from it (`crop = frame_bgr[...]`, no writes). **Frame ownership is
  safe** — verified, not assumed.
* Switching: distance hysteresis (1.5 m / 2.0 m) + `confirm_frames = 5` +
  `debounce_interval_s = 0.5`. Oscillation test passes (72 switches → 0).
* `allow_acoustic_fallback = False`, documented as deliberately rejected
  because acoustic range cannot resolve a 1.5 m decision.
* **Defect: BUG-002** — the switching frame publishes a mismatched
  (frame, camera-id) pair.
* **Defect: BUG-006** — no active-camera memory across a visual gap.
* **Risk: BUG-013** — one projector width shared by both cameras.

---

## 13. YOLO / Hailo Audit

Static only — **HARDWARE VERIFICATION REQUIRED** for everything below that
touches the accelerator.

* HEF load: `HailoInference` verifies `HEAD_CONFIG` against the HEF and prints
  "All names match" — a real self-check, not an assumption.
* Sigmoid: auto-detected from the observed logit range over the first calls,
  then **locked**. Sound, but the lock is permanent for the session; a
  pathological first few frames would lock the wrong branch.
* LTRB decode mode: auto-detected by vote (`stride` vs `pixel`), then locked,
  with `stride` as the documented default while voting. Same lock caveat.
* Grid decode: **verified correct** — `gy, gx = np.where(mask)`, `cx_grid` from
  `gx`, `cy_grid` from `gy`. **No X/Y swap.**
* Letterbox reversal: `(p − pad) / ratio` applied to all four coordinates
  consistently.
* NMS: `cv2.dnn.NMSBoxes` over merged heads with `conf_thresh`/`iou_thresh`
  from configuration.
* Class mapping: single class head (`cls_arr[:, :, 0]`).
* **Defect: BUG-014** — boxes are not clipped to the frame.
* Duty cycle: `frame_skip = 2` while a confirmed track exists, every frame
  otherwise — preserved from the original.
* Channel order into the detector: `swap_detector_channels = False` preserves
  the original behaviour; the correct value is **UNKNOWN** and documented as
  requiring an A/B test against a real drone.

---

## 14. Sensor Fusion Audit

* Seven states declared; all seven have a priority mapping (verified
  programmatically — no state without a priority).
* `_next_state` is the single transition table; every transition is logged once
  with a reason.
* Visual confirmation is evaluated **before** acoustic loss, so a visible
  target is never dropped because the microphone lost it (test 5).
* `in_range is None` (unmeasurable) is distinguished from `False` (measured and
  out of range) — a documented past bug, still correctly handled (test F1).
* `_last_contact_time` prevents a long-held track being declared lost instantly
  (test F2).
* Coasting observations are excluded from the velocity history, so a held
  bearing cannot fabricate a "STEADY" closure.
* **No impossible or unreachable transition found.** `TARGET_LOST → SEARCHING`
  is time-based; every other edge is condition-based and reachable.
* **Defects:** BUG-006, BUG-007.

---

## 15. Latency Audit

Measured on the development PC (x86 Windows), **NOT HARDWARE VERIFIED on the
Raspberry Pi 5**:

```
audio block          500 ms   (configuration)
analysis window     2000 ms   (configuration)
confirmation        1000 ms   (2 blocks × 500 ms, exact)
audio_wait     mean  500 ms   p95 500   (loop period; latency contribution ≈ half)
features       mean   13 ms   p95  63
inference      mean    0-15ms p95  16
block_total    mean    3-16ms p95  16   max 63   (budget 500)
publish_to_fuse mean  17-26ms p95  63           (was 357 ms before the wake-on-audio fix)
hud_render          NOT MEASURED — requires a run with the UI active
led_write           NOT MEASURED — requires the array
────────────────────────────────────────────
TOTAL (mean)       1279 ms, EXCLUDING window fill
```

**Against the 0.5–1.0 s requirement: NOT MET (1.28 s).** The gap is structural,
not a defect:

* half a block of read quantisation (250 ms) — reducible only by a shorter hop;
* 1000 ms of confirmation — that *is* the evidence; shortening it weakens the
  decision;
* **window fill is not measured at all.** A drone audible for 0.5 s occupies a
  quarter of the 2 s analysis window and the model was trained on windows that
  were entirely drone. This term needs a real drone and a reference
  microphone. **HARDWARE VERIFICATION REQUIRED.**

`acoustic.block_seconds` exposes the hop with the measured trade-off documented
(0.25 s hop + 3 confirmations ≈ 0.90 s, at 2× inference CPU). The default is
deliberately unchanged.

---

## 16. Hysteresis / Coasting Audit

| Parameter | Value | Verified |
|---|---|---|
| Audio block | 500 ms | Yes, from `radar.BLOCK_SEC` |
| `confirm_blocks` | 2 → 1000 ms | Yes, by test and by driving the real `Detector` |
| `P_START` | 0.775 (from `model_config.json`) | Yes |
| `P_HOLD` | 0.50 absolute | Yes |
| `miss_tolerance` | 6 → exactly 3000 ms | Yes, counted block by block |
| Confirmations consecutive? | Yes — cleared on a miss while in TRACK | Yes, regression test present |
| Coasting on a gated (silent) block | Yes | Yes |
| Return from coasting | Same block the signal returns | Yes |

Behaviour matches the specification exactly. The historical off-by-one
(`misses > tolerance` giving 3.5 s) is fixed and pinned by a test asserting
3000 ms.

**One coupling worth noting:** `acoustic.lost_after_s = 3.0` equals the coasting
window exactly. `acoustic_worker` warns if coasting **exceeds** it, but they are
equal, so the fusion layer's "lost" and the engine's "give up" fire on the same
block. No margin. Not a bug; a tight coupling with no slack.

---

## 17. Timing Audit

Wall-clock (`time.time()`) uses found in the runtime graph:

| File | Line | Purpose | Verdict |
|---|---|---|---|
| `doa.py` | 418, 464 | DOA reading stamp / staleness | **BUG-004, HIGH** |
| `doa.py` | 697 | Tracker release timeout | **BUG-003, HIGH** |
| `radar.py` | 693 | UART debounce | **BUG-011, MEDIUM** |
| `telegram_alert.py` | 132 | Alert cooldown | **BUG-012, MEDIUM** |
| `TWO_CAMERAS_FIXED.py` | 1468-1651 | Standalone loop FPS | **BUG-020, LOW** (not in station path) |

Correct (`time.monotonic`): `target_state.now()`, `acoustic_worker`,
`camera_worker`, `camera_manager`, `audio_logger`, `latency`, `main`,
`respeaker_led`, `TWO_CAMERAS_FIXED._compute_dt`.

No negative-duration guard exists anywhere; every timeout above assumes a
monotonic difference.

---

## 18. Threading / Concurrency Audit

| Thread | Created by | Writes | Read by |
|---|---|---|---|
| `acoustic` (daemon) | `AcousticWorker.start` | `latest`, `health`, `BUDGET` | UI thread |
| `camera` (daemon) | `CameraWorker.start` | `latest`, `health` | UI thread |
| HardwareDOA `_loop` (daemon) | `DOAProvider.__init__`, on the audio thread | `_reading`, `_stamp` (under `_lock`) | audio thread |
| `led` (daemon) | `RespeakerLed.start` | device, `BUDGET` | — |
| Telegram `_broadcast` (daemon) | per alert, cooldown-bounded | network | — |
| main | process | UI | — |

* All cross-thread state goes through `LatestValue` (single lock, reference
  assignment). Torn-read test passes at 710 k publishes / 582 k reads.
* Observations are frozen dataclasses — no half-updated record can be observed.
* `latency.Stage` is lock-protected.
* `RespeakerLed._last_submit` / `_submit_stamp` are written by the UI thread and
  read by the LED thread **without a lock**. Float assignment is atomic under
  CPython and both feed only coarse timeouts. **Acceptable, but unsynchronised
  by design rather than by accident — worth recording.**
* `HardwareDOA` thread is never joined; it is a daemon and `stop()` sets an
  event it waits on. A `subprocess.run(timeout=3)` in flight delays exit by up
  to 3 s. Acceptable.
* No lock is ever held across a blocking call. **No deadlock path found.**
* UI never blocks a worker: `cv2.waitKey` releases the GIL.

---

## 19. Memory / Resource Audit

Every append site in the runtime graph was enumerated and checked:

| Structure | Bound |
|---|---|
| `AcousticTrackHistory._range_hist` | `deque(maxlen=64)` |
| `AcousticTrackHistory._trail` | `deque(maxlen=256)` |
| `DOATracker._history` | `deque(maxlen=8)` |
| `DOATracker._pending` | cleared at 3 — never exceeds `JUMP_CONFIRMATIONS` |
| `SensorFusion.transitions` | capped at 200, trimmed to 100 |
| `AcousticWorker._block_times` | capped at 200 |
| `latency.Stage._samples` | capped at 512 |
| `TrajectoryHistory` | 512 records (regression-tested) |
| `AudioLogger._record_chunks` | capped, tested against a hostile clock |

**No unbounded growth found.**

Resources: engine closed, UART closed, LED device released, cameras closed by
`CameraManager`, `cv2.destroyAllWindows` pumped. `_canvas` is reallocated only
on a size change.

---

## 20. Configuration / Calibration Audit

| Value | Status | Notes |
|---|---|---|
| `doa_offset_deg` | **MEASURED on the Pi** (353.7) | Not in git — BUG-022 |
| `doa_handedness` | **MEASURED on the Pi** (CCW, residual 7.9°) | Not in git — BUG-022 |
| `doa_invert` | legacy mirror of handedness | Written for compatibility |
| `srp_zero_deg` | `null` — NOT CALIBRATED | SRP unused on this hardware |
| `mic_positions_m` | **DEFAULT, never measured** | HARDWARE VERIFICATION REQUIRED |
| `mic_channels` | `[]` ⇒ "first N" assumption | HARDWARE VERIFICATION REQUIRED |
| `camera_boresight_deg` | user set 0.0 — **measured under the OLD (mirrored) frame** | **INVALID after the handedness change** |
| `boresight_calibrated_at_doa_offset_deg` | not set | Staleness check silent |
| `boresight_calibrated_handedness` | not set | Mirror check silent |
| `camera_focal_px[0]` | MEASURED (1274, two calibrations agreeing to 2.8%) | Trustworthy |
| `camera_focal_px[1]` | **UNVERIFIED** (502 vs 1003 unresolved) | All NEAR distances inherit this |
| `led_count` | user set 12; confirmed by the device's own arity error | Good |
| `led_zero_offset_deg` | 0.0 — **UNVERIFIED** | CALIBRATION REQUIRED |
| `led_index_clockwise` | `True` default — **UNVERIFIED** | CALIBRATION REQUIRED |
| `led_value_order` | `"rgb"` assumed | HARDWARE VERIFICATION REQUIRED |
| `range_ref_*` | user measured (3 m @ −22 dBFS) | Plausible; the synthetic-config warning should no longer fire |
| `noise_floor_dbfs` | −55 | User set |
| `range_spreading_db` | 22.0 default | Not measured |
| `drone_real_width_m` | 0.25 assumed | Scales every visual distance linearly |
| `min_detectable_box_px` | 16 assumed | Sets the whole camera-range gate |
| `cue_agreement_deg` | 20° | No force on the FAR lens (see §11) |

---

## 21. UI Audit

Correctly distinguished states, verified in code and by rendering:

`SEARCHING` · `ACOUSTIC CONTACT` · `ACOUSTIC TRACKING` · `VISUAL ACQUISITION` ·
`VISUAL TRACKING` · `VISUAL LOST - ACOUSTIC` · `TARGET LOST` ·
`COASTING` + `HOLD n/6` + `LAST BRG` · `ACOUSTIC ONLY — OCCLUDED` ·
`BEARING UNVERIFIED` · `(UNCAL)` · `COARSE` · `ELEV NOT MEASURED` ·
`CAMERA SIGNAL LOST` / `CAMERA STALLED` + age · `STALE` tag · `N/A` · `UNKNOWN`

* Acoustic estimate vs visual confirmation are visually separated: red dashed
  reticle + "ACOUSTIC ONLY" against green solid YOLO boxes; `C_VISUAL` is
  reserved for real detections.
* A frozen camera frame is dimmed to 28%, bordered, and captioned with its age
  — the strongest anti-staleness measure in the project.
* Acoustic distances always carry `~` and an interval.
* **Gap:** during a visual dropout both a coasting box and an acoustic reticle
  are shown (BUG-007).
* **Gap:** the LED, the most glanceable indicator, carries no caveat at all
  (BUG-001).

---

## 22. Error Handling Audit

| Failure | Behaviour | Verdict |
|---|---|---|
| Microphone missing | `resolve_input` falls back to the system default with a printed warning; stream open retries 6→2→1 channels | Degrades |
| Audio model missing | `RadarEngine` raises `SystemExit`; `_run` catches `BaseException` → subsystem OFFLINE, process survives | Correct |
| Model/front-end mismatch | `SystemExit` with an explicit message | Correct |
| Camera 0 or 1 missing | `CameraManager` opens whichever it can; fails over after repeated failures | Correct (tested) |
| Both cameras missing | `camera subsystem could not start`, acoustic continues | Verified in a live run |
| HEF / Hailo missing | `_detector_available = False`, tracking continues without detection | Correct |
| LED unavailable | UNAVAILABLE + backoff retry, never raises | Correct (tested) |
| `xvf_host.py` missing | DOA reports unavailable once; SRP fallback | Correct |
| Worker thread stalls | Station-level watchdog marks it OFFLINE regardless of what the worker claims | Correct (tested) |
| UI window closed | `getWindowProperty` → clean exit | Correct |
| Ctrl+C / SIGTERM | Signal handler → `station.stop()` → orderly shutdown | Correct |

**No failure path was found that takes an unrelated subsystem down.**

---

## 23. Startup / Shutdown Audit

Startup order verified by a live headless run: logging → config → banner
(calibration status, frame-consistency check, range gate, switching, LED) →
acoustic worker → camera worker → LED (last, never waited on) → UI loop.

Shutdown: LED stopped **first** (so the ring is restored while the process is
healthy) → camera and acoustic `stop()` → `join(timeout=3)` → windows destroyed
→ session statistics + latency budget logged.

`Station.inject_bearing` is declared on the class so the health helpers work on
a partially-constructed `Station` (the watchdog test builds one with `__new__`).

**No defect found.**

---

## 24. Existing Test Coverage

202 checks, all passing. Genuinely covered:

* Fusion state machine — 10 scenarios including handover, both-lost,
  oscillation, microphone failure, camera failure, load.
* Detector timing — 500 ms block, 2 confirmations = 1000 ms, 3000 ms coasting,
  P_HOLD 0.50, consecutive-confirmation guard, gated-block coasting, and a
  regression test proving the old tuning really took 3.5 s.
* Coordinate chain — 0/90/180/270 through radar + LED + camera, **plus a
  negative control** (a physically mirrored ring must fail on two axes).
* Mirror-vs-offset algebra.
* Per-source conventions; `doa_invert` alone rejected as evidence.
* Wrap: 359→1, circular mean, 179/−179.
* Beam tie determinism and ambiguity flagging; tracker does not flip on
  ambiguous readings but does follow a real manoeuvre.
* SRP blind band; degenerate pairs excluded; processed stereo still refused.
* `to_mono` channel selection.
* Camera cue: scenarios D/E/F/G, uncalibrated handling, LOST withdrawal,
  disagreeing box, per-camera projection.
* Marker: constant size vs confidence, configured size, on-bearing at every
  angle.
* LED: A–J including a stand-in that reproduces the real XVF3800 argparse
  signature and arity error.
* Honesty, staleness, watchdogs, thread safety, bounded histories.

## 25. Missing Tests

1. **BUG-002** — no test publishes a `VisualObservation` across a camera switch
   and asserts the frame and the id belong to the same camera.
2. **BUG-001** — no test asserts that an uncalibrated bearing changes anything
   about the LED, which is why the dead code was not caught.
3. **BUG-003/004** — no test injects a clock jump.
4. **BUG-006** — no test covers `visual=None` while the hardware's active
   camera is 1.
5. **BUG-008** — no test asserts `overflow` propagates.
6. **BUG-013** — no test uses a frame whose width differs from
   `config.visual.frame_width`.
7. **BUG-014** — no test feeds an out-of-frame box centre to `agreement_deg`.
8. No test drives `RadarEngine.process_block` end to end with synthetic audio
   (the detector FSM is tested directly, the DSP path is not).
9. No test exercises `DOAProvider.update`'s USB→SRP fallback, which is where
   BUG-005 lives.
10. Hailo/HEF decode has **no** test — the auto-detect locks (sigmoid, LTRB)
    are unverified in software.

**False confidence to be aware of:** the suite models sensors as synthetic
observation streams. It cannot detect a wrong physical channel order, a wrong
array geometry, a mirrored LED ring, or a wrong colour packing — all of which
are live uncertainties.

---

## 26. Hardware Verification Required

1. XVF3800 channel order — which channels are physical microphones
   (`mic_channels`).
2. Array geometry `mic_positions_m` against the real board.
3. LED colour packing (`led_value_order`) — is it `0xRRGGBB`?
4. `LED_EFFECT` value that actually disables the firmware animation.
5. LED ring index direction and zero position.
6. Camera 1 focal length (502 vs 1003 unresolved).
7. `swap_detector_channels` — A/B against a real drone.
8. Hailo decode: sigmoid and LTRB auto-detect locks.
9. `min_detectable_box_px` — walk a drone out until detection becomes
   intermittent.
10. Window-fill latency with a real drone and a reference microphone.
11. Pi-side `hud_render`, `led_write`, `block_total` under full load.
12. False-alarm rate at 2 confirmations instead of 5.

## 27. Calibration Required

1. **`camera_boresight_deg` — re-measure.** The frame was MIRRORED (CW→CCW),
   not rotated; no offset can repair the old value.
2. Record `boresight_calibrated_at_doa_offset_deg: 353.7` and
   `boresight_calibrated_handedness: "CCW"` so the staleness check can fire.
3. `led_index_clockwise` first, then `led_zero_offset_deg`
   (`respeaker_led.py --bearing sweep`).
4. `srp_zero_deg` — only if raw 4-channel mode is ever used.
5. `mic_channels`, `mic_positions_m`.
6. `range_spreading_db` (currently the 22.0 default).
7. `drone_real_width_m`, `min_detectable_box_px`.
8. **Commit the calibrated `radar_calibration.json` and `fusion_config.json`
   to git** (BUG-022) — they exist only on the Pi.

## 28. Dependency Chain

```
mic_channels + mic_positions_m
        ↓ (SRP-PHAT geometry only)
   srp_zero_deg

XVF3800 handedness + offset   ← calibrate.py doa, TWO points minimum
        ↓
   CANONICAL BEARING
        ├──────────────┬────────────────────┐
        ↓              ↓                    ↓
   radar screen   led_zero_offset_deg   camera_boresight_deg
                  led_index_clockwise           ↓
                        ↓                camera acoustic cue
                   LED sector                   ↓
                                        cue_agreement_deg ← camera_focal_px

range_ref_distance_m + range_ref_level_dbfs + range_spreading_db
        ↓
   acoustic distance
        ↓
   camera-range gate ← min_detectable_box_px + drone_real_width_m
                                              + camera_focal_px
```

**Invalidation rules:**

* Re-running `calibrate.py doa` invalidates `camera_boresight_deg` **and**
  `led_zero_offset_deg`. If the *handedness* changed, both must be
  **re-measured**, not adjusted.
* Changing `camera_focal_px` invalidates the camera-range gate and every
  visual distance.
* Changing `drone_real_width_m` scales every visual distance linearly.

---

## 29. Complete Bug Index

| ID | Sev | Category | File | Summary |
|---|---|---|---|---|
| BUG-001 | HIGH | SOFTWARE BUG | respeaker_led.py | LED unverified-convention warning and fallback are dead code |
| BUG-002 | HIGH | SOFTWARE BUG | camera_worker.py | Switch frame pairs old image with new camera id |
| BUG-003 | HIGH | SOFTWARE BUG | doa.py:697 | Tracker release timeout on the wall clock |
| BUG-004 | HIGH | SOFTWARE BUG | doa.py:418,464 | DOA reading staleness on the wall clock |
| BUG-005 | HIGH | SOFTWARE BUG | doa.py | Tracker mixes sources with different conventions |
| BUG-006 | MEDIUM | SOFTWARE BUG | sensor_fusion.py:425 | Camera 0 assumed when visual is absent |
| BUG-007 | MEDIUM | DESIGN RISK | sensor_fusion.py / hud.py | Two target indications during a visual dropout |
| BUG-008 | MEDIUM | SOFTWARE BUG | radar.py / acoustic_worker.py | `overflow` never set, never surfaced |
| BUG-009 | MEDIUM | SOFTWARE BUG | doa.py | Unknown source discards a valid angle |
| BUG-010 | MEDIUM | SOFTWARE BUG | bearing_frame.py | UNKNOWN handedness behaves as clockwise |
| BUG-011 | MEDIUM | SOFTWARE BUG | radar.py:693 | UART debounce on the wall clock |
| BUG-012 | MEDIUM | SOFTWARE BUG | telegram_alert.py:132 | Alert cooldown on the wall clock |
| BUG-013 | MEDIUM | DESIGN RISK | main.py / hud.py | One projector width for both cameras and the display |
| BUG-014 | MEDIUM | SOFTWARE BUG | TWO_CAMERAS_FIXED.py | YOLO boxes not clipped to the frame |
| BUG-015 | LOW | SOFTWARE BUG | bearing_frame.py:169 | Dead ternary |
| BUG-016 | LOW | DESIGN RISK | bearing_frame.py | Inconsistent zero handling between sources |
| BUG-017 | LOW | SOFTWARE BUG | telegram_alert.py | Unused import and variable |
| BUG-018 | LOW | SOFTWARE BUG | TWO_CAMERAS_FIXED.py | Unused `fh/fw`, ambiguous `l`, 93 semicolons |
| BUG-019 | LOW | DESIGN RISK | radar.py | Private `_to_canonical` called across modules |
| BUG-020 | LOW | SOFTWARE BUG | TWO_CAMERAS_FIXED.py | Wall clock in the standalone loop |
| BUG-021 | LOW | DESIGN RISK | latency.py | Process-wide singleton accumulates across runs |
| BUG-022 | LOW | CALIBRATION PROBLEM | repo | Uncalibrated config in git; real calibration only on the Pi |
| BUG-023 | LOW | DESIGN RISK | fusion_config.py | Banner does not report handedness |
| BUG-024 | LOW | DESIGN RISK | TWO_CAMERAS_FIXED.py | Import-time `SystemExit` caught only by `BaseException` |
| BUG-025 | LOW | DESIGN RISK | radar.py | Confirmation-clearing invariant hard to verify by reading |

Also recorded, not numbered: `azimuths_to_degrees` radians/degrees heuristic
(§8); `cue_agreement_deg` has no force on the FAR lens (§11);
`lost_after_s` exactly equals the coasting window with no margin (§16);
`RespeakerLed._last_submit` is written and read across threads without a lock
(§18).

---

## 30. Final Verdict

```
NEEDS FIXES
```

**Why.**

The architecture is correct and the direction chain is now provably consistent
end to end. Nothing here is a crash, a data-loss path, or a threading hazard —
the concurrency, memory and error-handling audits came back clean, and 202
checks pass.

But two HIGH defects directly undermine the feature the system exists for, and
both are recent regressions with no test coverage:

* **BUG-001** removes the only warning that would tell an operator the LED ring
  may be pointing at the opposite horizon;
* **BUG-002** violates, once per camera switch, the explicit specification that
  a cue must never be drawn with the wrong camera's calibration.

Three further HIGH items (BUG-003, BUG-004, BUG-005) make the direction
subsystem vulnerable to a clock step on a machine that has no RTC and to a
source fallback that mixes coordinate frames.

Separately, and not a software defect: `camera_boresight_deg` is **invalid**
after the handedness recalibration and must be re-measured before the camera
cue means anything, and the working calibration is not in version control.

The latency requirement (0.5–1.0 s) is **not met** at a measured 1279 ms. The
remaining gap is structural rather than a bug, and the trade-offs are
documented and exposed in configuration — but the requirement as written is
not satisfied, and the largest unmeasured term (window fill) needs hardware.

Recommended order for the fix pass: BUG-002 → BUG-001 → BUG-003/004 →
BUG-006 → BUG-005, then the MEDIUM group, then re-measure the boresight and
commit the calibration.

---
---

# FIX VERIFICATION

Fix pass date: 2026-08-14. Every finding below was re-read, located in the
current code, fixed at the root cause where one existed, and covered by a
regression check in `test_integration.py` (`test_16_audit_fixes`) unless
noted.

Validation after the fix pass:

```
python -m compileall .        PASS
test_integration.py           241/241  (was 202/202)
ruff                          120 findings (was 127 at audit time)
                              F401 / F841 / E741 / F821 — ALL CLEAR
module self-tests             11/11 OK
deterministic injection       0/90/180/270 all correct end to end
simulate_pi                   camera holds 37.0 fps, UI adds +0.29 cores
measured latency              903 ms (was 1279 ms)
```

---

### BUG-001 — LED unverified-convention warning was dead code

```
Status:        FIXED
Files changed: respeaker_led.py
Root cause:    frame_for_target() assigned calibrated = True
               unconditionally, so LedFrame.calibrated could never be
               False and BOTH of its consumers were unreachable.
Fix:           the flag now comes from the observation
               (acoustic.bearing_calibrated). _render no longer treats it
               as a reason to withhold the sector — an unrecorded
               convention does not make the direction unusable, and the
               radar and camera region both point with the same number.
               The uncertainty is REPORTED by _warn_unverified instead.
Regression:    four checks — the flag is True when calibrated and False
               when not; the warning fires for an unverified frame and
               stays silent for a verified one; an unverified bearing
               still lights a SECTOR (2 distinct colours), proving the fix
               did not quietly disable working behaviour.
Verification:  all four PASS.
```

### BUG-002 — Switch frame paired the old image with the new camera id

```
Status:        FIXED
Files changed: camera_worker.py
Root cause:    `active` served as both "which camera produced this frame"
               and "which camera is selected now", and was re-read after
               the switching policy ran.
Fix:           frame_camera is read ONCE, immediately after a successful
               capture, and is the only id used for the tracks, the switch
               distance, the calibration print and the published
               observation. A switch decided during an iteration takes
               effect on the NEXT capture — which is when the frame will
               actually come from the other sensor.
Regression:    four checks — the source of _loop must contain the single
               capture-time read, must NOT contain a post-switch re-read,
               must publish active_camera=frame_camera; plus a simulation
               of four consecutive switches asserting every (frame, id)
               pair is self-consistent.
Verification:  all four PASS. INVARIANT NOW HOLDS —
               published_frame.camera_id == camera used for that frame ==
               calibration and FOV used for its cue.
```

### BUG-003 — Tracker release timeout on the wall clock

```
Status:        FIXED
Files changed: doa.py (DOATracker.update)
Root cause:    now = time.time() used for a DURATION against RELEASE_SEC.
Fix:           time.monotonic(), matching the project rule stated in
               target_state.py. Callers may still inject a clock.
Regression:    a backward step of one hour must not destroy the track, and
               the release must still fire on real elapsed time.
Verification:  both PASS.
```

### BUG-004 — DOA reading staleness on the wall clock

```
Status:        FIXED
Files changed: doa.py (HardwareDOA._loop, HardwareDOA.read)
Root cause:    stamp and age comparison both used time.time().
Fix:           monotonic for both.
Regression:    a source-level check asserts no time.time() remains
               anywhere in doa.py.
Verification:  PASS.
```

### BUG-005 — Tracker mixed sources with different conventions

```
Status:        FIXED (architectural)
Files changed: doa.py, bearing_frame.py
Root cause:    smoothing happened BEFORE the convention was applied, so one
               circular mean could contain USB and SRP angles expressed in
               two different frames, and the convention applied was
               whichever source answered last.
Fix:           DOAProvider._canonicalise() converts EACH reading with ITS
               OWN convention immediately; the tracker therefore smooths
               canonical angles only. _to_canonical() is now packaging, not
               transformation. DOAReading carries calibrated and convention
               so provenance survives smoothing.
Regression:    the same raw angle must map differently per source, and the
               smoothed result must sit between the two CANONICAL values.
Verification:  both PASS (usb 90 -> 264, srp 90 -> 300).
```

### BUG-006 — Camera 0 assumed when the camera is absent

```
Status:        FIXED
Files changed: sensor_fusion.py
Root cause:    active_camera = 0 if visual is None.
Fix:           _last_active_camera remembers the camera that most recently
               produced a frame; that is used when the worker has published
               nothing. It drives the range GATE as well as the cue, so
               this was a fusion decision, not only a display one.
Regression:    with the last frame from camera 1, a subsequent update with
               visual=None must keep the 65 deg HFOV, not fall back to
               camera 0's 28 deg.
Verification:  PASS.
```

### BUG-007 — Two target indications during a visual dropout

```
Status:        RECLASSIFIED -> DOCUMENTATION DEFECT, then FIXED
Reason:        Re-verification confirmed both are drawn, but this is the
               REQUIRED reacquisition behaviour (specification State C):
               the acoustic region must return when YOLO loses the target,
               and the last-known box is genuinely useful next to it. The
               defect was the CueRole docstring asserting the display
               "never" shows two positions, which was false.
Evidence:      hud._draw_boxes explicitly appends the coasting primary
               track; sensor_fusion returns CueRole.SEARCH in the same
               update. Styling separates them: amber dashed LAST KNOWN
               versus red bracketed ACOUSTIC ONLY.
Fix:           the docstring now states the guarantee that actually holds —
               never more than ONE of them is presented as a LIVE visual
               detection — and records the decision explicitly.
Regression:    both must be present after a dropout, by design.
Verification:  PASS.
```

### BUG-008 — overflow never reached the observation

```
Status:        FIXED
Files changed: radar.py, acoustic_worker.py
Root cause:    process_block() set st.overflow = False and had no way to
               learn otherwise; the audio thread detected overflows but
               never passed them in.
Fix:           process_block(block, overflowed=False); both the station
               worker and the standalone radar loop now pass the flag.
Regression:    the signature must accept it, and the flag must survive
               translation into AcousticObservation.
Verification:  both PASS.
```

### BUG-009 — Unknown source discarded a valid angle

```
Status:        FIXED
Files changed: doa.py
Root cause:    conventions.get(source) returning None produced a
               CanonicalBearing with deg=None, throwing away a good angle
               because its convention was unregistered.
Fix:           an unregistered source now gets a default UNCALIBRATED
               convention (registered once, with a printed warning) and the
               angle passes through marked as uncalibrated.
Regression:    a reading from source "mystery" must still yield a bearing,
               with calibrated=False.
Verification:  PASS.
```

### BUG-010 — UNKNOWN handedness silently behaved as clockwise

```
Status:        FIXED
Files changed: bearing_frame.py
Root cause:    UNKNOWN fell through an `if COUNTER_CLOCKWISE` into the
               clockwise branch, while the docstring claimed the opposite.
Fix:           assumed_handedness names the fallback in ONE testable place;
               zero_measured is tracked separately from handedness because
               an unmeasured zero is a ROTATION error while an unknown
               handedness is a MIRROR. SRP-PHAT now keeps its PROVEN
               counter-clockwise handedness even when its zero is
               unmeasured — previously it was demoted to UNKNOWN and then
               silently assumed clockwise, the one case where the fallback
               is provably wrong.
Regression:    UNKNOWN must name its assumption and report uncalibrated;
               SRP with no zero must stay CCW and uncalibrated.
Verification:  both PASS.
```

### BUG-011 — UART debounce on the wall clock

```
Status:        FIXED
Files changed: radar.py (UartSender.send)
Fix:           time.monotonic(). A backward step used to silence the
               bearing output on /dev/serial0 for the length of the jump.
Regression:    covered by the "no wall clock in the runtime path" sweep.
```

### BUG-012 — Telegram cooldown on the wall clock

```
Status:        FIXED
Files changed: telegram_alert.py
Fix:           time.monotonic() for the COOLDOWN_SEC duration.
```

### BUG-013 — One projector width for both cameras and the display

```
Status:        FIXED
Files changed: camera_cue.py, sensor_fusion.py
Root cause:    the cue was computed against visual.frame_width from the
               CONFIG while the HUD sized its canvas from the frame the
               camera actually delivered and scaled YOLO boxes in that real
               space. The two silently diverged if they ever differed.
Fix:           project(..., width_px=) and bearing_of_pixel(..., width_px=)
               take the real frame width and scale the focal length with
               it, so angle-per-pixel is preserved; the fusion layer passes
               visual.frame_width.
Regression:    a 1280-wide frame must project to exactly twice the x of a
               640-wide one, with the HFOV unchanged.
Verification:  both PASS (x=545 -> x=1089, 28.2 deg both).
SECOND-AUDIT   the first version of this fix updated project() but NOT
CATCH:         bearing_of_pixel(), so the forward and reverse projections
               would have used different pixel spaces. Caught in the second
               pass and fixed; a round-trip now returns the input bearing
               to 0.0000 deg at both widths.
```

### BUG-014 — YOLO boxes not clipped to the frame

```
Status:        FIXED
Files changed: TWO_CAMERAS_FIXED.py (_letterbox, _decode_head)
Root cause:    un-letterboxed corners could fall outside the sensor; the
               HUD clipped for drawing but sensor_fusion fed the box CENTRE
               to bearing_of_pixel(), which extrapolated past the real
               field of view and could wrongly cancel or sustain the
               acoustic search region.
Fix:           _letterbox records the source size; _decode_head clips all
               four coordinates to it and drops degenerate boxes. Both call
               sites already handled a (None, None) return — verified.
Regression:    HARDWARE VERIFICATION REQUIRED for the decode path itself
               (no Hailo available); the clipping arithmetic is
               straight-line and was reviewed twice.
```

### BUG-015 — Dead ternary in to_canonical

```
Status: FIXED   Files: bearing_frame.py
The no-op ternary is gone; the sign is applied once, via assumed_handedness.
```

### BUG-016 — Inconsistent zero handling between sources

```
Status: FIXED   Files: bearing_frame.py
zero_measured now expresses "the zero was not measured" separately from
"the handedness is unknown", so SRP keeps its proven handedness and its
unmeasured zero at the same time. This was the same defect as BUG-010's
SRP half.
```

### BUG-017 — Dead code in telegram_alert.py

```
Status: FIXED   Files: telegram_alert.py
Unused `text` removed. `import urllib.error` removed. `import urllib.parse`
HOISTED out of _api: as a function-local import it created a local `urllib`
binding that SHADOWED the module-level one, so the urllib.request.urlopen
on the next line resolved only because the package attribute happened to
already exist. Latent fragility, now removed.
```

### BUG-018 — Dead code / style in TWO_CAMERAS_FIXED.py

```
Status: PARTIALLY FIXED
Fixed:  unused fh, fw removed (verified during the audit that the decode
        genuinely does not need them); ambiguous `l` renamed to `left`.
NOT fixed (deliberate): the 93 semicolon-joined statements. They are style
        only, in a file the project explicitly preserves "essentially
        untouched", and rewriting 93 lines carries real risk for zero
        functional gain.
SECOND-AUDIT CATCH: the l -> left rename initially missed the use in the
        return statement, leaving an undefined name that would have raised
        NameError inside _letterbox and killed the whole camera pipeline.
        The test suite could NOT catch it (no Hailo hardware); ruff F821
        did. Fixed and re-verified — F821 now clear.
```

### BUG-019 — Private _to_canonical called across modules

```
Status: FIXED   Files: doa.py, radar.py
DOAProvider.hold() is now part of the public contract for "a block with no
measurement", alongside update(). radar.py no longer reaches into the
provider's internals or assigns its state from outside.
```

### BUG-020 — Wall clock in the standalone TWO_CAMERAS_FIXED loop

```
Status: NOT FIXED — RECLASSIFIED as out of scope for the station.
Reason: main() at TWO_CAMERAS_FIXED.py:1425 is reachable only via
        `python TWO_CAMERAS_FIXED.py`. camera_worker imports the module's
        classes and helpers, never its main loop — verified by reading
        every `import TWO_CAMERAS_FIXED as tc` call site.
Evidence: the station's frame timing uses time.monotonic() in
        camera_worker._loop and camera_manager throughout.
Impact:  affects only the standalone diagnostic script's FPS readout.
```

### BUG-021 — Latency budget accumulated across runs

```
Status: FIXED   Files: latency.py, main.py
LatencyBudget.reset() added; Station.start() clears the singleton so each
run reports its own numbers.
Regression: reset() must clear a recorded stage. PASS.
```

### BUG-022 — Uncalibrated configuration in git

```
Status: PARTIALLY FIXED — the software half is fixed; the process half
        cannot be fixed from here.
Fixed:  the station now STATES the angle convention of every DOA source at
        start-up (BUG-023), so a checkout running on shipped defaults
        announces "HANDEDNESS UNKNOWN — may be MIRRORED" instead of looking
        healthy.
NOT fixed: the repository's radar_calibration.json still holds the shipped
        defaults, and no fusion_config.json is tracked. The user's real
        calibration (doa_offset_deg 353.7, doa_handedness CCW, measured
        over five points with a 7.9 deg residual) exists only on the Pi.
        I will not invent or transcribe calibration values into the repo.
ACTION REQUIRED BY THE OPERATOR: commit the Pi's radar_calibration.json and
        fusion_config.json, or a `git pull` will keep reverting the
        direction calibration to an uncalibrated state.
```

### BUG-023 — Banner did not report handedness

```
Status: FIXED   Files: fusion_config.py, main.py
describe_bearing_sources() prints the convention of every DOA source at
start-up, before the calibration warnings.
Regression: must return one line per source and surface UNKNOWN. PASS.
```

### BUG-024 — Import-time SystemExit

```
Status: FIXED   Files: TWO_CAMERAS_FIXED.py
_check_camera_manager_is_current now raises RuntimeError. SystemExit
derives from BaseException and slipped past every `except Exception`,
degrading gracefully only because camera_worker happens to catch
BaseException. A stale camera_manager is an ordinary recoverable
configuration problem and now behaves like one at every call site.
```

### BUG-025 — Confirmation-clearing invariant duplicated

```
Status: FIXED   Files: radar.py
The gated branch and the weak-signal branch expressed one invariant in two
different ways. Both now call Detector._miss(), which is the single place
that decides what a missed block does: ALARM/COASTING keep their
confirmation count (so the signal returning re-arms the alarm on the same
block), TRACK clears it (so confirmations must be CONSECUTIVE).
Regression: the existing coasting and consecutive-confirmation tests all
still pass, which is the point — behaviour is identical, the invariant is
now readable.
```

---

# LATENCY

```
BEFORE:  1279 ms measured (mean, excluding window fill)
AFTER:    903 ms measured (mean, excluding window fill)
TARGET:   500-1000 ms      -> MET
```

| Term | Before | After | Why |
|---|---|---|---|
| audio read quantisation | 250 ms | 125 ms | hop 0.5 s -> 0.25 s |
| confirmation | 1000 ms | 750 ms | 0.75 s of evidence instead of 1.0 s |
| block processing | 3-16 ms | 3-13 ms | unchanged work, measured |
| publish -> fusion | 17-26 ms | 24 ms | unchanged |

**Classification quality is NOT reduced by the hop change.** The 2 s
analysis window is untouched; only how often it is re-evaluated changed, so
every individual classification sees exactly the same audio as before.
Measured processing is 13 ms mean / 31 ms max against the new 250 ms
budget — 5% and 12% — so the doubled inference rate has ample headroom, and
the worker warns if processing ever exceeds 90% of the hop.

**Classification quality IS slightly reduced by the confirmation change**,
and this is stated rather than hidden: 3 consecutive windows at a 0.25 s hop
is 0.75 s of new audio instead of 1.0 s. It is MORE independent looks and
LESS new audio. The false-alarm consequence cannot be measured without real
airtime — see REQUIRES REAL HARDWARE.

Architectural fix behind this: detector timing is now expressed in SECONDS
(confirm_seconds, coast_seconds, stable_seconds_for_acquisition) and
converted to block counts against the hop actually in use. A raw block count
silently changes meaning when the hop changes — confirm_blocks = 2 is 1.0 s
at a 0.5 s hop and 0.5 s at 0.25 s, from the same config line. Coasting
stayed exactly 3.0 s across the hop change because of this, verified by test.

**NOT MEASURED — window fill.** A drone audible for less than 2 s is diluted
inside the analysis window and scores lower. No threshold change removes
this; only a shorter window, which would need retraining. Quantifying it
needs a real drone and a reference microphone.

---

# REMAINING ISSUES

1. **BUG-022 process half** — the operator must commit the Pi's calibration.
2. **BUG-018 semicolons** — 93 style findings deliberately left in a legacy
   file.
3. **BUG-020** — reclassified, standalone script only.
4. `cue_agreement_deg = 20 deg` still has no force on the FAR camera: its
   HFOV is 28 deg, so no box inside the frame can disagree with the
   boresight by 20 deg. Documented, not a defect.
5. `coast_seconds` (3.0) exactly equals `lost_after_s` (3.0) — no margin.
   The worker warns only if coasting EXCEEDS it.
6. `azimuths_to_degrees` still distinguishes radians from degrees by range.
   A firmware reporting degrees with a target between 0 and 6 deg would be
   misread. Low probability, non-zero, unchanged by this pass.
7. `camera_boresight_deg` on the Pi is still INVALID — measured under the
   pre-CCW (mirrored) frame. Software cannot repair a mirror; it must be
   re-measured.

---

# VERIFIED IN SOFTWARE

* Camera frame/id/calibration consistency across switches in both directions.
* Monotonic timing under simulated forward and backward clock steps.
* Per-source canonicalisation; one raw angle mapping differently per source.
* Canonical bearing agreeing across radar, LED and camera cue at 0/90/180/270,
  with a negative control proving the check can fail.
* Angle wrap at 359/0 and 179/-179.
* Cue projection round-trip at two frame widths, exact to 0.0000 deg.
* LED calibration warning reachability, and that the sector still lights.
* Acoustic search region: occluded -> confirmed -> lost -> returns.
* Coasting: 3.0 s exactly, last bearing held, alarm re-armed on return.
* Overflow propagation, budget reset, banner convention reporting.
* Bounded memory, thread safety, graceful degradation of every subsystem.

# REQUIRES REAL HARDWARE

* XVF3800 channel order (`mic_channels`) — still an assumption.
* Physical array geometry (`mic_positions_m`) — still a default.
* Physical LED ring orientation (`led_zero_offset_deg`,
  `led_index_clockwise`) and the colour packing order.
* `LED_EFFECT` value that actually disables the firmware animation.
* Camera boresight — must be RE-MEASURED after the handedness change.
* Camera 1 focal length (502 vs 1003 unresolved).
* Hailo decode: the BUG-014 clipping, the sigmoid and LTRB auto-detect locks.
* Real Hailo inference latency and camera capture on the Pi.
* Physical camera switching.
* Acoustic classification against a real drone, including the window-fill
  latency term and the false-alarm rate at 0.75 s confirmation.
* Real-world distance accuracy.

---

# FINAL STATUS

```
READY FOR HARDWARE TESTING
```

Every one of BUG-001 through BUG-025 has been dealt with: 21 FIXED, 2
PARTIALLY FIXED with the unfixable half named (BUG-018 style, BUG-022
process), 1 RECLASSIFIED with evidence (BUG-020), 1 RECLASSIFIED then fixed
as a documentation defect (BUG-007). The second pass caught two regressions
introduced by the fixes themselves — an asymmetric projection and an
undefined name that would have killed the camera pipeline — and both are
fixed and covered.

Not READY, because the directional chain is only as correct as three values
that no amount of software can establish: the LED ring's physical
orientation, the microphone channel order, and a camera boresight that is
currently INVALID after the handedness recalibration. The software is now
consistent, instrumented, and honest about each of them.

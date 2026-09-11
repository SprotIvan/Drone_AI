# FORENSIC AUDIT — findings only

Hardware as stated by the operator: 2x Arducam B0240 IMX477P, lens 6 mm (WIDE)
and Kowa 25 mm (TELE); ReSpeaker XVF3800; Raspberry Pi 5 + Hailo-8L.

Verification levels used below: **CODE VERIFIED** (read from source),
**MATH VERIFIED** (arithmetic, no hardware), **LOG VERIFIED** (operator's own
run output), **UNVERIFIED** (needs hardware).

---

## [CRITICAL] C1 — Cameras are told apart only by enumeration index

**File:** `camera_manager.py:140` (`_open_camera`), `camera_manager.py:47-48`
**Evidence:** `picam = _load_picamera2()(camera_num=camera_id)`. `FAR_CAMERA_ID = 0`,
`NEAR_CAMERA_ID = 1`. Grep for `global_camera_info`, `.Id`, serial: no matches.
Both sensors are `imx477` (LOG VERIFIED, operator boot log shows the imx477
tuning file loaded twice).
**Why:** Nothing binds "index 0" to the 25 mm lens. If enumeration order changes
(kernel/libcamera update, one camera failing to probe, cable reseat), WIDE and
TELE swap silently. Both report the same sensor model, so no check can notice.
With one camera unplugged the survivor becomes index 0 and inherits the TELE
focal length.
**Fix:** Read `Picamera2.global_camera_info()` at startup, match each role to a
configured `Id` substring (the device-tree path, e.g. `i2c@88000`), refuse to
start on mismatch. Add `camera_id_hint` per role to `fusion_config.json`.

---

## [CRITICAL] C2 — The two focal lengths cannot both be right

**File:** `TWO_CAMERAS_FIXED.py:111,138` (`CAMERA_FOCAL_PX`),
`fusion_config.py:402` (`camera_focal_px`)
**Evidence (MATH VERIFIED):** identical sensors in an identical mode
(`1332x990`, LOG VERIFIED for both) ⇒ pixel focal ratio must equal lens ratio.
```
configured 1274.0 / 501.7 = 2.54
lenses         25 /   6   = 4.17      -> 39% mismatch
```
Implied lens from each configured value: 8.22 mm and 3.24 mm. Neither is 6 or 25.
**Why:** `estimate_distance_m()` scales linearly with focal, so every visual
distance is wrong; the camera-switch thresholds (1.5/2.0 m) and the fusion range
gate are all driven by it.
**Observed consequence (LOG VERIFIED):** `camera switched: FAR -> NEAR (0.50 m)`
requires a 637 px box at focal 1274 — i.e. the drone filled the 640 px frame.
With the optically derived 3875 px the same box is 1.52 m. A ~3x underestimate.
**Fix:** Re-run the `c` calibration on BOTH cameras at a large box
(`MIN_CALIBRATION_BOX_PX = 70`), or compute from optics:
`f_px = f_mm / 4.129 mm * 640` → 930 px (6 mm), 3875 px (25 mm).
**UNVERIFIED:** which of the two current numbers is wrong — requires a real
measurement.

---

## [CRITICAL] C3 — Camera 1 is labelled IMX708; it is an IMX477

**File:** `camera_worker.py:_camera_name()`, `fusion_config.py:825`
**Evidence (LOG VERIFIED):** operator log — `imx477.json` tuning file for BOTH
cameras; both device paths end `imx477@1a`. Code prints `NEAR/IMX708`.
Worse, `TWO_CAMERAS_FIXED.py:134-137` derives the 501.7 value from *"the IMX708
Camera Module 3 datasheet figure of 66 deg"* — a datasheet for a sensor that is
not installed.
**Why:** The operator is shown a false sensor model, and a focal length derived
from the wrong sensor's optics.
**Fix:** Rename to WIDE/TELE by role, delete the IMX708 reasoning, re-derive
from IMX477 geometry.

---

## [CRITICAL] C4 — Tracker and ego-motion state survive a camera switch

**File:** `camera_worker.py:325` (tracker created once), `:519`, `:535-540`;
`TWO_CAMERAS_FIXED.py:996-1004` (`process_frame`)
**Evidence (CODE VERIFIED):** `self._tracker` is constructed once in `_setup()`
and never reset. `process_frame` resets `ego_estimator` only when
`if self.tracks:` is false. `CameraSwitchPolicy.evaluate()` changes the active
camera with no notification to the tracker.
**Why:** After a switch, IMM tracks hold pixel coordinates from the *other*
lens, and `EgoMotionEstimator.prev_gray` is the previous camera's frame. Optical
flow is computed between two different cameras and the resulting affine warp is
applied to every track via `warp_all_models()`. With the stated lenses the pixel
scale changes by 4.17x in one frame.
**Fix:** In `camera_worker._loop`, when `frame_camera` differs from the previous
iteration's, call `self._tracker.tracks.clear()` and
`self._tracker.ego_estimator.reset()` before `process_frame`.

---

## [HIGH] H1 — The expected "camera points at the target" logic does not exist

**File:** whole project
**Evidence (CODE VERIFIED):** no servo / gimbal / pan / tilt / PWM / motor code
anywhere (grep returns only `mixer.py` audio-EQ "tilt"). `BearingProjector.project()`
is consumed at exactly one runtime site, `sensor_fusion.py:499`, and its output
is only drawn by `hud.py`.
**Why:** The cameras are FIXED. Acoustic bearing cannot steer anything; it draws
a red "look here" band. Camera selection is by DISTANCE only
(`CameraSwitchPolicy.evaluate(distance_m, max_box_height_px)`), never by DOA —
`allow_acoustic_fallback = False` deliberately.
**Fix:** None required in code; the architecture expectation must be corrected,
or a pan/tilt subsystem added (not present today).

---

## [HIGH] H2 — Derived visual range is wrong, and it gates sensor fusion

**File:** `fusion_config.py:781-799` (`derive_visual_range_m`), `:801-814`
**Evidence (MATH VERIFIED):** current 1274 px → 20 m, 501.7 px → 8 m (LOG
VERIFIED in the banner). Optically correct values give 61 m (25 mm) and 15 m
(6 mm).
**Why:** `activation_distance_m()` / `release_distance_m()` drive the
VISUAL_ACQUISITION gate in `sensor_fusion`. The station believes its tele camera
is useful to 20 m when the optics say ~61 m.
**Fix:** follows automatically once C2 is resolved.

---

## [HIGH] H3 — `doa_invert` is written by calibration and never read at runtime

**File:** `calibrate.py:489` writes it; `calibration.py:37` defaults it.
**Evidence (CODE VERIFIED):** runtime readers = none. Only `respeaker_led.py:1168`
reads it, inside `_bearing_test()` — a CLI diagnostic, not the station path.
`bearing_frame.source_convention()` reads `doa_handedness` ONLY.
**Why:** An operator who edits `doa_invert` sees no effect and no warning. A
calibration file can contain `doa_invert: true` and `doa_handedness: "CW"`
simultaneously, which is self-contradictory and silently resolved as CW.
**Fix:** Either delete the key on write, or refuse to start when `doa_invert`
contradicts `doa_handedness`.

---

## [HIGH] H4 — Two of the four DSP beams are frozen and contaminate the bearing

**File:** `doa.py:164-265` (`select_azimuth`)
**Evidence (LOG VERIFIED, `diagnose.py beams`, n=20 over ~7 s):**
```
beam 0:  1 distinct value  [104.4]          spread   0.0 deg
beam 1:  2 distinct values [190.2, 190.5]   spread   0.3 deg
beam 2: 20 distinct values                  spread 273.7 deg
beam 3: 20 distinct values                  spread 208.6 deg
```
Re-running the real selection code: a frozen beam falls inside the 25° cluster
and changes the answer in **11 of 20** samples by 0.6–8.3°, and raises the
reported "agreement" from 50% to 75% on no new evidence.
**Why:** `confidence` feeds `camera_cue.project()` band width and the weights of
`DOATracker._accept()`. Fabricated confidence, small angular error.
**Fix:** Exclude beams whose value is unchanged across the last N queries;
guard against excluding all of them.
**UNVERIFIED:** whether beams 0/1 are always frozen — 20 samples is thin, and it
is not established that a steady source was present during that run.

---

## [HIGH] H5 — SRP-PHAT fallback runs on beamformed stereo with invented geometry

**File:** `doa.py:962-1019` (`DOAProvider`), `audio_io.py:102-140`
**Evidence (CODE + LOG VERIFIED):** device reports 2 channels ("оброблене
стерео"). `mic_channels` resolves to `(0, 1)`; `mic_positions_m` stays at the
DEFAULT 43 mm square; `array_ok = (n_channels >= 2)` = True. Measured channel
correlation 0.8632 < the 0.999 degeneracy threshold, so `usable_pairs()` accepts
the pair.
**Why:** If the USB DOA errors or times out, the station silently switches to an
angle computed from a two-microphone 43 mm geometry applied to two *beamformed*
channels that are not microphones at known positions. The result is plausible
and unfounded; it is flagged only as `ambiguous`.
**Fix:** Require `raw_array` (>= 4 channels) before arming SRP-PHAT, or require
`mic_channels` to be explicitly configured.

---

## [HIGH] H6 — Range calibration was not produced by the calibration tool

**File:** `radar_calibration.json`
**Evidence (CODE VERIFIED, and the station warns at startup):**
`range_ref_distance_m = 3.0`, `range_ref_level_dbfs = -22.0`, both perfectly
round, and `range_spreading_db` absent — a key `calibrate.py range` always
writes alongside the other two.
**Why:** Every acoustic distance in metres is arbitrary.
**Fix:** `python calibrate.py noise` then `python calibrate.py range`, two
points at well-separated distances.

---

## [MEDIUM] M1 — Acoustic observations are timestamped at publish, not at capture

**File:** `acoustic_worker.py:_to_observation()` (`timestamp=now()`),
`sensor_fusion.py:407-412`
**Evidence (CODE VERIFIED):** the observation describes a 2.0 s sliding window
and needed 0.75 s of confirmation, but is stamped with the moment it was
published. `classify_age()` then rates it FRESH for 1.5 s.
**Why:** An ALARM stamped "now" describes audio centred ~1 s in the past.
Fusion pairs it with a camera frame that is 20 ms old. Measured total is 946 ms
(LOG VERIFIED).
**Fix:** Carry a separate `audio_capture_ts` and expose the known offset, or
document that acoustic freshness is relative to the decision, not the sound.

---

## [MEDIUM] M2 — Dead configuration keys

**File:** `fusion_config.py:414` (`cue_is_azimuth_only`), `:702` (`bearing_quantum_deg`)
**Evidence (CODE VERIFIED):** exactly one reference each — their own declaration.
**Why:** Settings that appear tunable and do nothing.
**Fix:** Remove, or wire them up.

---

## [MEDIUM] M3 — Optics constants duplicated across two files, synced one way only

**File:** `TWO_CAMERAS_FIXED.py:92-139` vs `fusion_config.py:396-408`;
sync at `camera_worker._sync_optics()`
**Evidence (CODE VERIFIED):** `CAMERA_FOCAL_PX` / `DRONE_REAL_WIDTH_M` exist in
both. `_sync_optics` pushes config → module at startup only.
Also duplicated: `SWITCH_TO_NEAR_BELOW_M`, `THRESHOLD_NEAR_HEIGHT`.
**Why:** `estimate_distance_m()` reads the module global. Anything that mutates
it after startup, or any direct `import TWO_CAMERAS_FIXED` user, gets the stale
copy.
**Fix:** Make `fusion_config` the only source; have `estimate_distance_m` take
focal as an argument.

---

## [MEDIUM] M4 — Both camera boresights are exactly 0.0, and the staleness guard is off

**File:** `fusion_config.json`
**Evidence (CODE VERIFIED):** `camera_boresight_deg = {0: 0.0, 1: 0.0}`;
`boresight_calibrated_at_doa_offset_deg` and `boresight_calibrated_handedness`
are null, so `check_bearing_frames()` (`fusion_config.py:856-913`) returns
early and never warns.
**Why:** Two physically separate cameras having an identical boresight of
exactly 0.0 is the signature of a placeholder, not a measurement. The cue is
drawn from it regardless.
**Fix:** Measure both, then record the offset/handedness they were measured
under to arm the guard.

---

## [MEDIUM] M5 — LED ring value count disagrees with configuration

**File:** `respeaker_led.py`, `fusion_config.json`
**Evidence (LOG VERIFIED):**
```
LED_RING_COLOR takes 75 value(s) — a single colour for the whole ring
...but led.led_count is 12, which does not match. Set led_count to 75.
LED ring command wants 12 values (learned from the device's own error) — adapting
```
**Why:** The first probe misreads the arity, warns, then self-corrects from an
error string. Works, but the operator is told to set a value (75) that would be
wrong.
**Fix:** Trust the error-derived arity over the probe, or suppress the
misleading advice.

---

## [LOW] L1 — Tracker rate is not reported separately

**File:** `camera_worker.py`
**Evidence:** `process_frame()` is called exactly once per loop iteration.
**Why:** Not a defect — tracker FPS is identical to loop FPS by construction.
Adding a second counter would duplicate a number under a new name. Documented
here so the omission is not mistaken for a gap.

---

# CHECKED AND NOT A PROBLEM

- **Streaming architecture** — one capture thread, one `LatestValue`, one JPEG
  encode per frame shared by all clients, no per-client capture.
  `web_server.FrameBus.encoded()` has an explicit thundering-herd guard. **PASS.**
- **FastAPI blocking** — every blocking endpoint is `def`, so Starlette runs it
  in the anyio threadpool; `_mjpeg` is a sync generator (`iterate_in_threadpool`).
  The event loop is never blocked. **PASS.**
- **`except: pass`** — zero occurrences project-wide. The broad handlers that
  exist are in `release()`/`_close_camera()` teardown paths and are appropriate.
- **Memory growth** — `Stage` 512 samples, `TrajectoryHistory` 512,
  `_trail` 256, `_client_meters` removed in `finally`, Hailo/picamera2 released.
  No unbounded structure found.
- **Wall-clock misuse** — `time.monotonic()` everywhere except one deliberate
  `time.time()` in the MJPEG part header, which is compared against the
  browser's `Date.now()` and is correct there.
- **Frame staleness** — measured `frame age 12–20 ms` against a 33.3 ms period
  across four runs. No queue backlog, no stale frames. **PASS.**
- **Camera FPS** — sensor pinned at 30.0 (`FrameDurationLimits(33333, 33333)`),
  loop 30.0–31.4 in every condition. The historical "70 FPS" is not reproducible
  and came from an FPS counter whose first interval was microseconds; fixed.
- **Hailo utilisation** — `hailortcli benchmark` on the operator's device:
  37.79 FPS, 24.66 ms HW latency, `hw_only == streaming` (no PCIe bottleneck).
  Measured full inference call is 29 ms, so ~4.3 ms is CPU. **The NPU is the
  bottleneck and the model's ceiling is 37.8 FPS, not 60.**
- **DOA handedness** — measured by the operator: CW residual 0.8°, CCW residual
  89.2°. Mirroring is ruled out. **PASS.**

---

# UNVERIFIED — HARDWARE REQUIRED

1. Which of the two configured focal lengths is wrong (C2).
2. Whether libcamera enumeration order is stable across reboots on this Pi (C1).
3. Whether the XVF3800 can be configured to expose 4 raw microphone channels
   over USB (it currently reports 2).
4. Whether beams 0/1 are permanently frozen or were frozen only during that run (H4).
5. Whether `camera_boresight_deg = 0.0` is true for either camera (M4).
6. Physical microphone calibration: `mic_positions_m` and `mic_channels` are both
   DEFAULTS, never measured against the real board.

---

# MICROPHONE / DOA AUDIT

```
Audio Input            PASS        (16 kHz, float32, device matched by name)
Microphone Mapping     UNVERIFIED  (mic_channels=(0,1) is derived, not measured)
DOA Math               PASS        (bearing_frame transforms verified, single
                                    application, no double offset)
Calibration Logic      PASS        (two-point fit, CW/CCW residuals compared)
Physical Calibration   PARTIAL     (handedness + zero measured: 0.8° residual.
                                    Range calibration NOT measured — H6.
                                    Array geometry NOT measured.)
```

# CAMERA AUDIT

```
WIDE  (IMX477P + 6 mm)        FAIL  — focal 501.7 px implies a 3.24 mm lens;
                                      labelled IMX708; identity not pinned
TELE  (IMX477P + Kowa 25 mm)  FAIL  — focal 1274 px implies an 8.22 mm lens;
                                      identity not pinned
TWO CAMERA LOGIC              PARTIAL — switching works and is debounced +
                                      temporally confirmed, but tracker state
                                      is not reset across the switch (C4) and
                                      no pointing/cueing actuation exists (H1)
SENSOR FUSION                 PARTIAL — freshness/staleness handled correctly;
                                      acoustic timestamp semantics (M1) and the
                                      wrong range gate (H2) remain
```

# FINAL VERDICT

```
NOT READY — CRITICAL FIXES REQUIRED
```

Project Health Score: **5 / 10**

The audio pipeline, threading model, streaming architecture and the measurement
layer are sound and now properly instrumented. The camera optics/identity layer
is not: two cameras that cannot be told apart, focal lengths that contradict the
installed lenses, and tracker state that crosses between two different fields of
view. Every visual distance and the fusion range gate depend on those.

Fix order: **C1 → C2/C3 → C4 → H6 → H4.**

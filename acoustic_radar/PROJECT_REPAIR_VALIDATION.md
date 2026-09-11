# PROJECT REPAIR + FORENSIC VALIDATION

Station: Raspberry Pi 5 + Hailo-8L + 2× Arducam B0240 (IMX477P) + ReSpeaker XVF3800.

## Where this session ran, and what that limits

This repair was performed on a **Windows development machine**, not on the Pi:

```
$ uname -a
MINGW64_NT-10.0-19043 DESKTOP-R2H4PID ... x86_64 Msys
$ ls /proc/asound
NO /proc/asound (not Linux)
$ which arecord
(not found)
```

There is no libcamera, no Hailo device, no ReSpeaker. **Every `arecord`,
`hailortcli` and `libcamera-hello` command in the brief is therefore
`BLOCKED — HARDWARE REQUIRED`, and no such measurement is reported below as
if it had been taken.** All nine hardware procedures are written up in
[HARDWARE_TEST_REQUIRED.md](HARDWARE_TEST_REQUIRED.md).

The project's own numpy install is broken (`import numpy` exits with
`0xC0000005`, an access violation — it is a MinGW build under an MSVC
Python). Tests were run in an isolated virtualenv created in a scratch
directory; **the user's environment was not modified.**

## Test baseline

| | Result |
| --- | --- |
| Before any edit | **303 / 305** (2 pre-existing failures) |
| After the first repair pass | **363 / 363** |
| After fixing N1 and C1 | **382 / 382** |
| After the forensic review's defect fixes (D1/D2/D3/D5/D6 + C1 rework) | **421 / 421** |

The two pre-existing failures were investigated and were **test defects, not
product defects** — see F-13 and F-14. 116 new checks were added in total.

---

# FIXED

---

## C1 — Cameras were told apart only by enumeration index

**OLD STATUS:** CRITICAL, unfixed.

**ROOT CAUSE.** `camera_manager.py` carried `FAR_CAMERA_ID = 0` /
`NEAR_CAMERA_ID = 1` and opened `Picamera2(camera_num=camera_id)`. Grep for
`global_camera_info`, `.Id` or a serial returned **zero matches** in the whole
repository. An index is a position in libcamera's enumeration order, not an
identity; both cameras are the same sensor model, so no check could notice a
swap. Role decides focal length, boresight and switch direction, so a swap
silently transposes every distance, every cue and every lens choice.

**⚠️ THE AMBIGUITY, AND HOW IT WAS CLOSED.** The two available *statements*
about the mapping contradict each other:

| Source | Claims |
| --- | --- |
| The code | index 0 = FAR = the narrow 28° lens ⇒ **index 0 = TELE (25 mm)** |
| Brief §2 hardware facts | **index 0 = WIDE (6 mm)** |

The first repair pass concluded that nothing in software could settle this
and left it hardware-blocked. **That conclusion was wrong.** Nothing in the
*metadata* distinguishes the cameras — same model, same tuning file, same
mode — but the *pixels* do, decisively. The lenses differ by 25/6 = **4.17×
in magnification**, and both cameras look the same way from the same mast,
so the telephoto image is literally the centre of the wide image blown up.
That is measurable, and it is a measurement of the hardware as it actually
is, which beats either statement about it.

**FILES CHANGED:** `camera_identity.py` (new), `camera_manager.py`,
`camera_worker.py`, `fusion_config.py`, `fusion_config.json`,
`TWO_CAMERAS_FIXED.py`.

**FIX — two independent sources that check each other.**

1. **The optical measurement** (primary). `CameraManager` grabs one frame
   from each camera at start-up, cuts the middle 1/4.17 of each, stretches
   it to full size and correlates it against the other. The correct
   assignment scores far above the swapped one. This *determines* the roles
   with no configuration at all.
2. **The device-tree `Id`** (`camera_role_id_hint`), whose `i2c@NNNNN` node
   is the physical CSI socket — stable across reboots and reordering,
   changing only if a cable is physically moved. This makes the mapping
   *stable*.

Together they answer both halves: the optics say **which lens**, the Id says
**which socket**, and a disagreement is a **hard error** — precisely the
silent lens swap the whole mechanism exists to catch. Optics are keyed by
**role**; the device index is a runtime binding from
`GeometryConfig.bind_roles()`.

Resolution remains **fail-closed** per your earlier decision: if the scene
has no structure (blank sky) *and* no hint is configured, the station
refuses and prints the discovered cameras plus the exact JSON to paste. It
never guesses.

**TEST / EVIDENCE.** `test_19_camera_identity` (9 identity checks + 5
optical) and `test_24_camera_manager_role_resolution` (9 end-to-end checks
against a fake Picamera2):

```
[PASS] C1: a configured hint binds each role to one device — {'WIDE': 0, 'TELE': 1}
[PASS] C1: enumeration order changes -> the role follows the socket — {'WIDE': 1, 'TELE': 0}
[PASS] C1: refuses — no hint configured / only one role / no match / ambiguous /
           both roles one camera / only one camera / unexpected sensor model
[PASS] C1: the optics identify which camera holds the long lens
           — NCC(A=WIDE,B=TELE)=+0.889 vs NCC(B=WIDE,A=TELE)=+0.003
[PASS] C1: and the answer follows the images, not the argument order
[PASS] C1: the correct assignment correlates, the swapped one does not
[PASS] C1: a featureless scene is INCONCLUSIVE, never a guess
[PASS] C1: a self-similar scene is INCONCLUSIVE too
[PASS] C1: with no hint, the roles are resolved from the images — WIDE=0 TELE=1
[PASS] C1: swapping the lenses swaps the resolved roles — WIDE=1 TELE=0
[PASS] C1: a hint that matches the optics is CONFIRMED by them
[PASS] C1: a hint contradicting the optics is REFUSED
[PASS] C1: blank scene + no hint is still fail-closed
[PASS] C1: a configured hint still works when the scene is blank
[PASS] C1: and it is reported as UNCONFIRMED, not as measured
[PASS] C1/C2: focal is derived per role once the roles are known
[PASS] C1/C2: and it is still labelled THEORETICAL
```

Separation on the synthetic pair is **0.889 vs 0.003**.

### ⚠️ THE FIXED-CENTRE MATCHER WAS REPLACED (forensic review → rework)

The forensic review measured the first implementation and found it fragile:
it assumed the telephoto view was the **exact centre** of the wide view, and
since the telephoto field is only 9.4° across, 1° of pointing error shifted
the image ~68 px and collapsed the correlation from 0.898 to 0.022. Two
cameras on a mast are misaligned by more than that, so the check would have
returned INCONCLUSIVE in exactly the situation it was built for.

**It now SEARCHES for the alignment** — bounded normalised cross-correlation
over ±25% of the wide frame (`camera_identity._match_peaks`) — instead of
assuming it. The bound matters: an unbounded slide would happily match a
random distant patch of a repetitive scene, which is a wrong answer rather
than an honest refusal. A second guard rejects matches whose winning
alignment is not **unique** (the best peak must beat the best peak outside
its own neighbourhood), which is what catches self-similar scenes.

Re-measured after the rework:

```
pointing offset 1°, 4°, 8°            -> resolved correctly (NCC ~0.91)
offset in both axes                   -> resolved correctly
offset + exposure difference + noise  -> resolved correctly
featureless scene                     -> INCONCLUSIVE (refuses)
coarse self-similar scene             -> INCONCLUSIVE (alignment not unique)
swapped pair                          -> the swapped roles, correctly
```

**NEW STATUS:**

* the role→index binding mechanism, the refusal paths and the optics/Id
  cross-check: `FIXED — CODE VERIFIED`;
* the optical classifier, including robustness to realistic pointing
  offsets: `ALGORITHM VERIFIED (synthetic only)`;
* knowing which physical socket holds which lens on **this** station:
  **`BLOCKED — HARDWARE REQUIRED` (HW-1)**.

⚠️ **Synthetic validation is not hardware validation.** The tests model
pointing offset, exposure, noise and distortion on generated images. Real
optics, parallax, scenes and the sensor pipeline are not represented. The
automatic path is now expected to work, but that expectation is untested on
the hardware — which is what HW-1 is for. Pinning `camera_role_id_hint`
remains recommended, because it makes the mapping independent of whatever
the cameras happen to be looking at and turns the optical check into a
permanent cross-check.

---

## C2 — The two focal lengths could not both be right

**OLD STATUS:** CRITICAL, unfixed.

**ROOT CAUSE (MATH VERIFIED).** Identical sensors in an identical mode must
have a pixel-focal ratio equal to their lens ratio:

```
configured   1274.0 / 501.7 = 2.54
lenses            25 /   6  = 4.17     -> 39% mismatch
```

Solving each configured value back to a lens gives **8.22 mm** and
**3.24 mm** — neither is fitted. `501.7` was additionally justified from the
**IMX708 Camera Module 3** datasheet, a sensor that is not installed.

**FILES CHANGED:** `TWO_CAMERAS_FIXED.py:92-154`, `fusion_config.py`,
`camera_manager.py`, `camera_identity.py`.

**FIX** (per your decision: THEORETICAL + loud warning). Focal length is
computed from optics:

```
f_px = (f_mm / 1.55 µm) × (output_width / sensor_crop_width)
WIDE   6 mm → (6/0.00155)  × (640/2664) =  930 px  (HFOV 37.9°)
TELE  25 mm → (25/0.00155) × (640/2664) = 3875 px  (HFOV  9.4°)
```

It is **recomputed at start-up from the sensor's real `ScalerCrop`**, so a
libcamera version that picks a different mode cannot silently invalidate it.
Provenance is carried in `camera_focal_source` and printed every run:

```
OPTICS NOT CALIBRATED: camera TELE: focal 3875 px is THEORETICAL, derived from
the nominal 25 mm lens — NOT a measurement. Distances carry a few percent of
error. Calibrate: HARDWARE_TEST_REQUIRED.md HW-2
```

`estimate_distance_m()` now accepts `focal_px` explicitly, and the module
global is documented as a start-up cache, not configuration.

**TEST / EVIDENCE.** `test_19_camera_identity`:

```
[PASS] C2: theoretical focal from IMX477 optics (WIDE 6 mm) — 930.0 px
[PASS] C2: theoretical focal from IMX477 optics (TELE 25 mm) — 3874.8 px
[PASS] C2: focal ratio now equals the lens ratio 25/6 — 4.167 vs 4.167 (old 1274/501.7 = 2.54)
[PASS] C2: a full-FOV mode gives a different focal for the same lens — 610.8 px vs 930 px binned
[PASS] C2: the station says out loud that the focal is THEORETICAL
[PASS] C2: and never calls a theoretical value calibrated
```

**NEW STATUS:** `PARTIALLY FIXED` — the contradiction is removed and the
values are internally consistent and honestly labelled, but a **real
calibration is `BLOCKED — HARDWARE REQUIRED`** (HW-2). These are theoretical
values; they are not measured and are not called measured.

---

## C3 — Camera 1 was labelled IMX708; it is an IMX477P

**OLD STATUS:** CRITICAL, unfixed.

**ROOT CAUSE.** `camera_worker._camera_name()` returned `"NEAR/IMX708"`;
`fusion_config.py:825` and `TWO_CAMERAS_FIXED.py:1750` did the same. The
`501.7` focal length was *derived* from the IMX708's published 66° FOV.

**FILES CHANGED:** `camera_worker.py`, `fusion_config.py`,
`TWO_CAMERAS_FIXED.py`, `test_integration.py`.

**FIX.** Labels are now `<ROLE>/IMX477P`, derived from the resolved role
rather than from an index — so the label cannot contradict either the
hardware or the identity binding. The IMX708-based reasoning is deleted, and
the withdrawn justification for `CALIBRATION_DISTANCE_M` is stated
explicitly rather than quietly dropped.

**TEST / EVIDENCE.** Repository-wide scan using `tokenize` (so that comments
explaining the removal do not trip it):

```
[PASS] C3: no IMX708 in executable code (both cameras are IMX477P) — clean
```

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## C4 — Tracker and ego-motion state survived a camera switch

**OLD STATUS:** CRITICAL, unfixed.

**ROOT CAUSE (CODE VERIFIED).** `self._tracker` is built once in `_setup()`.
In `_loop()`, a change of `frame_camera` reset only `_last_sensor_ns`. So on
the first frame after every switch:

1. `EgoMotionEstimator.prev_gray` still held **the other camera's frame**, so
   Lucas-Kanade ran between a WIDE frame and a TELE frame and the affine warp
   fitted to that meaningless flow was applied to every track via
   `warp_all_models()`.
2. IMM tracks held pixel state from the other lens. With these lenses the
   angular scale changes by **4.17× in one frame**.

**FILES CHANGED:** `camera_worker.py` (new `_reset_camera_state()`, called
from `_loop`).

**FIX.** On a real camera change (not the first frame), clear
`tracker.tracks`, call `ego_estimator.reset()`, and invalidate
`_last_detect_t`. After a switch there genuinely is no prior observation of
the target through the new lens, so the honest state is no state.

**TEST / EVIDENCE.** `test_20_camera_switch_resets_state`:

```
[PASS] C4: tracks from the previous lens are dropped — []
[PASS] C4: the optical-flow reference frame is cleared — resets=1
[PASS] C4: the detector-rate interval is invalidated too
[PASS] C4: the reset is symmetric (TELE -> WIDE clears as well)
[PASS] C4: WIDE -> TELE -> WIDE never carries a track across a switch — [False, False, False]
[PASS] C4: the frame loop calls the reset when the camera changes
```

Both required cycles (`WIDE→TELE→WIDE` and `TELE→WIDE→TELE`) are covered.

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## H3 — `doa_invert` was written by calibration and never read at runtime

**ROOT CAUSE.** `calibrate.py:489` wrote it; the only runtime reader in the
entire project was `respeaker_led.py:1168`, inside a CLI diagnostic.
`bearing_frame.source_convention()` reads **only** `doa_handedness`. So a
config could contain `doa_invert: true` and `doa_handedness: "CW"`
simultaneously — self-contradictory, silently resolved as CW.

**FILES CHANGED:** `calibration.py`, `calibrate.py`, `respeaker_led.py`,
`radar_calibration.json`.

**FIX.** Retired the key. Removed from `DEFAULTS`; `calibration.save()` now
strips it from the file on every write via `RETIRED_KEYS`; `calibrate.py` no
longer writes it; the LED diagnostic reports the real `doa_handedness`, and
prints `UNCAL (assumed CW)` when it has not been measured rather than
printing `False` (which reads as "measured, not mirrored").

**TEST / EVIDENCE.**

```
[PASS] H3: doa_invert is not a calibration default any more
[PASS] H3: and it is stripped from the file on the next save
[PASS] H3: no runtime code reads or writes doa_invert — clean
[PASS] H3: doa_invert is gone from the shipped calibration file
```

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## H5 — SRP-PHAT ran on beamformed stereo with invented geometry

**ROOT CAUSE (CODE VERIFIED).** `DOAProvider.__init__` had
`self.array_ok = n_channels >= 2`. The XVF3800 reports 2 **processed**
channels; `mic_positions_m` stayed at the default 43 mm square. SRP-PHAT is
geometric — every term requires the channels to be microphones at known
positions. The degeneracy guard could not catch it: it only rejects
*identical* channels, and the measured correlation was 0.8632, below the
0.999 threshold.

**FILES CHANGED:** `doa.py`.

**FIX (hardened after forensic review defect D5).** The gate originally read
`n_channels >= 4 or mic_channels`, and **neither disjunct proved anything**:
a device can expose four *processed* channels, and `mic_channels: [0,1]` is
an operator typing a tuple into JSON — an assertion, not a measurement.

SRP-PHAT now requires **two explicit attestations**, each of which only a
hardware test can justify:

| Condition | Requires |
| --- | --- |
| Raw microphone channels | `mic_channels` non-empty **AND** `mic_channels_verified: true` (HW-5 tap test) |
| Measured array geometry | `mic_positions_m` ≠ default **AND** `mic_geometry_verified: true` (HW-6) |

Both flags default to `false`. Arming SRP-PHAT is therefore a deliberate act
that records "a human ran this test", rather than a side effect of filling
in a config field. Otherwise it is disabled and `describe()` prints *why*,
naming the missing attestation and the test that produces it.

The gate is correct *as shipped*: `radar_calibration.json` has none of these
keys, so SRP-PHAT is off.

**TEST / EVIDENCE.** `test_22_srp_gating`:

```
[PASS] H5: 2 processed channels + default geometry -> DISABLED
[PASS] D5: channel count alone (4 ch) does NOT enable SRP-PHAT
[PASS] D5: a mic_channels tuple alone does NOT enable SRP-PHAT
[PASS] D5: even 4 channels AND a mic_channels tuple, without the verified
           flag, does NOT enable it
[PASS] D5: non-default geometry without its verified flag -> DISABLED
[PASS] D5: the verified flag with DEFAULT geometry -> still DISABLED
[PASS] D5: VERIFIED raw channels + VERIFIED geometry -> ENABLED
[PASS] D5: both attestations default to False
[PASS] H5: the shipped gate is no longer `n_channels >= 2`
[PASS] D5: and it no longer accepts a bare channel count or list
[PASS] D5: it requires both explicit attestations
```

**NEW STATUS:** `FIXED — CODE VERIFIED`. Whether 4 raw channels exist at all
is `BLOCKED — HARDWARE REQUIRED` (HW-4/HW-5/HW-6).

---

## H6 / M4 — Placeholders masquerading as calibration

**ROOT CAUSE.** `radar_calibration.json` held `range_ref_distance_m: 3.0` and
`range_ref_level_dbfs: -22.0` — both perfectly round, with
`range_spreading_db` **absent**, a key `calibrate.py range` always writes
alongside the other two. `fusion_config.json` held
`camera_boresight_deg: {0: 0.0, 1: 0.0}`; two physically separate cameras
having an identical boresight of exactly zero is a placeholder, not a
measurement. And because `boresight_calibrated_at_doa_offset_deg` was null,
`check_bearing_frames()` returned early and never warned.

**FILES CHANGED:** `radar_calibration.json`, `fusion_config.json`,
`fusion_config.py`.

**FIX.** Both are now `null`. `is_range_calibrated()` returns False, so the
station prints «н/д» instead of an arbitrary metre value; the acoustic cue is
disabled until a boresight is measured. The index-keyed maps are rejected on
load with a message pointing at the role-keyed replacements, so a placeholder
cannot be bound to an enumeration index again.

**⚠️ THIS REMOVES FUNCTIONALITY UNTIL YOU CALIBRATE.** Acoustic distance in
metres and the on-screen bearing cue are unavailable until HW-8 and HW-3 are
done. Unlike focal length, there is **no theoretical substitute** for an
acoustic reference level — it depends on your specific drone and site — so
`null` is the only honest option.

**TEST / EVIDENCE.**

```
[PASS] M4: no camera boresight is a placeholder 0.0 — {'WIDE': None, 'TELE': None}
[PASS] M4: the retired index-keyed boresight is gone from the config
[PASS] H6: the un-measured acoustic range reference is null, not 3.0 m — None m / None dBFS
[PASS] H6: so the station reports acoustic range as uncalibrated
```

**NEW STATUS:** `FIXED — CODE VERIFIED` (the placeholder no longer lies). The
calibrations themselves: `BLOCKED — HARDWARE REQUIRED` (HW-3, HW-8).

---

## M1 — Acoustic observations were timestamped at publish, not at capture

**ROOT CAUSE.** `_to_observation()` set `timestamp=now()`. The record
describes a **2.0 s sliding window**, so the sound it represents is centred
about 1 s earlier — yet fusion paired it with a camera frame ~20 ms old and
`classify_age()` rated it FRESH.

**FILES CHANGED:** `target_state.py`, `acoustic_worker.py`.

**FIX.** Four instants are now distinguished: `capture_start` /
`capture_centre` / `capture_end` / `timestamp` (the decision). The worker
passes the monotonic instant the audio block was read, plus the classifier's
window length. `sound_age_s()` and `decision_lag_s()` expose the difference.
`timestamp` deliberately keeps driving the freshness timeouts — those answer
"is the subsystem still producing output?", which really is about decision
time. All clocks are `time.monotonic()`, so an NTP step on a Pi with no RTC
cannot corrupt an age.

**TEST / EVIDENCE.** `test_23_acoustic_timestamp_semantics`:

```
[PASS] M1: the sound is ~1.1 s old the moment the decision is published — 1.12 s
[PASS] M1: while the decision itself is 0 s old — they are not the same
[PASS] M1: the structural lag is exposed — 1.12 s window-centre -> publish
[PASS] M1: absent capture timing reports None, not a fabricated zero
```

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## M2 — Dead configuration keys

`cue_is_azimuth_only` and `bearing_quantum_deg` each had exactly one
reference: their own declaration. Both removed with the reason recorded in
place. `bearing_quantum_deg` was additionally **superseded** — the LED
controller already suppresses redundant writes by comparing the computed LED
sector, so the effective threshold is the ring's real angular pitch
(360/led_count) rather than a fixed 10° that would be wrong for any ring that
is not 36 LEDs.

```
[PASS] M2: cue_is_azimuth_only removed (it had no readers)
[PASS] M2: bearing_quantum_deg removed (superseded by ring geometry)
```

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## M3 — Optics constants duplicated across two files

`CAMERA_FOCAL_PX` existed in both `TWO_CAMERAS_FIXED` and `fusion_config`,
synced one way at start-up only. Now `fusion_config` is the single source of
truth, keyed by role; the module dict is an explicitly documented start-up
cache that is **replaced, not merged** (a leftover entry for an index that no
longer holds that role is exactly the stale copy this prevents); and
`estimate_distance_m()` accepts focal explicitly.

**NEW STATUS:** `FIXED — CODE VERIFIED`.

---

## H4 — The frozen-beam diagnostic was too short to conclude anything

**ROOT CAUSE.** The claim "beams 0 and 1 are frozen" rested on **20 samples
over ~7 s**. That cannot separate *frozen* (the DSP is not updating the beam)
from *steady* (the beam is working and the source genuinely did not move) —
and the distinction matters, because a constant always has spread 0.0 and so
always looks like the tightest cluster to `select_azimuth()`.

**FILES CHANGED:** `diagnose.py`.

**FIX.** Default samples raised 40 → 200. Per-beam reporting extended from
mean/range to: sample count, unique values, mean, min, max, spread, standard
deviation, variance, **number of changes**, and **longest run of identical
consecutive readings** — the last two being the actual test of whether a beam
is being updated. The tool now emits a per-beam verdict and **refuses to
print "FROZEN" below 100 samples**, instructing the operator to change the
acoustic scene mid-run.

**HARDENED after forensic review defect D6.** The first rework still printed
an unqualified `FROZEN` whenever a beam showed zero changes over ≥100
samples — but a healthy beam watching a source that never moved also shows
zero changes, and **no sample count can separate those**. Only a deliberate
scene change can.

The verdict is now a pure, testable function (`diagnose.beam_verdict`) and
the tool will not say FROZEN unless the operator passes `--scene-changed`,
asserting they moved or switched off the source mid-run. Without it a silent
beam is reported `NOT PROVEN — STATIC SCENE`, and the instruction to re-run
properly is printed on **every** run rather than only short ones.

```
[PASS] D6: a static scene with no change is NOT PROVEN, not FROZEN
[PASS] D6: no response ACROSS a source change is FROZEN
[PASS] D6: a beam that followed the source change RESPONDED
[PASS] D6: fewer than 100 samples is INSUFFICIENT, whatever happened
[PASS] D6: a beam that does update is never called frozen
```

**NEW STATUS:** `FIXED — CODE VERIFIED` for the tooling; the answer about
these particular beams is `BLOCKED — HARDWARE REQUIRED` (HW-7). **No beam
has been excluded**, because that would be acting on evidence that does not
exist yet.

---

# NEW FINDINGS (not in the prior audit)

## N1 — The bearing-agreement gate reported agreement it could not have withheld

**Surfaced by fixing C2. Now fixed.**

**⚠️ MY FIRST ANALYSIS OF N1 WAS WRONG, BY A FACTOR OF TWO.** It compared the
tolerance against the **half** field and concluded the gate was dead on both
cameras. That silently assumed the acoustic bearing always sits exactly on
the boresight. It does not: both the box angle θ and the bearing b range
over ±h, so the largest disagreement an in-view pair can produce is

```
max |b − θ| = 2h = the FULL field of view
```

reached when the bearing is at one edge of the frame and the box at the
other. The corrected criterion gives **different answers per camera**:

| Camera | Field of view | vs 20° tolerance | Gate |
| --- | --- | --- | --- |
| TELE (25 mm) | 9.4° | 9.4 < 20 | **can never reject** |
| WIDE (6 mm) | 37.9° | 37.9 > 20 | **works normally** |

So the gate is not dead — it is dead *on the telephoto camera only*.

**ROOT CAUSE.** The real defect was never the number 20. It was that a
**guaranteed** pass was reported identically to an **earned** one:
`CueRole.CONFIRMED` withdrew the acoustic search region and the snapshot
said "the sensors agree", with nothing recording that on TELE the check had
no power to disagree.

**FILES CHANGED:** `camera_cue.py`, `sensor_fusion.py`, `fusion_config.py`.

**FIX.** `BearingProjector` gained `max_agreement_deg()` (the full field, not
half) and `agreement_is_discriminating(camera_id, tolerance)`. `SensorFusion`
computes it for the camera that produced *this* frame and carries it on every
`FusedTarget` as `sensor_agreement_discriminating`.

**The detection decision is deliberately unchanged.** On a 9.4° field an
unrelated object really is within the DOA's uncertainty of the drone, so
treating the box as confirming is not wrong — it is simply not *evidence*.
Rejecting it instead would discard real detections. And the tolerance itself
is not a free parameter: it is the DOA's angular uncertainty, and tightening
it below the sensor's true accuracy would start rejecting correct boxes on
noise. Measuring that accuracy is HW-6. What the code now does is state
where it has no power instead of pretending it does.

**TEST / EVIDENCE.**

```
[PASS] N1: TELE's whole field is inside the tolerance, so the gate cannot
           reject anything it can see — fov 9.4deg vs tolerance 20deg
[PASS] N1: WIDE's field exceeds the tolerance, so the gate DOES work there
           — fov 38.0deg vs tolerance 20deg
[PASS] N1: the span is the FULL field of view, not half of it — 38.0deg
[PASS] N1: uncalibrated optics report None, not a fabricated span
[PASS] N1: a pass on TELE is reported as NON-discriminating — False
[PASS] N1: while on WIDE the same check IS discriminating — True
```

### ⚠️ THE FLAG WAS SPLIT IN TWO (forensic review defect D2 → fixed)

The mathematics was right, but the single flag conflated two different
questions. `agreement_is_discriminating` answered a *camera-geometry*
question — "can an in-view pair ever exceed the tolerance?" — while being
published per observation as though it described that particular check. When
the acoustic bearing falls **off-screen** the gate really does reject, and
the snapshot still reported `False`:

```
bearing 190.0 -> agreement 40.0 deg > 20 deg tolerance -> REJECTED
old flag                                               -> False   (wrong)
```

There are now **two fields**, each asserted against the question it actually
answers:

| Field | Question | TELE |
| --- | --- | --- |
| `sensor_agreement_capable` | Can this LENS ever reject an in-view pair? | `False` (9.4° < 20°) |
| `sensor_agreement_discriminating` | Could THIS check have gone the other way? | `True` when it rejected |

The per-observation rule is ordered: **an actual rejection proves the check
had power**, full stop; otherwise the pass is informative only if a
rejection was reachable for that bearing (`|bearing_rel| + half_fov >
tolerance`).

**The detection decision remains unchanged.** On a 9.4° field an unrelated
object really is within the DOA's uncertainty of the drone, so treating the
box as confirming is not wrong — it is simply not *evidence*. Rejecting it
instead would discard real detections.

**STATUS:** `FIXED — CODE VERIFIED` / `MATH VERIFIED`. Measuring the
XVF3800's real angular accuracy, so the tolerance can be derived rather than
assumed, remains hardware-dependent (HW-6).

---

## N2 — One drone → two targets: mechanism identified

**Investigated per brief §21, without assuming it was DOA.**

* **The radar tile cannot be the source.** It draws a single `FusedTarget`
  via one un-looped `_draw_target` call, and the ±mirror ghost at
  `(180 - bearing)` was removed in an earlier session (asserted by test 18).
* **The HUD draws one box per tracker track.** So two tracker tracks for one
  drone are two boxes on screen.
* **C4 is a mechanism that produced exactly that.** With `max_age = 10` a
  track survives ten consecutive misses. After a camera switch the old lens's
  tracks coasted at their old pixel coordinates while the detector — seeing
  the same drone at a 4.17× different scale — failed to associate and opened
  **new** tracks beside them. Two boxes, one drone, for roughly a third of a
  second after every switch.

Fixing C4 removes this mechanism.

```
[PASS] N2: a track survives many misses, so stale ones linger visibly — max_age=10 detector frames
[PASS] N2: the radar tile draws exactly one target (not a loop)
[PASS] N2: the HUD draws one box per track, so duplicate tracks are duplicate boxes
```

**STATUS:** `PARTIALLY FIXED`. One duplicate-generating mechanism is
confirmed and removed. Whether it is the one you observed cannot be
established from code — if you saw **two blips on the radar dial** rather
than two boxes on the camera image, that is a different cause and I have not
found it. `BLOCKED — HARDWARE TEST REQUIRED`: reproduce with a single drone
and record whether the duplicate appears on the camera image or the radar
dial, and whether it coincides with a camera switch.

---

# CHECKED AND NOT A PROBLEM

Re-verified during this session, not merely inherited from the prior audit:

* **H1 — camera pointing / gimbal.** A tokenize-based repository scan for
  `servo|gimbal|pan_tilt|GPIO|PWM|stepper|actuator` in executable code
  returns **clean**. The cameras are **fixed**. The architecture is
  `fixed cameras + camera selection`, **not** active camera pointing.
  `BearingProjector` draws a cue band and steers nothing; camera selection is
  by distance only, and `allow_acoustic_fallback = False` is deliberate.
  Asserted: `[PASS] H1: no servo/gimbal/PWM code — the cameras are fixed`.
* **`except: pass`** — zero occurrences project-wide. The broad handlers that
  exist are in `release()` / `_close_camera()` teardown paths.
* **Streaming architecture** — one capture thread, one `LatestValue`, one
  JPEG encode per frame shared by all clients, explicit thundering-herd
  guard. `[PASS] three clients share ONE encode per frame`.
* **FastAPI event loop** — every blocking endpoint is `def`, so Starlette
  runs it in the anyio threadpool; `_mjpeg` is a sync generator. The loop is
  never blocked.
* **Clocks** — `now()` is `time.monotonic()`. The single `time.time()` is in
  the MJPEG part header, compared against the browser's `Date.now()`, where
  wall clock is correct.
* **F-13 (`the Hailo rate is NOT counted in the rate-limited UI loop`).** A
  **test defect**: the check was a substring match over `Station.run`'s
  source, and it matched main.py's own *comment* explaining that the call had
  been removed — so the better the fix was documented, the more certainly the
  guard failed. Now strips comments before checking. The production code was
  already correct.
* **F-14 (`every client is counted` → 4).** A **test race**: the previous
  reader was closed client-side, but the server-side generator only runs its
  `finally` when it next writes into the dead socket. Accounting is correct
  (`pop` in a `finally`). The test now polls for the count to settle and
  still asserts exactly 3.

---

# CAMERA FINAL TABLE

| Camera | Physical HW | Stable ID | Role | Sensor | Lens | Focal px | Calibration status | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WIDE | Arducam B0240 | device-tree `Id` — **operator-supplied, HW-1** | WIDE | IMX477P | 6 mm | **930 (THEORETICAL)** | NOT CALIBRATED (HW-2) | blocked on HW-1 |
| TELE | Arducam B0240 | device-tree `Id` — **operator-supplied, HW-1** | TELE | IMX477P | Kowa 25 mm | **3875 (THEORETICAL)** | NOT CALIBRATED (HW-2) | blocked on HW-1 |

* **Stable ID is deliberately not a literal here** — the actual `Id` strings
  belong to your Pi and I have never seen them. Writing a specific `i2c@…`
  value into this table would be inventing a camera ID.
* **Corrected:** an earlier version of this row said the ID was "resolved at
  run time" and marked the status OK. The station *attempts* that, but the
  optical resolution is unreliable under realistic camera misalignment
  (see C1 above), so the mapping is in practice **operator-supplied and
  hardware-blocked on HW-1**.
* Focal values are **THEORETICAL**; the size of their error is **unknown**
  until HW-2.
* Boresight: `null` for both. **Not** 0.0.

---

# AUDIO FINAL TABLE

| Item | Result | Evidence | Status |
| --- | --- | --- | --- |
| Physical microphones | 4 | operator's hardware statement | ASSUMED — not independently verified here |
| USB channels | 2 reported | prior run log; `arecord` unavailable this session | `BLOCKED — HARDWARE REQUIRED` (HW-4) |
| Raw channels | **unknown** | cannot be derived from a channel count | `BLOCKED — HARDWARE REQUIRED` (HW-4) |
| Processed channels | likely 2 (XVF3800 DSP output) | `audio_io` labels it «оброблене стерео» | `POTENTIAL — believed case B, unconfirmed` |
| Channel mapping | **unknown** | `mic_channels` defaults to "first N" — an assumption | `BLOCKED — HARDWARE REQUIRED` (HW-5) |
| Mic geometry | **DEFAULT, unmeasured** | 43 mm square from the reference design | `BLOCKED — HARDWARE REQUIRED` (HW-6) |
| DSP beams | inconclusive | 20 samples is not enough to call a beam frozen | `BLOCKED — HARDWARE REQUIRED` (HW-7) |
| DOA handedness | CW | operator's measurement: CW residual 0.8°, CCW 89.2° | `HARDWARE VERIFIED` (prior session) |
| SRP-PHAT validity | **DISABLED** | gate requires raw channels + measured geometry | `FIXED — CODE VERIFIED` |
| Acoustic range calibration | **NOT CALIBRATED** | reference values nulled; `range_spreading_db` was absent | `BLOCKED — HARDWARE REQUIRED` (HW-8) |

---

# FILES CHANGED

| File | Change |
| --- | --- |
| `camera_identity.py` | **NEW** — role definitions, stable-identity resolution, **optical role classification**, theoretical optics |
| `camera_cue.py` | N1: `max_agreement_deg` / `agreement_is_discriminating`; demo binds roles |
| `sensor_fusion.py` | N1: carries `sensor_agreement_discriminating` per frame |
| `HARDWARE_TEST_REQUIRED.md` | **NEW** — 9 hardware procedures |
| `PROJECT_REPAIR_VALIDATION.md` | **NEW** — this file |
| `camera_manager.py` | fail-closed role resolution (**optical + Id, cross-checked**); runtime focal derivation; role-named switching |
| `camera_worker.py` | C4 reset; role-index switching; role/IMX477P labels; optics sync after binding |
| `fusion_config.py` | role-keyed optics; `bind_roles`; focal provenance; M2 removals; N1 documented |
| `TWO_CAMERAS_FIXED.py` | focal cache + explicit-focal `estimate_distance_m`; IMX708 reasoning deleted |
| `doa.py` | SRP-PHAT physical-validity gate |
| `target_state.py` | capture/decision timestamp semantics |
| `acoustic_worker.py` | passes real capture time and window length |
| `calibration.py` / `calibrate.py` / `respeaker_led.py` | `doa_invert` retired |
| `diagnose.py` | long-run beam statistics and verdict |
| `fusion_config.json` / `radar_calibration.json` | placeholders → null; role hints added |
| `test_integration.py` | role-binding fixture; 6 new test groups; 3 corrected tests; N1 guard replaced |

Pre-existing uncommitted work (Hailo stage timing, the corrected 129 ms
`xvf_host` figure) was left intact. No commit was made.

---

# TESTS

```
$ python -m compileall -q .                       exit 0
$ python -c "import <each of 17 modules>"          all exit 0
$ python test_integration.py                       421/421 passed, exit 0
$ python diagnose.py beams --help                  --scene-changed present
$ python camera_worker.py                          exit 0; oscillation 72 -> 0 switches;
                                                   approach still switches once
$ python camera_cue.py                             exit 0; TELE HFOV 9.4°, WIDE 37.9°,
                                                   pixel<->bearing round-trip exact
$ python fusion_config.py                          exit 0
$ python bearing_frame.py                          exit 0
$ python calibration.py                            exit 0
$ git status                                       only repair-related files modified
```

Test groups added across all passes: `test_19_camera_identity`,
`test_20_camera_switch_resets_state`, `test_21_hardware_claims`,
`test_22_srp_gating`, `test_23_acoustic_timestamp_semantics`,
`test_24_camera_manager_role_resolution`,
`test_25_forensic_review_fixes`.

**Not run — no hardware:** real Picamera2 capture, camera switching against
real optics, Hailo inference, ReSpeaker capture, `arecord`, `hailortcli`.

---

# REMAINING RISKS

All of these are calibration/measurement items or accepted design
trade-offs. None is an unresolved software defect.

1. **The optical role check needs a textured scene.** Pointing offset no
   longer defeats it, but blank sky does — and with no `camera_role_id_hint`
   configured the station then refuses to start its cameras rather than
   guess. Setting the hint once (HW-1(b)) removes the dependency and adds a
   permanent cross-check.
2. **The optical check is validated only synthetically.** It is expected to
   work on the hardware; that expectation is untested (HW-1).
3. **Acoustic distance in metres and the bearing cue are unavailable** until
   HW-8 and HW-3. Deliberate: the previous values were placeholders.
4. **The error in visual distances is UNKNOWN** — they are theoretical, not
   calibrated (HW-2). No error bound is claimed anywhere, in the documents
   or at runtime.
5. **The camera-switch thresholds (1.5 / 2.0 m) have not been re-derived**
   against the corrected focal lengths. They are compared against a distance
   computed from focal, so the range at which the swap physically occurs has
   moved even though the numbers did not. Re-check after HW-2.
6. **On TELE the bearing-agreement check cannot reject an in-view pair** —
   its 9.4° field is inside the DOA's uncertainty. The station reports this
   per frame (`sensor_agreement_capable` / `sensor_agreement_discriminating`).
   Deriving the tolerance from a measured DOA accuracy needs HW-6.
7. **SRP-PHAT is disabled**, and now requires two explicit hardware
   attestations to re-enable. A USB DOA failure yields «н/д» rather than a
   fabricated angle. That is the intent, but it is less redundancy.
8. **`max_age = 10` was not re-tuned.** Outside a camera switch a genuinely
   lost track still coasts ten detector frames.

---

# FINAL VERDICT

```
READY FOR HARDWARE VALIDATION
```

Every defect the forensic review identified is now fixed **and covered by a
regression test**:

| | Defect | Fix | Tests |
| --- | --- | --- | --- |
| **D1** | Fabricated "few percent" error bound printed at runtime | Replaced with "THE ERROR … IS UNKNOWN until calibration"; withdrawn from comments and docstrings too | 4 |
| **D2** | One flag conflated camera capability with what this check did | Split into `sensor_agreement_capable` and `sensor_agreement_discriminating`; an actual rejection now always counts as discriminating | 11 |
| **D3** | `suggest_hint_config` could emit a config the next startup rejects | Fragments are checked for uniqueness against every camera, falling back to longer path pieces and finally the full Id | 3 round-trips |
| **D5** | Channel count / operator-typed tuple treated as proof of raw mics | Requires `mic_channels_verified` **and** `mic_geometry_verified`, both defaulting to false | 11 |
| **D6** | Unqualified `FROZEN` for a healthy beam on a static scene | Verdict extracted to `beam_verdict()`; FROZEN requires `--scene-changed` | 7 |
| **C1** | Optical matcher collapsed at 1° of pointing offset | Bounded shift search + peak-uniqueness rejection | 10 |

**Suite: 421/421, up from 382/382.** 39 checks added.

**What this verdict means.** No software/design/config issue known to this
repair remains open. Everything still outstanding requires a physical
measurement that no amount of code can substitute for: focal calibration
(HW-2), boresight (HW-3), the XVF3800's channel semantics, microphone
mapping and array geometry (HW-4/5/6), whether beams 0/1 are truly frozen
(HW-7), the acoustic range reference (HW-8), and Hailo throughput (HW-9).
HW-1 confirms a role mapping the station now derives for itself.

**What it explicitly does NOT mean.** The optical role check is validated
**synthetically only**. It models pointing offset, exposure, noise and
distortion on generated images; real optics, parallax and scenes are not
represented. It is expected to work on the hardware — that is not the same
as knowing it does, which is why HW-1 exists and why pinning the hint is
still recommended.

**`READY — HARDWARE VERIFIED` remains unavailable** and is not claimed: this
session had no Pi, camera, ReSpeaker or Hailo. Nothing in this report is
hardware-verified except the DOA handedness, measured in a previous session.

Suggested order: **HW-1 → HW-2 → HW-3**, then **HW-4 → HW-5 → HW-6 → HW-7**,
then **HW-8**, with **HW-9** any time.

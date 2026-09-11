# FINAL FORENSIC REVIEW

A verification pass over my own repair claims, against the code as it now
stands. No production code was changed in this pass. Where a change is
required, it is listed in **PENDING CODE CHANGES** and is awaiting your
approval.

**Headline: the previous verdict `READY FOR HARDWARE VALIDATION` was NOT
justified.** Two claims were stronger than the evidence, one documented
hardware procedure does not match the CLI, and three real software defects
remain. Corrected verdict at the bottom.

---

## Verification table

| ID | Claim as written | Evidence in actual code | Test proving it | Status | Remaining hardware dependency |
| --- | --- | --- | --- | --- | --- |
| C1-a | Software can tell WIDE from TELE by image content | `camera_identity.classify_roles_by_optics` / `optical_assignment_score`; fixed centre-crop + NCC | `test_19` (5 synthetic checks), `test_24` (fake Picamera2) | **ALGORITHM VERIFIED (synthetic only)** — and **fragile**: fails at ≥1° camera misalignment (measured below) | **HW-1 — required.** Whether it works on the real optical pair is unproven |
| C1-b | Software knows which device Id belongs to each role | It does **not** derive this. `resolve_roles` matches an operator-supplied substring | `test_19` refusal cases | **CODE VERIFIED (matching logic only)** | **HW-1 — required.** The role↔socket fact is operator input |
| C1-c | A libcamera `Id` denotes the physical CSI socket | Nothing in code establishes this. `_distinguishing_fragment` merely greps `i2c@` | none | **NOT VERIFIED — background knowledge** | **HW-1.** Must be confirmed on the Pi |
| C1-d | `camera_role_id_hint` gives stable socket binding | Substring match + uniqueness/duplicate checks, `camera_identity.py:207-260` | `test_19` (ambiguous / no-match / duplicate) | **CODE VERIFIED** for the matching; stability inherits C1-c | HW-1 (reboot check) |
| C1-e | Optical check survives real-world conditions | — | — | **DISPROVED for pointing error** (see below) | HW-1 |
| C2 | Focal 930 / 3875 px is arithmetically right for the stated crop | `camera_identity.theoretical_focal_px`; runtime `ScalerCrop` read in `CameraManager._derive_focal_px` | `test_19` C2 checks | **MATH VERIFIED** (re-derived independently this pass) | **HW-2** for the real value |
| C2-b | "Distances carry a few percent of error" | `fusion_config.py:557` prints this at runtime | none — **there is no evidence for this bound** | **OVERCLAIM — DEFECT D1** | HW-2 |
| C3 | No IMX708 in executable code | tokenize scan over all `*.py` | `test_21` | **CODE VERIFIED** | none |
| C4 | Tracker/ego state cleared across a switch | `camera_worker._reset_camera_state`, called from `_loop` | `test_20` (6 checks, both directions) | **CODE VERIFIED** | none |
| H3 | `doa_invert` fully retired | removed from `DEFAULTS`; `RETIRED_KEYS` strip in `calibration.save` | `test_21` (tokenize scan) | **CODE VERIFIED** | none |
| H5 | Gate proves channels are raw microphones | `doa.py`: `n_channels >= 4 or explicit_channels` | `test_22` | **OVERCLAIM — neither condition proves rawness** (D5) | **HW-4/HW-5** |
| H5-b | SRP-PHAT is disabled in the shipped config | `radar_calibration.json` has neither `mic_channels` nor `mic_positions_m` → both defaults | `test_22` | **CODE VERIFIED** | HW-4/5/6 to re-enable |
| H6/M4 | Placeholders nulled, station reports uncalibrated | both JSON files; `is_range_calibrated` | `test_21` | **CODE VERIFIED** | HW-3, HW-8 |
| M1 | Capture vs decision time separated | `target_state.AcousticObservation` (4 instants); `acoustic_worker` passes `t0` | `test_23` | **CODE VERIFIED / MATH VERIFIED** | none |
| M2 | Dead keys removed | dataclass field sets | `test_21` | **CODE VERIFIED** | none |
| M3 | One canonical optics source | `bind_roles`; module dict replaced not merged | `test_19`, `test_24` | **CODE VERIFIED** | none |
| H4 | Beam diagnostic can answer frozen-vs-steady | `diagnose.py cmd_beams` — verified line by line | CLI `--help` verified | **CODE VERIFIED**, with one wording flaw (D6) | **HW-7** |
| N1 | max disagreement = FULL fov, not half | `camera_cue.max_agreement_deg` | `test_19` N1 checks | **MATH VERIFIED** (re-derived: TELE 9.442°, WIDE 37.975°) | none |
| N1-b | `sensor_agreement_discriminating` reports whether the check had power | `camera_cue.agreement_is_discriminating` | `test_19` | **DEFECT D2** — reports `False` in cases where the check *did* reject | HW-6 for a measured tolerance |
| N2 | Duplicate targets = camera boxes, not radar blips | `radar_overlay` single `_draw_target`; `hud` loops `draw_list` | `test_20` N2 checks | **CODE VERIFIED** (mechanism), cause of *your* sighting still unknown | operator observation |
| HW-8 | `calibrate.py range` run twice gives a two-point fit | **FALSE.** `cmd_range` loops for points in ONE session; a second run starts empty and overwrites | — | **DOC DEFECT D4** | HW-8 |
| HW-9 | Metric names exist | `camera_fps`, `hailo_fps`, `jpeg_fps`, `mjpeg_fps`, `frame_age_ms`, `camera_fps_is_sensor` all present in `web_server.py`; `hailo_letterbox/infer/decode/nms` in `latency.ORDER` | `test_17` | **CODE VERIFIED** for names; `hailortcli` syntax **unverified** | **HW-9** |

---

## The decisive finding — C1 is not resolved in practice

I measured the optical classifier's sensitivity by synthesising a TELE view
from a WIDE view under realistic perturbations. Scores are NCC; "verdict" is
what `classify_roles_by_optics` actually returns.

```
condition                                     good  swapped  verdict
ideal                                       +0.898   -0.003  CORRECT
gain x1.6 + bias 20 (exposure diff)         +0.898   -0.003  CORRECT
sensor noise sigma=12                       +0.885   -0.002  CORRECT
sensor noise sigma=30                       +0.823   -0.007  CORRECT
barrel distortion k=0.10                    +0.297   -0.002  CORRECT
pointing off by 1.0 deg                     +0.022   +0.004  INCONCLUSIVE
pointing off by 2.0 deg                     -0.007   +0.003  INCONCLUSIVE
pointing off by 4.0 deg                     -0.002   +0.005  INCONCLUSIVE
pointing off by 8.0 deg                     +0.019   -0.004  INCONCLUSIVE
combined: 2deg + gain + noise12 + k.05      +0.011   +0.003  INCONCLUSIVE
```

**It collapses at one degree of misalignment.** The reason is structural:
`optical_assignment_score` assumes the telephoto view is the *exact centre*
of the wide view. The TELE field is only 9.4° across, so 1° of pointing
error displaces the image by ~68 output pixels — enough to destroy
correlation of a textured scene.

Two cameras bolted to a mast are realistically misaligned by more than 1°;
mechanical tolerance alone is typically ±1–2°.

**Therefore, on the real hardware, the optical check will almost certainly
return INCONCLUSIVE, and the station will fall back to the configured
hint — i.e. to operator input, which is where we started.** The previous
report's "the station determines the roles itself" and "HW-1 reduced to a
one-time confirmation" are both wrong.

**The one genuinely good news:** it never returned a WRONG answer in any
tested condition. It degrades to INCONCLUSIVE, which is fail-closed. The
code is *safe*; it is just far less capable than I claimed.

It is also **fixable**. Searching for the best alignment instead of assuming
a centred crop recovers it completely:

```
condition                             centred   shift  shift-swapped
ideal                                  +0.898  +0.911         +0.042
pointing 1.0 deg                       +0.022  +0.915         +0.042
pointing 4.0 deg                       -0.002  +0.914         +0.042
pointing 8.0 deg                       +0.019  +0.914         +0.035
combined 2deg+gain+noise12+k.05        +0.011  +0.500         +0.039
```

That is a ~5-line change (`cv2.matchTemplate` with `TM_CCOEFF_NORMED`
instead of a fixed centre crop). **I have not made it** — see PENDING.

---

## PENDING CODE CHANGES — awaiting your approval

None of these have been applied.

### D1 — `fusion_config.py:557` prints a fabricated error bound *(highest priority)*

```python
f"measurement. Distances carry a few percent of "
f"error. Calibrate: HARDWARE_TEST_REQUIRED.md HW-2")
```

There is **no evidence** for "a few percent". It is not measured, not
derived, and not bounded by anything in the code. It assumes the lens is
within a few percent of nominal, the projection is distortion-free, and the
mount is ideal — none of which is checked. This is a runtime, operator-facing
string, and it is exactly the failure this project's own standard forbids: a
plausible-looking number standing in for a measurement.

**Proposed:** replace with "the real distance error is UNKNOWN until HW-2 is
performed." Same correction in `camera_identity.theoretical_focal_px`'s
docstring (`:449-452`) and `fusion_config.py:465,469`.

### D2 — `agreement_is_discriminating` reports False when the check did discriminate

Measured this pass, TELE, boresight 150°, box at frame centre:

```
TELE discriminating flag: False
  bearing 150.0 -> agreement   0.0 deg, exceeds tolerance? False
  bearing 160.0 -> agreement  10.0 deg, exceeds tolerance? False
  bearing 190.0 -> agreement  40.0 deg, exceeds tolerance? True   <-- rejected
```

The flag is a **camera-geometry** property (can an *in-view pair* ever
exceed the tolerance?) but is published per observation as though it
described *that* check. When the acoustic bearing is off-screen the gate
does reject, and the snapshot still says the check had no power.

**Proposed (either):**
* make it per-observation — discriminating iff `|bearing_rel| + half_fov >
  tolerance`; or
* rename to `agreement_span_covers_tolerance` and document it as a camera
  property.

The first is more useful and is a two-line change.

### D3 — `suggest_hint_config` can emit a config that can never work

Reproduced this pass with both sensors on one I2C controller:

```
"camera_role_id_hint": { "WIDE": "i2c@88000", "TELE": "i2c@88000" }
-> REFUSED: camera role WIDE ... matches 2 cameras — the hint is ambiguous
```

`_distinguishing_fragment` returns the first `i2c@` node, which is identical
for both cameras in that topology. The operator is handed a suggestion that
the very next startup rejects. It fails closed, so nothing unsafe happens.

**Proposed:** fall back to the shortest suffix that is unique across the
reported cameras.

### D4 *(documentation, not code)* — HW-8 does not match the CLI

`calibrate.py cmd_range` collects points in an **interactive loop inside one
invocation** (`while True: input("Відстань ...")`). Running
`python calibrate.py range` twice does **not** accumulate two points: the
second run starts with an empty list, takes the single-point branch, keeps
`range_spreading_db` at whatever is already in the file, and **overwrites**
the reference distance and level.

My HW-8 told the operator to run it twice. Following it would produce a file
with a plausible-looking `range_spreading_db` that was never fitted to
anything — the exact H6 failure, recreated by my own procedure.

Also: with a single point the tool *does* write `range_spreading_db` (the
assumed 22.0), and it silently substitutes 22.0 whenever a fit lands outside
12–40. So **the presence of that key does not prove a two-point fit.**

**Corrected in `HARDWARE_TEST_REQUIRED.md` this pass** (documentation only).

### D5 — the SRP-PHAT gate does not establish "raw"

```python
raw_channels_available = n_channels >= 4 or explicit_channels
```

Neither disjunct proves the channels are raw microphones:

* `n_channels >= 4` — a device can expose four *processed* channels (beams,
  or 2 beams + 2 references). Channel count is not channel semantics.
* `explicit_channels` — `mic_channels: [0,1]` is an operator typing a tuple
  into JSON. It is an assertion, not a measurement.

**This is currently harmless**: the shipped `radar_calibration.json` contains
neither key, so both conditions fail and SRP-PHAT is disabled. But the gate
can be opened by editing configuration alone, with no hardware evidence —
which is what H5 was about.

**Proposed:** require an explicit attestation key written *only* by a
hardware procedure (e.g. `mic_channels_verified: true`, set by
`calibrate.py check` after the tap test), and drop `n_channels >= 4` as
independent proof. Report wording corrected this pass regardless.

### D6 — the beam verdict can print FROZEN from a static scene

`diagnose.py` prints `FROZEN (no change in the whole run)` whenever
`changes == 0` and `n >= 100`. But a beam that is working perfectly and
watching a source that never moved also has `changes == 0`. The 100-sample
threshold does not distinguish them — **only changing the scene does**, and
the instruction to do that is printed *only when the run was too short*.

So a 200-sample run with a static scene yields a confident, unqualified
`FROZEN` — the precise error H4 exists to prevent.

**Proposed:** always qualify — `FROZEN *if the scene changed during this
run*; otherwise INCONCLUSIVE` — and print the scene-change instruction on
every run, not only short ones.

---

## Confirmed correct — no change needed

* **N1 mathematics.** Re-derived independently: for two angles both within
  ±h, `max|b−θ| = 2h`. TELE 9.442°, WIDE 37.975°, tolerance 20° → TELE
  cannot reject an in-view pair, WIDE can. The earlier half-field version was
  wrong; the current full-field version is right.
* **C2 arithmetic.** Re-derived: 6 mm → 929.96 px, 25 mm → 3874.84 px at
  crop 2664 / output 640; ratio 4.1667 exactly equals 25/6. Full-FOV mode
  gives 610.80 / 2545.01 — confirming the value is mode-dependent and must
  be read from `ScalerCrop`, as the code does.
* **HW-7 CLI.** `--samples` (default 200), `--interval`, `--timeout`,
  `--show`, `--script` all exist; change-count and longest-identical-run are
  genuinely computed; `FROZEN` is gated at ≥100 samples. Only the wording
  flaw D6 applies.
* **HW-9 metric names.** All exist in `web_server.py` and `latency.ORDER`.
  One nuance: the dashboard's `hailo_fps` is the *detector invocation rate*,
  measured around `predict_with_scores()`, which includes CPU letterbox,
  decode and NMS — it is not pure NPU throughput. The per-stage split
  separates them. `hailortcli benchmark` syntax is **unverified** — check it
  against your installed version.
* **N2.** Left cautious, as instructed. The radar tile draws one target from
  one un-looped call; the HUD draws one box per track. C4 is a *confirmed
  mechanism* for duplicate **camera boxes**. Whether your observed duplicate
  was two boxes or two radar blips is **not established** — if it was two
  blips, C4 is not the cause and the cause is not yet found.

---

## What did NOT change

No production code was modified in this review. `git diff` for this pass
touches only `FINAL_FORENSIC_REVIEW.md` (new),
`CORRECTED_PROJECT_REPAIR_VALIDATION.md` (new),
`CORRECTED_HARDWARE_TEST_REQUIRED.md` (new), and corrections applied to the
two existing reports. The test suite still passes at **382/382** because
none of D1–D6 is covered by an existing assertion — which is itself worth
noting: a green suite did not catch two overclaims and a wrong procedure.

---

## FINAL VERDICT

```
NOT READY — CODE ISSUES REMAIN
```

**Why.** Three confirmed software defects are unfixed (D1, D2, D3), plus two
that weaken guarantees the reports claim (D5, D6). D1 is the serious one: the
station tells the operator, at runtime, that its distances carry "a few
percent of error", and that number is invented. On a project whose central
rule is never to let a plausible default stand in for a measurement, shipping
that string is not a documentation nit.

**What this is not.** No detection-logic bug was found. C2, C3, C4, H3, H6,
M1, M2, M3 and the N1 mathematics all hold up under inspection, and the
optical role check — though far weaker than I claimed — fails safe rather
than wrong.

**To reach READY FOR HARDWARE VALIDATION:** approve D1, D2 and D3 (small,
contained), and decide on D5 and D6. The optical robustness improvement is
optional but would make C1's automatic path actually usable rather than
theoretical.

**`READY — HARDWARE VERIFIED` remains unavailable** and is not claimed: this
session had no Pi, no camera, no ReSpeaker and no Hailo.

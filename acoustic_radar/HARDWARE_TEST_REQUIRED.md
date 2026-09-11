# HARDWARE TESTS REQUIRED

Everything in this file is a claim the **software cannot prove**. It is listed
here instead of being answered with a plausible default.

The repair session that produced this file ran on a Windows development
machine. There is no Raspberry Pi, no libcamera, no Hailo device and no
ReSpeaker on it — `uname` reports `MINGW64_NT-10.0`, there is no
`/proc/asound`, and `arecord` is not installed. **No hardware measurement in
this file has been performed.** Every "Expected" below is a specification of
what to look for, not a result.

Record results by editing the **RESULT** line of each test. Leave it as
`NOT PERFORMED` until it genuinely has been.

---

## HW-1 — Establish and confirm the camera role mapping

**Blocks:** the camera subsystem, unless the optical check happens to
succeed. **Treat this as required.**

### How the station resolves this, and what is still unproven

The two lenses differ by 25/6 = 4.17× in magnification, so the telephoto
image is a region near the centre of the wide image, blown up.
`CameraManager` grabs one frame from each camera at start-up and **searches
for the alignment** (bounded normalised cross-correlation, ±25% of the wide
frame) rather than assuming the views are concentric.

An earlier implementation assumed an exact centre crop and collapsed at 1° of
pointing error — inside normal mounting tolerance. That is fixed; measured on
synthetic pairs:

```
pointing offset 1°, 4°, 8°            -> resolved correctly (NCC ~0.91)
offset in both axes                   -> resolved correctly
offset + exposure difference + noise  -> resolved correctly
featureless scene                     -> INCONCLUSIVE (refuses)
self-similar scene                    -> INCONCLUSIVE (non-unique alignment)
```

⚠️ **THESE ARE SYNTHETIC RESULTS AND DO NOT ESTABLISH HARDWARE
BEHAVIOUR.** They model pointing offset, exposure, noise and distortion on
generated images. Real optics, real parallax, real scenes and the real
sensor pipeline are not represented. **That is what this test is for.**

It never produced a *wrong* answer in any tested condition — it degrades to
INCONCLUSIVE, which is fail-closed.

At start-up you will see one of:

```
[CameraManager] camera roles resolved by lens magnification (measured from the images): WIDE=index 0, TELE=index 1
[CameraManager] camera roles resolved by device Id, CONFIRMED by lens magnification: ...
[CameraManager] camera roles resolved by device Id (optical confirmation unavailable this run): ...
```

### What still needs you

**(a) Confirm what the station decided, by eye.** Misalignment no longer
defeats the check, but `INCONCLUSIVE` is still an expected, correct outcome
when the cameras see a featureless scene — empty sky, which is this
station's usual view. It refuses rather than guessing. Point the cameras at
something with visible structure once and restart if you want the automatic
result.

**(b) STILL RECOMMENDED: pin the mapping.** Setting the hint makes the
mapping independent of what the cameras happen to be looking at, and turns
the optical check into a permanent cross-check that will catch a cable being
moved later. Without it, every start-up depends on the scene.

```bash
libcamera-hello --list-cameras
# or:
python -c "from picamera2 import Picamera2; \
           import json; print(json.dumps(Picamera2.global_camera_info(), indent=2))"
```

### ✅ OBSERVED ON THIS STATION — the Ids are known

Run on the operator's Pi 5 (`Picamera2.global_camera_info()`):

```json
[
  { "Model": "imx477", "Num": 0, "Rotation": 180,
    "Id": "/base/axi/pcie@1000120000/rp1/i2c@88000/imx477@1a" },
  { "Model": "imx477", "Num": 1, "Rotation": 180,
    "Id": "/base/axi/pcie@1000120000/rp1/i2c@80000/imx477@1a" }
]
```

This **confirms**, on the real hardware:

* both cameras are `imx477` — consistent with the stated IMX477P, and the
  IMX708 references removed in C3 were indeed wrong;
* the two `Id`s differ in the I2C controller node, `i2c@88000` vs
  `i2c@80000`, so the assumption behind `camera_role_id_hint` holds here and
  those two fragments are valid, unambiguous hints;
* both report `Rotation: 180` — consistent with each other, so the optical
  role check compares like with like.

⚠️ **WHAT THIS STILL DOES NOT TELL US: which of them carries the 6 mm lens
and which the Kowa 25 mm.** Nothing in this output distinguishes them —
identical model, identical rotation. That is the remaining part of HW-1 and
it needs eyes on the images. **Do not fill in the hint from a guess:** a
wrong hint that goes unnoticed is precisely the silent lens transposition
this whole mechanism exists to prevent.

Determine it either way below, then record the matching Id fragments:

```json
"geometry": {
  "camera_role_id_hint": {
    "WIDE": "i2c@88000",
    "TELE": "i2c@80000"
  }
}
```

With both a hint and a usable scene, the two sources **cross-check each
other** and the station reports `CONFIRMED`. A disagreement is a hard error:
it means a camera was moved to the other socket, or the lenses were swapped
between bodies — the exact silent transposition this mechanism exists to
catch.

### Deciding which is which — 30 seconds, and it settles HW-1

```bash
# Newer Raspberry Pi OS:
rpicam-still --camera 0 -o /tmp/cam0.jpg -n -t 800
rpicam-still --camera 1 -o /tmp/cam1.jpg -n -t 800
# Older OS uses libcamera-still with the same arguments.
```

Open both images. **The TELE (25 mm) image is dramatically more
magnified — about 4.17×.** It shows a small central slice of what the WIDE
(6 mm) image shows. This is unmistakable by eye; no measurement is needed.

Then, for the Ids observed above:

| If the magnified image is… | WIDE hint | TELE hint |
| --- | --- | --- |
| `--camera 1` (`i2c@80000`) | `i2c@88000` | `i2c@80000` |
| `--camera 0` (`i2c@88000`) | `i2c@80000` | `i2c@88000` |

⚠️ If the two images look **similarly wide**, stop: the lenses are not what
the hardware notes say, and every distance, gate and threshold downstream is
void until that is resolved.

**Cross-check:** with a textured scene in view the station works this out
itself and prints its answer. If your hint and its measurement disagree it
refuses to start and says so — that is the check doing its job, not a fault.

### Reboot stability (the point of the whole fix)

```bash
sudo reboot
# after boot, re-run step 1 and compare
```

* **PASS** — the station starts and reports the same role→index mapping,
  *or* reports a different index for the same socket and still assigns the
  roles correctly. Both are success: the role follows the socket, not the
  enumeration order.
* **FAIL** — the station refuses to start, or a role maps to a socket you did
  not configure. Then a cable was moved; re-do this test.

**RESULT: NOT PERFORMED — but no longer blocking. The mapping is resolved
automatically from the optics; this test confirms it and pins it.**

---

## HW-2 — Real focal-length calibration

**Blocks:** C2, H2, and the camera-switch thresholds.

### Why software cannot answer it

A focal length in pixels is a property of the lens *and* the sensor mode
*and* the actual glass. The values now in the config are **THEORETICAL**:
computed from the IMX477 pixel pitch, the nominal lens length and the
sensor's real `ScalerCrop`. They assume a "25 mm" lens is exactly 25 mm and
the projection is distortion-free. Neither is measured.

The values they replaced (1274 px and 501.7 px) were provably wrong: their
ratio was 2.54 where the lenses demand 25/6 = 4.17.

### Procedure

For **each** role, one at a time:

1. Start the station. Press `f` to freeze camera switching (otherwise the
   camera swaps out from under you at the 1.5 m threshold).
2. Place the drone at a **measured** distance, square-on to the camera, so
   the detector box is **at least 70 px wide** (`MIN_CALIBRATION_BOX_PX`).
   For TELE's narrow field this will be much further away than for WIDE.
   Measure the distance with a tape, not by eye.
3. Confirm the box is **tight** around the drone. An inflated box is the
   documented cause of the historical 2x error.
4. Press `c`. The log prints the derived focal length.
5. **Repeat at a second, well-separated distance.** For a pinhole camera
   `box_width_px × distance_m` must be constant. If the two disagree by more
   than ~5%, one of them is wrong — do not average them, find out which.
6. Also record the drone's **real width** in metres and put it in
   `geometry.drone_real_width_m`. Every distance scales linearly with it.

### Record the answer

```json
"geometry": {
  "camera_focal_px_by_role": { "WIDE": 000.0, "TELE": 0000.0 },
  "camera_focal_source":     { "WIDE": "CALIBRATED", "TELE": "CALIBRATED" }
}
```

* **PASS** — two distances agree within ~5%, and the result is in the same
  region as the theoretical value (930 px WIDE / 3875 px TELE at 640 px
  output in the 1332×990 mode). The start-up "focal is THEORETICAL" warning
  disappears.

  ⚠️ The ~5% and "same region" figures are **sanity bands for this
  procedure, not an error bound on the theoretical value**. How far the
  theoretical numbers are from the truth is **UNKNOWN until this test is
  performed** — that is the entire reason this test exists. Any statement
  elsewhere that theoretical distances are accurate "to a few percent" is
  unsupported; see FINAL_FORENSIC_REVIEW.md defect D1.
* **FAIL** — the two distances disagree, or the result is far from
  theoretical. A large disagreement usually means the box was not tight, or
  the sensor is in a different mode than assumed. Check the mode the station
  logs before trusting either number.

**After this passes, re-check `switch_to_near_below_m` / `switch_to_far_above_m`
(1.5 / 2.0 m) and `cue_agreement_deg`** — see finding N1. Those thresholds
were chosen under the old, wrong focal lengths.

**RESULT: NOT PERFORMED**

---

## HW-3 — Camera boresight

**Blocks:** M4. The acoustic cue is **disabled** until this is measured.

### Why software cannot answer it

The boresight is where the camera's optical axis points, expressed in the
acoustic frame. It depends on how the two sensors are physically bolted to
the mast. Nothing in software knows that. The config previously carried
`0.0` for both cameras, which is the signature of a placeholder — two
physically separate cameras having an *identical* boresight of *exactly*
zero is not a measurement. Those are now `null`.

### Procedure

1. Complete HW-5 first (the DOA zero must be measured, or the frame this
   boresight is expressed in is itself unknown).
2. Put a steady sound source (a drone hovering, or a speaker playing
   broadband noise) somewhere the camera can also see it.
3. Move the source until it sits **exactly on the vertical centre line** of
   the camera image.
4. Read the bearing the HUD reports for it. **That number is the boresight
   for that camera.**
5. Repeat for the other camera without moving the mast.

### Record the answer

```json
"geometry": {
  "camera_boresight_deg_by_role": { "WIDE": 000.0, "TELE": 000.0 },
  "boresight_calibrated_at_doa_offset_deg": 180.0,
  "boresight_calibrated_handedness": "CW"
}
```

The last two fields **must** record the `doa_offset_deg` and
`doa_handedness` in force at the moment of measurement. They arm the
staleness guard: change the offset later and the station will tell you the
boresight has gone stale instead of silently pointing the cue the wrong way.
Leaving them null leaves the guard disarmed.

* **PASS** — the cue band lands on the source in both cameras.
* **FAIL** — it is offset by a constant: re-read the bearing more carefully.
  Offset by roughly *twice* the bearing: the handedness is mirrored, go back
  to HW-5.

⚠️ The two boresights should be **close but not identical**. If you measure
exactly the same number for both, suspect that you did not actually re-aim.

**RESULT: NOT PERFORMED**

---

## HW-4 — XVF3800 USB channel semantics

**Blocks:** H5, H7, and the meaning of every raw audio channel.

### Why software cannot answer it

The board has **4 physical microphones**. That is a hardware fact. It says
nothing about how many channels the firmware exposes over USB, or what those
channels *are*. Two channels may be a processed beamformed stereo pair, or
two raw microphones — the sample stream looks the same either way.

**Neither direction may be assumed.** "2 ALSA channels" is not proof that
only two microphones work, and "4 microphones on the board" is not proof
that Python can read four raw channels.

### Procedure

```bash
arecord -l
arecord -L
cat /proc/asound/cards
cat /proc/asound/devices

# Substitute the real card/device numbers:
arecord -D hw:1,0 --dump-hw-params
```

Record verbatim: `channels`, `rate`, `format`, and the device name.

Then, from the station's virtualenv:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
python calibrate.py check          # per-channel levels
```

### Classify the result as exactly one of

| | Meaning |
| --- | --- |
| **A — 4 RAW MICROPHONES VERIFIED** | `--dump-hw-params` offers 4+ channels *and* HW-5 confirms each responds independently |
| **B — 2 PROCESSED CHANNELS, EXPECTED XVF3800 MODE** | the device offers only 2, and they are the DSP's output. **This is a legitimate configuration, not a fault.** |
| **C — 2 CHANNELS, CONFIGURATION PROBLEM** | the firmware *can* expose 4 but is not doing so — a different USB mode or firmware build is needed |
| **D — BLOCKED** | the commands could not be run |

⚠️ **Do not force 4 channels merely because the board has 4 microphones.**
If the XVF3800 is correctly delivering 2 processed channels, that is the
right answer — but then those two channels must **not** be treated as
"physical microphone 0 and 1". They are beamformer outputs.

**RESULT: NOT PERFORMED — currently believed to be B, unconfirmed**

---

## HW-5 — Physical microphone mapping

**Blocks:** H5, SRP-PHAT.

### Why software cannot answer it

Even with 4 raw channels available, *which* channel is *which* microphone in
`mic_positions_m` order cannot be derived. `mic_channels` currently defaults
to "the first N channels", which is an assumption.

### Procedure

Only meaningful if HW-4 returned **A**.

1. Run `python calibrate.py check`, which prints per-channel levels live.
2. **Tap each microphone in turn**, firmly, one at a time. Note which
   channel index jumps.
3. Record, for each physical microphone position, the channel index that
   responded:

   ```
   mic at (-21.5, +21.5) mm -> channel ?
   mic at (+21.5, +21.5) mm -> channel ?
   mic at (+21.5, -21.5) mm -> channel ?
   mic at (-21.5, -21.5) mm -> channel ?
   ```

4. Cross-check with a steady source off to one side: the nearer microphones
   should lead the further ones in cross-correlation.

Write the result into `radar_calibration.json` as `mic_channels`, ordered to
match `mic_positions_m`.

### ⚠️ THE LIST ALONE DOES NOT ARM SRP-PHAT — AND MUST NOT

`mic_channels: [0,1,2,3]` is you typing a tuple into JSON. It is an
assertion, not a measurement, and SRP-PHAT deliberately ignores it on its
own. Only after you have **actually performed the tap test above** and seen
each channel respond independently may you record:

```json
"mic_channels": [0, 1, 2, 3],
"mic_channels_verified": true
```

`mic_channels_verified` defaults to `false` and exists precisely so that
arming SRP-PHAT is a deliberate act recording "a human ran this test",
rather than a side effect of filling in a config field. Do not set it
because the array *should* have four microphones.

* **PASS** — every channel responds to exactly one microphone, and the four
  are distinct. Record both keys.
* **FAIL** — two channels respond identically (they are the same beam), or a
  channel responds to nothing. Then HW-4 was really B or C, not A: leave
  `mic_channels_verified` false and SRP-PHAT correctly stays off.

**RESULT: NOT PERFORMED**

---

## HW-6 — Microphone array geometry

**Blocks:** H5/H7, SRP-PHAT accuracy.

### Why software cannot answer it

`mic_positions_m` is currently the **default** 43 mm square taken from the
ReSpeaker 4-Mic reference design. It has never been measured against this
board. A wrong baseline does not shift the angle by a constant — it warps
the mapping non-uniformly, and no calibration offset can undo that.

SRP-PHAT is now **disabled** while this geometry is the untouched default.

### Procedure

1. Find the official mechanical drawing for the **XVF3800** board being used
   (not the 4-Mic Array — a different product with a different spacing).
2. Failing that, measure it: with callipers, from the centre of each
   microphone port to the centre of each other. Record all six pair
   distances, not just the square's side, so the layout is over-determined
   and a mistake shows up as an inconsistency.
3. Write positions in metres, in the same order as `mic_channels`, with the
   array's own 0° along +X.

```json
"mic_positions_m": [[-0.0215, 0.0215], [0.0215, 0.0215],
                    [0.0215, -0.0215], [-0.0215, -0.0215]],
"mic_geometry_verified": true
```

⚠️ **Both keys are required.** A non-default `mic_positions_m` alone does
not arm SRP-PHAT — somebody could have edited the numbers without measuring
anything. `mic_geometry_verified` defaults to `false` and records that this
test was actually performed. Set it only after callipers or the official
drawing for **this** board.

* **PASS** — measured distances agree with the drawing to ~1 mm, and
  SRP-PHAT (once armed) produces bearings consistent with the USB DSP.
* **FAIL** — SRP-PHAT and the USB DSP disagree systematically. Do **not**
  adjust the geometry to make the answer look better; that is fitting the
  model to the desired output.

**RESULT: NOT PERFORMED**

---

## HW-7 — Are DSP beams 0 and 1 frozen?

**Blocks:** H4.

### Why software cannot answer it

A beam reporting one value for seven seconds is equally consistent with
"the DSP is not updating it" and "the source genuinely did not move". The
earlier 20-sample run could not tell those apart, and 20 samples is not
enough to call a beam permanently frozen.

This matters because a constant always has spread 0.0, so a frozen beam
always looks like the tightest cluster to `select_azimuth()` and can raise
the reported confidence without contributing evidence.

### Procedure

```bash
python diagnose.py beams --samples 200 --interval 0.5 --scene-changed
```

That is ~100 seconds. **Partway through the run, deliberately change the
acoustic scene** — move the source to a clearly different bearing, or switch
it off. Write down the wall-clock moment you did it.

⚠️ **`--scene-changed` is your attestation that you actually did that, and
the tool will not report any beam as FROZEN without it.** Run without the
flag and every silent beam comes back `NOT PROVEN — STATIC SCENE`, because a
healthy beam watching a still source is indistinguishable from a dead one.
Do not pass the flag unless you really changed the scene.

The tool now reports, per beam: sample count, unique values, mean, min, max,
spread, standard deviation, variance, the number of times the value changed,
and the longest run of identical consecutive readings.

The tool prints one verdict per beam, and only these are possible:

* **`FROZEN`** — zero changes across ≥100 samples **and** `--scene-changed`
  was given. The beam did not follow a real change in the scene, so it is
  not being updated. This is the only verdict that justifies excluding it.
* **`RESPONDED`** — the beam changed, including across your source change.
  It is live; do not exclude it.
* **`NEARLY STATIC`** — few changes, but it does update. Not frozen.
* **`NOT PROVEN — STATIC SCENE`** — zero changes, but `--scene-changed` was
  not given. Says nothing about the beam. Re-run properly.
* **`INSUFFICIENT`** — fewer than 100 samples parsed. Re-run longer.

⚠️ Do not delete a beam on anything but a `FROZEN` verdict. The tool cannot
observe whether you changed the scene, so it relies on your flag; passing
`--scene-changed` when you did not actually move the source would recreate
exactly the error finding H4 exists to correct.

**RESULT: NOT PERFORMED**

---

## HW-8 — Acoustic range calibration

**Blocks:** H6. Acoustic distance in metres is **unavailable** until done.

### Why software cannot answer it

The reference level of a drone at a known distance depends on that specific
drone and that specific site. There is no theoretical substitute — unlike
focal length, nothing can be derived from first principles.

The shipped file previously contained `range_ref_distance_m: 3.0` and
`range_ref_level_dbfs: -22.0`: both suspiciously round, and with
`range_spreading_db` absent — a key `calibrate.py range` always writes
alongside the other two. That is the signature of hand-editing, not of the
calibration tool. **Both are now `null`.**

### Procedure

### ⚠️ CORRECTED PROCEDURE — READ THIS, THE EARLIER VERSION WAS WRONG

An earlier draft of this file told you to run `python calibrate.py range`
**twice**, once per distance. **That does not work and would have produced a
fake calibration.** Verified against `calibrate.py:230-311`:

`cmd_range` collects points in an **interactive loop inside a single
invocation** — it keeps prompting `Відстань до дрона у метрах` until you
press Enter on an empty line. A *second* invocation starts with an empty
list, takes the one-point branch, leaves `range_spreading_db` at whatever is
already in the file, and **overwrites** the reference distance and level.

Following the old instructions would have left a file containing a
plausible-looking `range_spreading_db` that was never fitted to anything —
recreating finding H6 exactly.

### Procedure

```bash
# 1. Background noise, with the drone OFF and the site otherwise as it will
#    be in use. This is subtracted from every later level.
python calibrate.py noise

# 2. ONE invocation. Enter BOTH distances at its prompts.
python calibrate.py range
```

Inside that single run:

```
Відстань до дрона у метрах (Enter — завершити, зібрано 0): 5
   Тримайте дрон на 5 м і натисніть Enter...        <- hover at 5 m
Відстань до дрона у метрах (Enter — завершити, зібрано 1): 20
   Тримайте дрон на 20 м і натисніть Enter...       <- hover at 20 m
Відстань до дрона у метрах (Enter — завершити, зібрано 2):  <- press Enter
```

With two or more points it prints `Підгонка по N точках:` and the fitted
`Згасання: X дБ/декаду`. **If you do not see that line, you did not give it
two points and the exponent is still an assumption.**

* **PASS** — the run printed `Підгонка по N точках` with N ≥ 2, and the
  fitted spreading landed in 20–26 dB/decade (20 = free-field spherical
  spreading; over ground with absorption, 22–26 is normal).
* **FAIL** — only `Одна точка:` was printed (one point; the exponent stays
  assumed), or the fit was rejected. Note that when the fit falls outside
  12–40 the tool **silently substitutes 22.0** and says so — read the output,
  do not just check the file.

⚠️ **The presence of `range_spreading_db` in the JSON does NOT prove a
two-point fit.** The single-point branch writes it too, carrying the assumed
22.0 forward. Only the `Підгонка по N точках` line in the run output is
evidence. Record that output.

* Any point where the drone is less than 3 dB above the noise floor is
  discarded by the tool with a warning — check you did not lose one that way.
* Do **not** hand-edit the numbers to look reasonable.

**RESULT: NOT PERFORMED**

---

## HW-9 — Hailo throughput and the real frame budget

**Blocks:** the performance figures in any report.

### Why software cannot answer it

No Hailo device exists on the development machine. Published "Pi 5 +
Hailo-8L does 60 fps" figures are pipelined NPU throughput, which is not the
same measurement as this station's serial per-frame loop.

### Procedure

```bash
hailortcli benchmark bestty_yolo260508.hef
```

⚠️ **This exact invocation is UNVERIFIED.** No Hailo tooling exists on the
machine this was written on, and `hailortcli` sub-command syntax has changed
between HailoRT releases. Run `hailortcli --help` / `hailortcli benchmark
--help` first and adapt. Do not treat the line above as known-good.

The metric names below **are** verified to exist in the current code
(`web_server.py` and `latency.ORDER`).

Then run the station and read the four separate rates it now reports:
camera (sensor) FPS, loop FPS, detector FPS, JPEG FPS — plus the per-stage
split (`hailo_letterbox`, `hailo_infer`, `hailo_decode`, `hailo_nms`) that
the latency budget records.

Record:

```
hailortcli FPS            :
hailortcli HW latency     :
hw_only vs streaming      :          (equal => no PCIe bottleneck)
measured predict_with_scores total (ms):
  letterbox / infer / decode / nms   :
sensor fps / loop fps / detector fps :
```

* If `infer` dominates, the NPU is saturated and only a different model or
  async submission helps.
* If `letterbox`+`decode`+`nms` dominate, the NPU is idle and the fix is on
  the CPU side.

⚠️ **The dashboard field named `hailo_fps` is not pure NPU throughput.** It
is the *detector invocation rate*, measured around `predict_with_scores()`,
which includes the CPU letterbox, decode and NMS stages. The per-stage split
is what separates NPU time from CPU time. Read the name as "detector rate".

⚠️ Report these as measurements only. **Do not quote a Hailo FPS anywhere
without having run this.**

**RESULT: NOT PERFORMED**

---

## Summary

| Test | Subject | Blocks | Result |
| --- | --- | --- | --- |
| HW-1 | Camera role ↔ physical socket | C1 — **required in practice** | NOT PERFORMED |
| HW-2 | Focal-length calibration | C2, H2, N1 | NOT PERFORMED |
| HW-3 | Camera boresight | M4 | NOT PERFORMED |
| HW-4 | XVF3800 USB channel semantics | H5, H7 | NOT PERFORMED |
| HW-5 | Physical microphone mapping | H5 | NOT PERFORMED |
| HW-6 | Microphone array geometry | H5, H7 | NOT PERFORMED |
| HW-7 | DSP beam freeze | H4 | NOT PERFORMED |
| HW-8 | Acoustic range calibration | H6 | NOT PERFORMED |
| HW-9 | Hailo throughput | performance | NOT PERFORMED |

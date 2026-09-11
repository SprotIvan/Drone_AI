# CORRECTIONS TO HARDWARE_TEST_REQUIRED.md

An **errata record**. The corrections below have already been **applied in
place** to `HARDWARE_TEST_REQUIRED.md`; this file exists so the changes are
auditable.

Each correction is a case where the documented procedure did not match what
the CLI or the code actually does.

---

## 1. HW-8 — THE PROCEDURE WAS WRONG AND WOULD HAVE FAKED A CALIBRATION

**This is the most serious correction in this pass.**

**Was:**
```bash
python calibrate.py noise
python calibrate.py range     # first distance
python calibrate.py range     # second distance
```
with "Two points are what fit the spreading exponent".

**Verified against `calibrate.py:230-311`:** `cmd_range` collects points in
an **interactive loop inside a single invocation** —

```python
points: list[tuple[float, float]] = []
while True:
    raw = input(f"   Відстань до дрона у метрах "
                f"(Enter — завершити, зібрано {len(points)}): ").strip()
    if not raw:
        break
    ...
    points.append((distance, target))
```

A **second invocation starts with an empty `points` list**, takes the
one-point branch, leaves `range_spreading_db` at whatever is already in the
file, and **overwrites** `range_ref_distance_m` / `range_ref_level_dbfs`.

Following the old instructions would have produced a file containing a
plausible-looking `range_spreading_db` that was never fitted to anything —
**recreating finding H6 by way of my own procedure.**

**Now:** one invocation, both distances entered at its prompts, with the
exact prompt transcript shown and the `Підгонка по N точках:` line named as
the *only* evidence that a real fit happened.

**Also added — two traps found in the same code:**
* the single-point branch **still writes** `range_spreading_db` (the assumed
  22.0), so **the key's presence does not prove a two-point fit**;
* when a fit lands outside 12–40 dB/decade the tool **silently substitutes
  22.0** (`calibrate.py:301-305`) — the run output must be read, not just the
  resulting JSON.

---

## 2. HW-1 — overstated the automation

**Was:** "now resolved automatically… this is a confirmation, not a
prerequisite", `Blocks: nothing, in the normal case`.

**Why wrong:** measured sensitivity of the optical check —

```
pointing off by 1.0 deg   -> NCC +0.022  -> INCONCLUSIVE
pointing off by 2.0 deg   -> NCC -0.007  -> INCONCLUSIVE
pointing off by 8.0 deg   -> NCC +0.019  -> INCONCLUSIVE
```

1° of misalignment defeats it, which is inside normal mounting tolerance.

**Now:** `Blocks: the camera subsystem… treat this as required`; step (b)
(pinning `camera_role_id_hint`) is marked **REQUIRED IN PRACTICE**, and
operators are told not to spend time coaxing the optical check. The
fail-safe behaviour (INCONCLUSIVE, never wrong) is stated so the change does
not read as alarm.

**Also added:** a warning that the `i2c@NNNNN`-identifies-the-CSI-socket
assumption is **not proven by any code**, and what to do if both suggested
hints come out identical (defect D3).

**Summary table:** HW-1's "Blocks" column changed from
*(confirmation only)* back to **C1 — required in practice**.

---

## 3. HW-2 — sanity band was phrased as an error bound

**Was:** "the result is within ~10% of the theoretical value".

**Why wrong:** read as a claim about how accurate the theoretical value is.
It is not — it is a sanity check on the *procedure*.

**Now:** explicitly labelled a sanity band for the procedure, with the
statement that **how far the theoretical numbers are from truth is UNKNOWN
until this test is performed** — which is the reason the test exists — and a
pointer to defect D1.

---

## 4. HW-7 — the FROZEN verdict needed a caveat

**Verified present and correct:** `--samples` (default 200), `--interval`,
`--timeout`, `--show`, `--script`; change-count and longest-identical-run
genuinely computed; `FROZEN` gated at ≥100 samples.

**What was missing:** the tool prints `FROZEN (no change in the whole run)`
whenever `changes == 0` and `n >= 100` — but a healthy beam watching a
source that never moved also has `changes == 0`. The sample count cannot
separate them; **only changing the scene can**, and that instruction is
printed only when the run was *too short*.

**Now:** an explicit warning that the `FROZEN` verdict is valid **only if you
changed the scene**, and to treat it as INCONCLUSIVE otherwise. Recorded as
defect **D6** for a tooling fix.

---

## 5. HW-9 — unverified command, and a metric-name nuance

**Was:** `hailortcli benchmark bestty_yolo260508.hef` presented as known-good.

**Now:** marked **UNVERIFIED** — no Hailo tooling exists on the machine this
was written on and `hailortcli` sub-command syntax has changed across
HailoRT releases; run `hailortcli --help` first.

**Metric names ARE verified** to exist in the current code: `camera_fps`,
`hailo_fps`, `jpeg_fps`, `mjpeg_fps`, `frame_age_ms`, `camera_fps_is_sensor`
in `web_server.py`; `hailo_letterbox` / `hailo_infer` / `hailo_decode` /
`hailo_nms` in `latency.ORDER`.

**Nuance added:** the dashboard's `hailo_fps` is the **detector invocation
rate**, measured around `predict_with_scores()` and therefore including the
CPU letterbox, decode and NMS stages — not pure NPU throughput. Only the
per-stage split separates the two.

---

## 6. HW-4 / HW-5 / HW-6 — re-checked, NO correction needed

Explicitly re-read against the forbidden assumptions. The document does
**not** assume any of them:

| Assumption to avoid | Status in the document |
| --- | --- |
| 4 USB channels ⇒ 4 raw microphones | **Avoided.** Outcome A requires `--dump-hw-params` **and** HW-5 confirmation |
| 2 channels ⇒ processed stereo | **Avoided.** Outcomes B and C are separate, and B is stated to be legitimate |
| channel index == physical mic index | **Avoided.** HW-5 requires a per-microphone tap test |
| default 43 mm geometry is correct | **Avoided.** HW-6 requires the drawing or callipers; all six pair distances |
| mapping inferrable without physical test | **Avoided.** HW-5 is gated on HW-4 returning A |

The existing text already warns "Do not force 4 channels merely because the
board has 4 microphones" and that two processed channels must not be treated
as "physical microphone 0 and 1".

⚠️ The **code** is weaker than this document: the SRP-PHAT gate accepts
`n_channels >= 4` as sufficient. That inconsistency is defect **D5** — the
document is right, the code needs hardening.

---

## Unchanged

HW-3 (boresight) and the overall structure. Every RESULT line still reads
`NOT PERFORMED`, which remains accurate: **no hardware test was run in this
session.**

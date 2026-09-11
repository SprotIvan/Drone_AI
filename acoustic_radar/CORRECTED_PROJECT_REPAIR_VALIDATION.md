# CORRECTIONS TO PROJECT_REPAIR_VALIDATION.md

This is an **errata record**, not a second copy of the report. The
corrections listed here have already been **applied in place** to
`PROJECT_REPAIR_VALIDATION.md`, so there is one source of truth; this file
exists so the changes are auditable.

Reason for each correction: the original statement was stronger than the
evidence in the code supports. Full analysis in
[FINAL_FORENSIC_REVIEW.md](FINAL_FORENSIC_REVIEW.md).

---

## 1. C1 — "fully resolved" → partially resolved, hardware-blocked

**Was:**
> The mapping is now determined by the station itself; HW-1 is reduced from a
> prerequisite to a one-time confirmation, and pinning the hint is a
> robustness recommendation rather than a precondition for starting.

**Why wrong:** `optical_assignment_score` assumes the telephoto view is the
exact centre of the wide view. Measured sensitivity shows it collapses at
**1° of pointing misalignment** (NCC 0.898 → 0.022), which is inside normal
mechanical mounting tolerance. On real hardware the check will usually
return INCONCLUSIVE and the station will fall back to the operator's hint.

**Now:** the *ambiguity handling* is `FIXED — CODE VERIFIED`; the *optical
classifier* is `ALGORITHM VERIFIED (synthetic only)` and unreliable in
practice; the *answer* — which socket holds which lens — is
`BLOCKED — HARDWARE REQUIRED` (HW-1). Setting `camera_role_id_hint` should
be treated as required.

**Not withdrawn:** the classifier never produced a wrong answer in any
tested condition. It degrades to INCONCLUSIVE, which is fail-closed.

---

## 2. N1 — "fully resolved" → mathematics right, reporting flag defective

**Was:** `STATUS: FIXED — CODE VERIFIED / MATH VERIFIED`.

**Why wrong:** the mathematics is correct and was independently re-derived
(TELE 9.442°, WIDE 37.975°, tolerance 20°). But
`agreement_is_discriminating` is a *camera-geometry* property published as
though it described a particular check. With an off-screen bearing the gate
**does** reject, and the flag still reports `False` — measured:

```
bearing 190.0 -> agreement 40.0 deg, exceeds tolerance? True   (rejected)
TELE discriminating flag: False
```

**Now:** mathematics `MATH VERIFIED`; the flag is recorded as defect **D2**,
pending approval for a two-line fix.

---

## 3. H5 — wording implied the gate proves channels are raw

**Was:**
> the channels are raw microphones (≥4 channels, or `mic_channels` explicitly
> configured)

**Why wrong:** neither condition establishes rawness. A device can expose
four *processed* channels, and `mic_channels: [0,1]` is an operator typing a
tuple into JSON — an assertion, not a measurement.

**Now:** described as "an operator attestation plus a channel count", with
rawness explicitly deferred to HW-4/HW-5. The gate remains correct **as
shipped** (both keys absent → SRP-PHAT disabled). Hardening proposed as
defect **D5**.

---

## 4. Withdrawn error bound — "a few percent"

**Was:** "Visual distances carry a few percent of error."

**Why wrong:** nothing measures, derives or bounds this. It assumes the lens
is within a few percent of nominal, the projection is distortion-free and
the mount is ideal — none of which is checked.

**Now:** "The error in visual distances is **UNKNOWN** until HW-2." The same
claim still exists **in the product**, printed at runtime — that is defect
**D1**, the highest-priority pending change.

---

## 5. Camera final table — Stable ID column

**Was:** `device-tree Id, **resolved at run time**` / status `OK`.

**Now:** `device-tree Id — **operator-supplied, HW-1**` / status
`blocked on HW-1`, with the reason stated inline. The literal `Id` strings
are still deliberately absent: they belong to your Pi and inventing one
would be fabricating a device identity.

---

## 6. FINAL VERDICT

**Was:** `READY FOR HARDWARE VALIDATION`.

**Now:** `NOT READY — CODE ISSUES REMAIN`, with the five defects tabulated
and the route back to READY spelled out.

---

## Unchanged — verified correct on re-inspection

C2 arithmetic (re-derived independently), C3, C4, H3, H6/M4, M1, M2, M3, the
N1 mathematics, the N2 cautious conclusion, and every "CHECKED AND NOT A
PROBLEM" entry. The test baseline (303/305 → 382/382) is accurate.

⚠️ Worth noting: **the suite was green at 382/382 throughout, and did not
catch any of these.** Two overclaims and a wrong hardware procedure lived
entirely in prose and in an unasserted reporting field. A passing suite is
not evidence that a report is true.

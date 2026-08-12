# Unified Drone Detection Station — Integration Report

Acoustic early warning + visual confirmation, merged into one application
with one window.

```bash
python main.py
```

---

## 0. The headline finding, first

Before anything else, one number changed the design of this integration.

**The camera's useful drone-detection range is about 20 m, not 200 m.**

It is not a guess. It follows from the optics already measured in this
repository:

```
box_width_px = focal_px × target_width_m / distance_m

FAR camera (IMX477): focal 1274 px  (two independent calibrations,
                                     5.0 m and 1.5 m, agree to 2.8%)
drone width:         0.25 m
smallest box YOLO can resolve reliably: ~16 px  [NEEDS FIELD VALIDATION]

→ distance = 1274 × 0.25 / 16 ≈ 20 m
```

At 200 m that same drone spans **1.6 pixels**. No detector finds it. The
acoustic sensor, meanwhile, is specified out to 500 m.

So the two sensors do not overlap the way the brief assumed — the
microphone owns the target for the overwhelming majority of its approach,
and the camera only ever gets the last ~20 m. The state machine is built
around that reality, and the range gate is *computed* from the optics
(`StationConfig.derive_visual_range_m`) rather than hardcoded, so it
re-derives itself the moment anyone re-calibrates a lens.

A second, more urgent finding is in §5 (bug **A3**): the acoustic distance
calibration currently in the repo was **hand-written, not measured**, so
every metre the acoustic sensor displays today is arbitrary. The system now
detects this and says so at startup.

---

## 1. Architecture

```
┌───────────────┐                                  ┌────────────────┐
│ MICROPHONE    │                                  │ 2× CSI CAMERA  │
│ ReSpeaker     │                                  │ IMX477 / IMX708│
└───────┬───────┘                                  └───────┬────────┘
        │ 0.5 s blocks                                     │ frames
        ▼                                                  ▼
┌───────────────────────────┐              ┌──────────────────────────┐
│ acoustic_worker.py        │              │ camera_worker.py         │
│  wraps radar.RadarEngine  │              │  wraps CameraManager,    │
│  (features → ONNX → FSM,  │              │  HailoInference,         │
│   doa.py, ranging.py —    │              │  AdvancedADASTracker     │
│   all UNCHANGED)          │              │  + CameraSwitchPolicy    │
└───────────┬───────────────┘              └───────────┬──────────────┘
            │ AcousticObservation                      │ VisualObservation
            │ (immutable, timestamped)                 │ (immutable, timestamped)
            ▼                                          ▼
        ┌───────────────── LatestValue ─────────────────┐
        │      one-slot mailbox, lock-free for readers  │
        └───────────────────────┬───────────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │ sensor_fusion.py         │
                  │  • freshness / staleness │
                  │  • range gate + hysteresis│
                  │  • priority state machine│
                  │  • derived closing speed │
                  │  → FusedTarget           │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │ hud.py + radar_overlay.py│
                  │  ONE window:             │
                  │  camera + radar overlay  │
                  └──────────────────────────┘
```

The two workers never touch each other's state. They publish immutable
snapshots; the fusion layer is the only place that combines them.

---

## 2. Files changed

| File | Change | Lines |
|---|---|---|
| `doa.py` | Bounded the `xvf_host.py` search (bug **D1**) | +64 / −5 |
| `camera_manager.py` | Lazy `picamera2` import; `get_frame()` hardened + failover; switch-order fix | ~+120 |
| `TWO_CAMERAS_FIXED.py` | Lazy Hailo import; `predict_with_scores()`; detector score plumbed to tracks | ~+110 |
| `.gitignore` | Ignore `logs/`, `detections/`, previews, `fusion_config.json` | +6 |

Every change to the two camera files is **additive and default-preserving**:
existing call signatures still work, and the original `main()` in
`TWO_CAMERAS_FIXED.py` runs exactly as before.

## 3. Files created

| File | Lines | Purpose |
|---|---|---|
| `main.py` | 387 | Single entry point, window, keyboard, shutdown |
| `fusion_config.py` | 573 | All tunables, each tagged DERIVED / MEASURED / CALIBRATE / POLICY |
| `target_state.py` | 468 | Observations, freshness, derived kinematics, `LatestValue` |
| `sensor_fusion.py` | 726 | Priority state machine, range gate, `FusedTarget` |
| `camera_cue.py` | 296 | Acoustic bearing → camera pixel (refuses without calibration) |
| `radar_overlay.py` | 481 | Radar widget, cached static artwork |
| `hud.py` | 633 | Unified window composition |
| `acoustic_worker.py` | 417 | Audio thread + supervision |
| `camera_worker.py` | 651 | Camera thread + `CameraSwitchPolicy` + supervision |
| `station_logging.py` | 208 | Levels, rotation, rate limiting, change-triggered events |
| `test_integration.py` | 812 | 83 automated checks (all ten required scenarios) |

## 4. Files intentionally untouched

Everything that actually detects drones was left alone, because it works
and because the README documents hard-won fixes in each:

`features.py` (the single train/inference front-end), `model.py`,
`ranging.py`, `radar.py` (`RadarEngine` is *used*, not modified),
`audio_io.py`, `calibration.py`, `calibrate.py`, `train.py`, `dataset.py`,
`mixer.py`, `download_dads.py`, `telegram_alert.py`, `audio_logger.py`.

`radar_gui.py` is also untouched and still runs standalone — but it is now
redundant, since its radar is inside the main window. Keep it as a fallback
until the unified UI is proven in the field, then retire it.

---

## 5. Bugs found

### D1 — Startup stalled for minutes on an unbounded directory scan (CRITICAL)

- **Problem** — The acoustic subsystem took minutes to start. Measured on
  this machine: `find_xvf_host()` had **not returned after 120 seconds**.
- **Cause** — `doa.py` ended with `home.glob("**/python_control/xvf_host.py")`,
  a fully recursive walk of the entire home directory (and `/home`),
  traversing caches, `node_modules`, and any mounted network share. It ran
  inside `DOAProvider.__init__`, i.e. on the path that must open the
  microphone.
- **Fix** — Replaced with a breadth-first walk that checks a deadline at
  every directory, skips known-heavy folders, and caps depth. `XVF_HOST_PATH`
  and the explicit candidate paths are unchanged and still take priority.
- **Impact** — Search capped at **2.0 s**; a genuine installation in the home
  directory is found in **0.00 s**. This affected `radar.py` and
  `radar_gui.py` too — it is fixed for them as well.

### A3 — The acoustic range calibration was never measured (CRITICAL, unfixed by design)

- **Problem** — The system displays acoustic distances in metres that are
  not based on any measurement.
- **Cause** — `radar_calibration.json` contains `range_ref_distance_m: 10.0`,
  `range_ref_level_dbfs: -35.0`, `noise_floor_dbfs: -60.0` — all perfectly
  round — and **lacks `range_spreading_db`**, which `calibrate.py cmd_range`
  *always* writes alongside the other two. The file was hand-edited, not
  produced by the calibration tool.
- **Fix** — Cannot be fixed in software; it requires a drone at a known
  distance. Instead the station now detects this signature at startup
  (reading the raw file, since `calibration.load()` merges a default that
  would mask it) and warns:
  *"every distance in metres shown by the acoustic sensor is therefore
  ARBITRARY."*
- **Impact** — This silently reintroduced exactly the failure the project's
  README says it eliminated (`dist = 6.0 / peak_volume` producing confident,
  meaningless metres). **Run `python calibrate.py range` before trusting any
  acoustic distance.**

### C1 — A camera hiccup killed the whole application

- **Problem** — Any capture error, or a closed camera, crashed the frame loop.
- **Cause** — `CameraManager.get_frame()` was one line:
  `return self.picams[self.active_camera].capture_array()`. `KeyError` if the
  active camera was not in the dict (reachable: `_close_camera()` pops it),
  and any exception from `capture_array()` propagated.
- **Fix** — Returns `None` on any failure (which every existing caller already
  handles), rate-limits the warning, counts consecutive failures, and fails
  over to the other camera after 15 of them.
- **Impact** — A dead camera now degrades to "CAMERA OFFLINE" with the
  acoustic radar still live, instead of terminating the process.

### C2 — A missing camera was misreported as a debounce rejection

- **Problem** — Switching to a camera that never opened logged nothing useful.
- **Cause** — `_switch()` tested the 0.5 s debounce *before* testing whether
  the target camera existed, so a permanently-absent camera was
  indistinguishable from "asked again too soon".
- **Fix** — Availability is checked first, with a rate-limited message.
- **Impact** — Diagnostic only, but it makes a one-camera setup obvious.

### X1 — Camera switching oscillated at the threshold

- **Problem** — A drone hovering near the switch distance could make the
  system flip cameras repeatedly.
- **Cause** — The original loop switched on a **single frame's** distance
  estimate. The 1.5 m / 2.0 m hysteresis band handles small jitter, but a
  drone mid-band with realistic bounding-box jitter produces individual
  frames on both sides. `CameraManager`'s 0.5 s debounce rate-limited the
  symptom without removing the cause.
- **Fix** — `CameraSwitchPolicy` requires the condition to hold for
  `confirm_frames` (default 5 ≈ 0.17 s at 30 fps) consecutive frames, on top
  of the unchanged distance hysteresis.
- **Impact** — Measured over 300 frames of a drone hovering at 1.75 m with
  σ = 0.35 m jitter: **72 switches → 0**. A genuine approach still switches
  exactly once in and once out.

### V1 — The real YOLO confidence was computed and thrown away

- **Problem** — No true detector confidence was available for the UI. The
  only per-target number was `quality_score`, a covariance-derived *track
  health* measure — it read **0.012** for a perfectly tracked object in test,
  so displaying it as "YOLO confidence" would have been a fabricated value.
- **Cause** — `HailoInference.predict()` ran NMS, kept the surviving indices,
  and returned boxes only, discarding `merged_s`.
- **Fix** — Added `predict_with_scores()`; `predict()` delegates to it and is
  byte-for-byte compatible. Scores flow through `process_frame(...)` →
  `_associate(...)` → `track.last_detection_score`, all via optional
  parameters defaulting to `None`.
- **Impact** — The UI shows a genuine detector confidence, and shows
  `COASTING` when a track is running on prediction with no fresh detection.

### X2 — No failure isolation between subsystems

- **Problem** — Importing the camera code on a machine without the Hailo
  runtime or `picamera2` raised `ImportError`, which would have taken the
  acoustic subsystem down with it.
- **Cause** — Both were top-level imports.
- **Fix** — Both are imported lazily at construction, with `hailo_available()`
  / `picamera2_available()` helpers. On a working Pi nothing changes.
- **Impact** — Required for the graceful-degradation behaviour; also what
  made it possible to test the tracker and fusion off-target.

### A1 — A blocking WAV write inside the audio loop

- **Problem** — Risk of audio buffer overrun (permanently lost audio) on
  every alarm.
- **Cause** — `radar_gui.py`'s audio thread called `AudioLogger.feed()`,
  which performs a synchronous 10 s WAV write **while holding a lock**, inside
  the loop that must return within 0.5 s.
- **Fix** — Partially mitigated: the call is wrapped so a failure cannot stop
  detection, and the loop now measures and warns when processing exceeds 90%
  of its budget. **The write itself is still synchronous** — see §15.
- **Impact** — Measured budget usage is currently 1–3%, so the write is the
  only realistic overrun source; it is now visible when it happens.

### A2 — `overflow` was silently dropped when copying status

- **Problem** — Audio buffer overruns were invisible in the GUI.
- **Cause** — `radar_gui.SharedState.set()` copied `RadarStatus` field by
  field (correctly, since the engine mutates one instance in place) but
  omitted `overflow`.
- **Fix** — `AcousticObservation` carries it, and overruns are logged
  rate-limited and summarised at shutdown.

### F1 — "Distance unknown" was treated as "target out of range" *(my bug, caught in test)*

- **Problem** — A target at 200 m armed visual acquisition against a camera
  with a ~20 m range.
- **Cause** — My gate tested `_in_range_streak == 0`, which is true both when
  the distance is *unmeasurable* and when it is *measured and too far*.
- **Fix** — The gate is now three-valued: `True` / `False` / `None`, and only
  a genuine `None` falls through to the "acquire anyway" policy.
- **Impact** — Regression-tested both ways.

### F2 — The target-lost timer measured from the wrong instant *(my bug, caught in review)*

- **Problem** — A target tracked for a long time would be declared lost
  instantly the moment contact dropped.
- **Cause** — The timeout compared against `_state_since` (time of last state
  change). After 99 s in `ACOUSTIC_TRACKING`, that already exceeds the 5 s
  timeout.
- **Fix** — Added `_last_contact_time`, updated whenever *either* sensor has
  usable contact; the timeout is measured from there.
- **Impact** — Regression-tested with a 99 s track.

### S1 — Hard-coded Telegram bot token committed to git (SECURITY)

- **Problem** — `radar_gui.py:48` contains a live bot token, and that file is
  tracked in git. **The token is in the repository history and must be
  treated as compromised.**
- **Fix** — No new file contains a token. The integration reads one from
  `ACOUSTIC_RADAR_TELEGRAM_TOKEN` or config, and Telegram is **off by
  default**.
- **Action required** — **Revoke the token via @BotFather.** Removing the line
  is not sufficient; it remains in history.

### P1 — The UI thread spun, starving the sensors (CRITICAL regression, now fixed)

- **Problem** — On the Pi the camera dropped from **37 fps to 19 fps**, the
  microphone stopped tracking entirely, and visual detection became
  intermittent. Reported from the field after the first integration.
- **Cause** — My UI loop rendered the *entire* HUD on every iteration, paced
  only by `cv2.waitKey(1)`. That is a spin, not a frame rate: it recomposited
  the same camera frame hundreds of times a second. `self._ui_dt` was computed
  and then never used, so the configured `max_ui_fps` had no effect at all.
  Compounding it, `display_scale` defaulted to 1.5 (a full-frame interpolation
  plus a 2.25× larger buffer per refresh) and `render()` copied the frame
  **twice** (into a fresh buffer, then into a freshly allocated canvas).
- **Fix** — Render only when a new camera frame has arrived, never faster than
  `max_ui_fps` (now 20 Hz), and spend the remaining time inside
  `cv2.waitKey(n)`, which blocks in C with the GIL released and therefore hands
  the CPU to the sensor threads. `display_scale` now defaults to 1.0, the
  canvas is allocated once and reused, and the frame is written exactly once.
- **Impact** — Measured: HUD render **7.89 ms → 1.12 ms**; UI CPU demand
  **+2.52 cores → +0.18 cores** (14× less). The camera holds full rate with the
  UI running.

### P2 — The displayed FPS was the wrong quantity

- **Problem** — The number on screen was not comparable to the original's.
- **Cause** — I displayed `_ui_fps`, the UI refresh rate. The original
  displayed the camera loop rate.
- **Fix** — The headline figure is the camera pipeline rate again; the UI rate
  is shown small and labelled `ui`.
- **Impact** — Because the display is now deliberately capped at 20 Hz, showing
  it as "FPS" would have looked like a permanent regression even when the
  camera was running at full speed.

### P3 — The fusion layer re-gated the acoustic engine's own verdict

- **Problem** — Acoustic targets could stay stuck in `ACOUSTIC_DETECTED` and
  never hand over to the camera.
- **Cause** — I added a fusion-level confidence gate defaulting to the engine's
  trained threshold (0.775). That sounds harmless and is not:
  `radar.Detector` uses 0.775 to **enter** a track but deliberately **holds**
  it at `threshold × HOLD_FACTOR` ≈ 0.54. An engine legitimately in ALARM
  routinely reports p between 0.54 and 0.775 — and my gate rejected exactly
  those.
- **Fix** — The gate is disabled by default; the engine's verdict is taken
  as-is, exactly as `radar_gui.py` always did.
- **Impact** — Regression-tested: an engine in ALARM at p = 0.60 now promotes
  to `ACOUSTIC_TRACKING` as it should.

### Also noted, not changed

- **`radar_gui.py` shutdown race** — sets `shared.running = False` then calls
  `engine.close()` and `sys.exit()` without joining the audio thread, which
  may still be inside `process_block()`. Not fixed because `radar_gui.py` is
  untouched; `main.py` joins both workers with a timeout.
- **Channel order** — picamera2's `"RGB888"` yields **B,G,R** byte order in
  numpy. The original code feeds the raw array to the detector and displays a
  swapped copy. Whether that matches the model's training order cannot be
  determined from this repository. Default behaviour is **preserved exactly**;
  `visual.swap_detector_channels` allows an A/B test. Worth testing — if it is
  wrong, fixing it is free detection accuracy.
- **NEAR camera focal length is unresolved** — the source comments leave two
  candidates, 501.7 and 1003.4, differing by exactly 2×. Any distance shown on
  the NEAR camera inherits that.

---

## 6. Sensor priority

| System state | Primary | Why |
|---|---|---|
| `SEARCHING` | none | no target |
| `ACOUSTIC_DETECTED` | **MICROPHONE** | only sensor with contact |
| `ACOUSTIC_TRACKING` | **MICROPHONE** | target beyond camera range |
| `VISUAL_ACQUISITION` | **MICROPHONE** | camera hunting, not yet confirmed |
| `VISUAL_TRACKING` | **CAMERA** | visually confirmed — bearing/position/identity from vision |
| `VISUAL_LOST_ACOUSTIC_TRACKING` | **MICROPHONE** | fallback for reacquisition |
| `TARGET_LOST` | none | both timed out |

The single source of truth is `_PRIORITY_OF_STATE` in `sensor_fusion.py`.
There is no priority logic anywhere else.

**The microphone is never stopped.** While the camera is primary it keeps
running and continues to supply the bearing, the range estimate, the derived
closing speed, and the re-cue if vision breaks.

---

## 7. State machine

```
                          ┌──────────────┐
             ┌───────────►│  SEARCHING   │◄────────────┐
             │            └──────┬───────┘             │ hold expires
             │   acoustic contact│                     │
             │                   ▼                     │
             │          ┌──────────────────┐    ┌──────┴──────┐
             │          │ ACOUSTIC_DETECTED│    │ TARGET_LOST │
             │          └────────┬─────────┘    └──────▲──────┘
             │  engine ALARM +   │                     │ no contact
             │  fusion confidence▼                     │ > 5 s
             │          ┌──────────────────┐           │
             │  ┌──────►│ ACOUSTIC_TRACKING├───────────┤
             │  │       └────────┬─────────┘           │
             │  │  in range for N│  ▲ left range       │
             │  │  updates       ▼  │                  │
             │  │       ┌──────────────────┐           │
             │  │       │VISUAL_ACQUISITION├───────────┤
             │  │       └────────┬─────────┘           │
             │  │  YOLO confirmed│                     │
             │  │  for N frames  ▼                     │
             │  │       ┌──────────────────┐           │
             │  │       │ VISUAL_TRACKING  ├───────────┤
             │  │       └────────┬─────────┘           │
             │  │   visual lost  ▼                     │
             │  │  ┌───────────────────────────┐       │
             │  └──┤VISUAL_LOST_ACOUSTIC_TRACK ├───────┘
             │     └─────────────┬─────────────┘
             │                   └──► VISUAL_ACQUISITION (re-cue)
             │
             └── visual-only: SEARCHING ──► VISUAL_TRACKING
                 (a silent but visible drone; configurable)
```

Every transition is implemented once, in `_next_state()`, and logged once
when it fires. Ordering rule: **visual confirmation is evaluated before
acoustic loss**, so a target the camera can see is never dropped because the
microphone lost it.

### Timeouts (all configurable, all derived from real subsystem rates)

| Parameter | Default | Derivation |
|---|---|---|
| `acoustic.stale_after_s` | 1.5 s | 3 missed 0.5 s updates |
| `acoustic.lost_after_s` | 3.0 s | matches `radar.MISS_TOLERANCE` = 6 blocks |
| `visual.stale_after_s` | 0.30 s | ~9 frames at 30 fps |
| `visual.lost_after_s` | 1.00 s | > tracker's 13-frame delete (~0.43 s) |
| `fusion.target_lost_after_s` | 5.0 s | 2 s margin over acoustic loss |
| `acoustic.stable_updates_for_acquisition` | 4 | 2.0 s of agreement |
| `visual.confirm_frames` | 3 | second gate after tracker's `min_hits` = 3 |

---

## 8. Camera switching

**Unchanged in substance, hardened in timing.**

```
distance available (pinhole, from the largest live box):
    FAR  → NEAR   when distance < 1.5 m   for 5 consecutive frames
    NEAR → FAR    when distance > 2.0 m   for 5 consecutive frames
    between 1.5 and 2.0 m: keep whichever camera is active

no focal calibration → pixel-height fallback (120 px / 80 px), same
                       confirmation gate

then CameraManager's own 0.5 s debounce, unchanged
```

Three independent guards: **distance hysteresis** (0.5 m band) → **temporal
confirmation** (5 frames, new) → **hard debounce** (0.5 s).

**Acoustic range deliberately does not drive this switch.** Its floor
(`MIN_DISTANCE_M = 2.0 m`) is already above the near threshold and its
accuracy is ±40–60%, so it cannot resolve a 1.5 m / 2.0 m decision even in
principle. The flag `switching.allow_acoustic_fallback` exists only to
record that the option was considered and rejected.

---

## 9. Radar UI

| Element | Meaning |
|---|---|
| **N marker** | Top of the dial. Bearings are in the acoustic frame *after* `doa_offset_deg`. If that offset is uncalibrated, N is the array's own zero, **not** magnetic north. |
| **Range rings** | Four rings, labelled in metres. Scale auto-selects from 25/50/100/250/500 m. |
| **Green sweep** | **Liveness indicator only.** The array listens in every direction at once; this is deliberately dim so it is not mistaken for a scanning beam. |
| **Dark wedge** | Bearing uncertainty — wider wedge = lower DOA confidence. |
| **Filled blip** | Target at (bearing, distance). Drawn **only when both are known**. |
| **Line through the blip** | The range interval (lo–hi), i.e. the honest uncertainty. |
| **Dashed radial** | Bearing known, **distance unknown** — drawn instead of a blip. The radar refuses to invent a radius. |
| **Full ring** | Distance known, **bearing unknown** — a ring at the right radius, labelled `BEARING N/A`. |
| **Hollow crossed marker** | Mirror-ambiguous solution (2-microphone geometry). |
| **Fading dots** | Target trail, 8 s. |
| **Header right** | Range scale, sensor status (`LISTENING` / `CONTACT` / `STALE` / `OFFLINE`), and closure (`CLOSING n m/s` / `OPENING` / `STEADY` / `CLOSURE UNKNOWN`). |

**In the camera view**, if — and only if — the mount geometry is calibrated,
the acoustic bearing is drawn as a **vertical band** spanning the full image
height. Never a point, never a box: the array measures azimuth only and has
no elevation, so a vertical position would be fabricated. Off-screen targets
get an edge arrow, not a clamped marker.

⚠️ A genuine limitation the UI states rather than hides: DOA accuracy is
roughly ±15° at best, while the FAR lens spans only 28°. When the resulting
band covers more than half the frame it is labelled `(COARSE)`. **The
acoustic bearing can tell you which way to look; it cannot aim this lens.**

---

## 10. Threading

Three threads, no queues, no shared mutable state.

| Thread | Rate | Owns |
|---|---|---|
| `acoustic` (daemon) | 2 Hz | PortAudio stream, `RadarEngine`, UART, WAV logger |
| `camera` (daemon) | ~30 Hz | `CameraManager`, Hailo device, IMM tracker, switch policy |
| main | ≤30 Hz | fusion, HUD, `cv2.imshow`/`waitKey`, keyboard |

Exchange is `LatestValue` — a one-slot mailbox holding an **immutable frozen
dataclass**. A lock is held only for a reference assignment. Not a queue,
deliberately: a queue would either grow unbounded when a consumer is slow
(and fall further behind reality) or block the producer. A live sensor
display wants the newest value and does not care about the ones it missed.

- Neither sensor can block the other or the UI.
- The UI never blocks a sensor — it renders the latest snapshot, including
  "nothing yet" and "offline".
- Both workers are **daemons**, so a device call wedged in the kernel can
  never prevent process exit.
- **Frame copies:** `capture_array()` allocates a fresh array per call, so
  the published frame is a *reference*, not a copy — the same bytes the
  original single-threaded program moved. The HUD makes the only copy, and
  with `display_scale ≠ 1.0` the resize already produces a new buffer, so
  there is no extra copy at all.

Verified: 882,459 publishes against 529,279 concurrent reads with zero torn
reads and clean thread termination.

---

## 11. Performance

⚠️ **All figures below were measured on the x86 development machine, not on
the Raspberry Pi 5.** They establish relative cost and headroom; absolute Pi
numbers must be measured on the Pi.

### Measured here

| Metric | Value |
|---|---|
| Acoustic processing per block | **6 ms mean, 15 ms max** of a 500 ms budget (~1–3%) |
| Fusion update | **0.16 ms** |
| Radar tile render | **0.42 ms** (cache defeated: 1.14 ms → cache saves 63%) |
| Full HUD compose, current defaults | **1.12 ms** (was 7.89 ms — see bug P1) |
| UI CPU demand, fixed loop | **+0.18 cores** (was +2.52 cores) |
| Fusion throughput | 3,300–10,000 updates/s |
| Memory over 20,000 updates | **+131 KB**, all histories provably capped |

### Simulated Pi 5 + Hailo-8L (`python simulate_pi.py --compare`)

Fake `picamera2` and `hailo_platform` modules with the real API surface, driving
the real worker, decode path, tracker, fusion and HUD:

| Scenario | Camera | UI | CPU |
|---|---|---|---|
| camera only, no UI | 37.2 fps | — | 0.45 cores |
| + UI, old spinning loop | 37.0 fps | 290.8 fps | **2.97 cores** |
| + UI, fixed loop | 37.2 fps | 18.8 fps | **0.62 cores** |

⚠️ The *FPS* column proves nothing about the Pi — this host has idle cores and
the fake sensor paces itself with `sleep()`, so even the broken loop kept 37 fps
here. The **CPU** column is what transfers: a 4-core Pi 5 that must also run the
ISP, the audio thread and the DOA subprocess cannot absorb an extra 2.5 cores,
and that is what took 37 fps to 19.

### NOT MEASURED

- **Camera FPS on the Pi** — needs the hardware. The sensor is hard-locked to
  30 fps (`max_fps=None`, deliberately: unlocking it shortens exposure and
  raises gain, hurting detection).
- **Hailo inference latency** — needs the Hailo device.
- **CPU usage** — not measured on either platform.
- **End-to-end detection latency** — not measured.

### Note on the HUD cost

The HUD runs on the **UI thread**, not the camera thread, so its cost bounds
UI frame rate, **not camera FPS or detection rate**. On the Pi, set
`ui.display_scale = 1.0` to halve it.

---

## 12. Calibration still required

Ordered by how much damage the missing value does.

1. **Acoustic range** — **DO THIS FIRST.** The current values are
   hand-written (bug **A3**); every acoustic metre is currently arbitrary.
   ```bash
   python calibrate.py noise    # 20 s of background, no drone
   python calibrate.py range    # hover at a known distance (two points is better)
   ```
2. **NEAR camera focal length** — unresolved between 501.7 and 1003.4 (exactly
   2× apart). Freeze switching with `f`, put the drone at 1.5 m, press `c`.
   The guard rejects boxes under 70 px, which is what produced the 2× error.
   Put the result in `fusion_config.json → geometry.camera_focal_px`.
3. **`min_detectable_box_px`** (default 16) — this sets the ~20 m range gate.
   Walk the drone out until detection becomes intermittent and read the box
   width. Currently an engineering assumption, not a measurement.
4. **`drone_real_width_m`** (default 0.25) — every visual distance scales
   linearly with it. A 2× error here is a 2× error everywhere.
5. **Camera boresight azimuth** — *the missing geometry.* Needed to map an
   acoustic bearing to an image column. Procedure: place a sound source that
   is also visible at a known azimuth, read the acoustic bearing, note where
   it appears in frame, and set `geometry.camera_boresight_deg[id]` to the
   azimuth the optical axis points along. Until then the cue is **disabled**
   and the UI says `BEARING->VIEW: NOT CALIBRATED`. (Given §9's ±15° vs 28°
   FOV, expect this to be a coarse cue at best on the FAR camera.)
6. **DOA offset / mirror** — `python calibrate.py doa` (two points).
   Until done, radar "N" is the array's zero, not north.
7. **`velocity_deadband_mps`** (default 1.5) — validate by logging a hovering
   drone and confirming it is not reported as approaching.

---

## 13. How to run

```bash
python main.py
```

That starts logging, config, microphone, acoustic ML, DOA, ranging, both
cameras, Hailo/YOLO26n, the tracker, camera switching, sensor fusion, and the
single unified window.

Useful flags:

```bash
python main.py --fullscreen
python main.py --log-level DEBUG
python main.py --no-camera          # acoustic only
python main.py --no-audio           # camera only
python main.py --headless           # everything but the window (SSH)
python test_integration.py          # 83 automated checks, no hardware needed
```

Optional `fusion_config.json` next to `main.py` overrides any default;
unknown keys are reported rather than silently ignored.

Keys while running: `q`/`ESC` quit · `h` help · `f` freeze switching ·
`c` focal calibration · `r` reset fusion.

## 14. Shutdown

Press **`q`** or **ESC**, close the window, or send **Ctrl+C** — all three run
the same orderly shutdown:

1. both workers are signalled to stop;
2. each is joined with a 3 s timeout (a device call stuck in the kernel is
   reported, not waited on forever — the threads are daemons);
3. Hailo vstreams and the activation context are released, then the cameras;
4. UART and the ONNX/DOA resources are closed;
5. session statistics are logged (frames, FPS, audio block times, overruns).

---

## 15. Known limitations

- **`AudioLogger` still writes WAV synchronously** on the audio thread
  (bug **A1**). Measured headroom is large (1–3% of budget), so it is not
  currently causing overruns, and the loop now warns if it ever does. Moving
  it to its own thread is the clean fix and is not done.
- **The `r` key resets fusion only** — it does not reset the visual tracker's
  track IDs.
- **Reconnection is retry-only** — a camera that dies is retried
  indefinitely; there is no re-enumeration of the CSI bus.
- **Single-target model** — the fusion layer tracks one target (the largest
  confirmed box). The IMM tracker handles multiple objects and all are drawn,
  but the state machine reasons about one.
- **Windows console** — some existing self-tests print emoji and fail on a
  cp1251 console. Pre-existing; use `PYTHONIOENCODING=utf-8`. The station's
  own logging is UTF-8-safe.

---

## 16. Verification performed

**Actually run, all passing:**

- `test_integration.py` — **83/83 checks**, covering all ten required
  scenarios plus regressions for D1, C1, F1, F2, V1, G1, R1 and a set of
  "never invent a value" assertions.
- Every new module's self-test, and the pre-existing self-tests for
  `features.py`, `doa.py`, `ranging.py`, `model.py` — confirming the audit
  changes broke nothing (`doa.py` SRP-PHAT still accurate to 0–10°).
- End-to-end `main.py --headless` on a machine with **no** camera and **no**
  Hailo: the real ONNX acoustic model loaded, a real microphone stream opened
  and processed, the camera subsystem failed gracefully, and shutdown was
  clean.
- HUD and radar rendered to PNG in every display state and visually
  inspected (`_hud_preview/`, `_radar_preview/`).
- Camera-switch oscillation measured quantitatively (72 → 0).
- Thread safety exercised under contention.

**Not verified — requires the Raspberry Pi 5 with cameras and Hailo:**

- Real Hailo inference and HEF decoding (HEAD_CONFIG tensor names *were*
  confirmed present in the HEF binary: `conv61/64/77/80/91/94`).
- Real Picamera2 capture, warmup, and physical camera switching.
- Real drone audio classification and end-to-end detection.
- Camera FPS, inference latency, CPU load on target.

I have not claimed any of the second group works.

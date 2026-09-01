# AUDIO / DOA FORENSIC AUDIT

**Об'єкт:** Drone Detection Station — виключно аудіо/DOA підсистема
**Дата аудиту:** 2026-08-31
**Режим:** ТІЛЬКИ ДОСЛІДЖЕННЯ. Жоден файл проєкту не змінено, жодного patch не створено.
**Робоча копія:** `c:\Users\user\Drone_AI\acoustic_radar`, гілка `main`, HEAD `59f94f9`

---

## 1. Scope

### Що аналізувалося

| Файл | Роль в аудіо/DOA ланцюгу |
|---|---|
| `audio_io.py` | вибір пристрою, кількість каналів, `mic_channels`, `to_mono` |
| `calibration.py` | `radar_calibration.json`, DEFAULTS, злиття ключів |
| `radar_calibration.json` | фактичні значення калібрування на цій машині |
| `doa.py` | `HardwareDOA` (USB/XVF3800), `ArrayDOA` (SRP-PHAT), `DOATracker`, `DOAProvider` |
| `bearing_frame.py` | канонічний кадр, `SourceConvention`, вихідні перетворення |
| `calibrate.py` | `cmd_doa`, `measure_raw_angle`, `cmd_check` |
| `radar.py` | `RadarEngine.process_block`, `Detector`, `_mic_block`, `UartSender` |
| `acoustic_worker.py` | аудіопотік, `AcousticObservation` |
| `target_state.py` | `AcousticObservation`, `AcousticTrackHistory`, трейл |
| `sensor_fusion.py` | `SensorFusion`, `FusedTarget`, `CueRole` |
| `radar_overlay.py` | рендер акустичного радара |
| `radar_gui.py` | застарілий Pygame-радар (той самий рендер напрямку) |
| `respeaker_led.py` | LED-кільце (споживач bearing + другий користувач `xvf_host.py`) |
| `camera_cue.py` | ТІЛЬКИ як споживач bearing (перевірка на подвійне перетворення) |
| `fusion_config.py`, `fusion_config.json` | geometry, led, acoustic, `check_bearing_frames` |
| `hud.py` | ТІЛЬКИ фрагмент `_draw_bearing_cue` — вивід акустичного напрямку |
| `features.py`, `ranging.py` | частота, смуга, рівень (як вхідні дані DOA-логіки) |
| `checker.md`, `INTEGRATION_REPORT.md`, `README.md` | попередні аудити як **джерело фактів про залізо** |

### Що НЕ аналізувалося (за вашою вимогою)

Камера, YOLO, Hailo, `camera_worker.py`, `camera_manager.py`, `TWO_CAMERAS_FIXED.py`,
`camera_cue` як система, MJPEG/FastAPI (`web_server.py` перевірено лише на предмет
того, що він **не** робить жодного перетворення кута — він транслює вже готовий
кадр), frontend, тренування моделі, детектор дрона як класифікатор.

### Обмеження цього аудиту

1. **Аудиторська машина — Windows без `numpy`, без `sounddevice`, без заліза.**
   Виконати `doa.py`, `radar.py`, `calibrate.py` тут неможливо.
2. **Виконано і перевірено виконанням:** `bearing_frame.py` (він чистий Python).
   Його самотест наведено нижче як ДОКАЗ координатних тверджень.
3. **Реплікація:** `select_azimuth`, `SourceConvention`, і фіт `calibrate.py doa`
   відтворені окремим скриптом у scratchpad з використанням
   `bearing_frame.circular_mean_deg` (математично тотожної реалізації
   циркулярного середнього) — бо `doa.circular_mean` потребує numpy.
   Це реплікація алгоритму, а не його прямий запуск; так і позначено.
4. **`radar_calibration.json` є в `.gitignore`.** Файл, який я бачу на цій
   машині, **може не збігатися** з тим, що зараз лежить на Raspberry Pi.
   Це критично для Finding #1 — див. TEST-01.

---

## 2. Repository Audio Architecture

Карта побудована **виключно** з коду. Кожен крок — файл, функція, вхід, вихід.

```
[ФІЗИЧНІ МІКРОФОНИ ReSpeaker XVF3800]
   │  (порядок і призначення каналів — НЕ ВСТАНОВЛЕНО, див. §4)
   ▼
─────────────────────────────────────────────────────────────────────
1. AUDIO CAPTURE
   audio_io.find_device(cfg["input_device"])          "ReSpeaker" за підрядком
   audio_io.resolve_input(cfg, 16000, prefer_raw=True)
        in : cfg, sample_rate
        out: InputConfig(device_index, channels, raw_array, mic_channels)
        логіка: max_ch>=4 → відкрити min(max_ch,6); інакше min(max_ch,2)
   audio_io.open_stream(inp, blocksize)
        sd.InputStream(samplerate=16000, dtype="float32",
                       channels=inp.channels, device=inp.device_index)
        падіння: inp.channels → 2 → 1
   acoustic_worker._loop():  block, overflowed = stream.read(block_samples)
        block: ndarray [samples, channels], hop 0.25 с (fusion_config
        acoustic.block_seconds = 0.25; radar.py standalone = 0.5)
─────────────────────────────────────────────────────────────────────
2. CHANNEL MAPPING
   audio_io.resolve_input → mic_channels
        cfg["mic_channels"] порожній  →  tuple(range(min(n_mics, channels)))
        тобто ПЕРШІ 4 КАНАЛИ. Це ПРИПУЩЕННЯ (§4).
   Розгалуження:
     a) radar.RadarEngine.process_block → audio_io.to_mono(block, mic_channels)
            → моно float32 → класифікатор (поза межами цього аудиту)
     b) radar.RadarEngine._mic_block(block) → block[:, mic_channels]
            → ТІЛЬКИ це йде в SRP-PHAT
─────────────────────────────────────────────────────────────────────
3. PREPROCESSING (лише в SRP-гілці)
   doa.ArrayDOA._bandpass(x)        rfft → обнулення поза 200–3500 Гц → irfft
        Прямокутне вікно у частотній області, БЕЗ вікна в часі.
─────────────────────────────────────────────────────────────────────
4. DOA / АЛГОРИТМ — ДВА НЕЗАЛЕЖНІ ДЖЕРЕЛА
   ┌── ДЖЕРЕЛО A (пріоритетне): АПАРАТНИЙ АЗИМУТ DSP ────────────────┐
   │ doa.HardwareDOA._loop()   власний потік, interval=0.35 с        │
   │   _query_once():                                                │
   │     subprocess.run([python, xvf_host.py, "AEC_AZIMUTH_VALUES"]) │
   │     doa.parse_azimuth_values(text)   regex → list[float]         │
   │     doa.azimuths_to_degrees(values)  ЕВРИСТИКА рад/град          │
   │     doa.select_azimuth(degrees, beam_index=-1)                   │
   │            → (angle, confidence, ambiguous)                      │
   │   кеш: self._reading + self._stamp (time.monotonic)              │
   │ doa.HardwareDOA.read(max_age=2.0)  ← аудіоцикл читає ТІЛЬКИ кеш  │
   │   out: DOAReading(angle_deg, confidence, source="usb",           │
   │                   ambiguous, raw=[4 азимути])                    │
   └──────────────────────────────────────────────────────────────────┘
   ┌── ДЖЕРЕЛО B (fallback, лише коли A не дав кута) ─────────────────┐
   │ doa.ArrayDOA.estimate(audio[samples, mic_channels])              │
   │   usable_pairs()   виключає вироджені пари (|corr| >= 0.999)     │
   │   _gcc_phat(x,y)   rfft·conj → PHAT (spec/|spec|) → irfft(n*8)   │
   │   scores[A] = Σ_pairs cc[expected_tau(pair, angle)]              │
   │        expected_tau = -((p_i - p_j)·u(θ))/c,  u=(cosθ, sinθ)     │
   │        angles = arange(0,360,5)  (n_angles=72)                   │
   │   best = argmax(scores);  ambiguous = (n_ch < 3)                 │
   │   out: DOAReading(angle, confidence, source="srp", ambiguous)    │
   └──────────────────────────────────────────────────────────────────┘
   Арбітраж: doa.DOAProvider.update(audio)
        reading = hardware.read() if hardware_ok else DOAReading(None)
        if not reading.ok and array_ok and audio is not None
                            and _array_disabled_reason is None:
            reading = array.estimate(audio)
─────────────────────────────────────────────────────────────────────
5. КАЛІБРУВАННЯ / КАНОНІЗАЦІЯ  (ЄДИНЕ місце перетворення конвенції)
   doa.DOAProvider._canonicalise(reading)
        conv = self.conventions[reading.source]
        conv = bearing_frame.source_convention(source, cfg)
             "usb": zero = cfg["doa_offset_deg"]
                    handedness = cfg["doa_handedness"]  (None → UNKNOWN)
             "srp": zero = cfg["srp_zero_deg"] (None → 0, zero_measured=False)
                    handedness = COUNTER_CLOCKWISE (доведено з коду)
        SourceConvention.to_canonical(raw):
             CCW  → wrap360(zero - raw)
             інакше (вкл. UNKNOWN→CW) → wrap360(zero + raw)
        out: DOAReading у КАНОНІЧНОМУ кадрі + calibrated + convention
─────────────────────────────────────────────────────────────────────
6. FILTERING / SMOOTHING / TRACKING
   doa.DOATracker.update(canonical_reading)
        HISTORY=8, OUTLIER_DEG=35, JUMP_CONFIRMATIONS=3, RELEASE_SEC=6.0
        angle = зважене циркулярне середнє останніх ≤8 прийнятих вимірів
   doa.DOAProvider._to_canonical() → bearing_frame.CanonicalBearing(
        deg, confidence, source, calibrated,
        ambiguous = tracker.ambiguous OR reading.ambiguous)
─────────────────────────────────────────────────────────────────────
7. BEARING У СИСТЕМІ
   radar.RadarEngine.process_block:
        st.angle_deg / angle_confidence / angle_source /
        angle_ambiguous / angle_calibrated / angle_reason
        (кут ОНОВЛЮЄТЬСЯ лише у станах TRACK і ALARM;
         у решті — doa.hold(), трек лише старіє)
   acoustic_worker._to_observation(status) →
        target_state.AcousticObservation(bearing_deg, bearing_confidence,
        bearing_source, bearing_ambiguous, bearing_calibrated, seq, timestamp)
   sensor_fusion.SensorFusion.update(acoustic, visual, ...) → FusedTarget
        (ОДИН акустичний obs, ОДИН трек; трейл — target_state.
         AcousticTrackHistory, вікно ui.radar_trail_s = 8.0 с)
─────────────────────────────────────────────────────────────────────
8. RADAR OUTPUT
   radar_overlay.RadarOverlay.render(target, dt)
        _draw_trail()   точки (bearing, distance) за останні 8 с
        _draw_target()  клин + бліп на bearing
                        + ГОСТ на (180 - bearing) % 360, якщо ambiguous
        _polar_to_xy(cx, cy, b, r) = (cx + r·cos(b-90°), cy + r·sin(b-90°))
   → hud.render() → один кадр → cv2.imshow і web_server.FrameBus.publish()
   (web_server НЕ виконує жодного перетворення кута — це доведено читанням:
    він приймає готовий BGR-кадр і кодує JPEG)
   Паралельні споживачі того самого канонічного кута:
        respeaker_led.frame_for_target → bearing_to_led_index (LED-сектор)
        camera_cue.BearingProjector.project (смуга на кадрі)
        radar.UartSender.send  →  "ANGLE:%03d\n"
```

---

## 3. Hardware and Audio Input Facts

### FACT (підтверджено кодом/файлами репозиторію)

| Факт | Доказ |
|---|---|
| Робоча частота дискретизації 16 000 Гц | `features.py:36` `SAMPLE_RATE = 16_000`; `radar.py:776` передає її у `resolve_input`; `model_config.json` `sample_rate: 16000` |
| Формат — float32 | `audio_io.open_stream`, `dtype="float32"` |
| Пристрій обирається за підрядком імені | `radar_calibration.json` → `"input_device": "ReSpeaker"`; `audio_io.find_device` |
| Кількість каналів визначається В RUNTIME, не зашита | `audio_io.resolve_input:111` читає `max_input_channels` з драйвера |
| Якщо каналів ≥4 — відкривається до 6 | `audio_io.py:118` |
| Апаратний азимут береться командою `AEC_AZIMUTH_VALUES` через `xvf_host.py` | `doa.py:448` |
| Код очікує **4 значення** азимута (променів) | `doa.py:26-30` коментар; `select_azimuth` кластеризує список |
| Смуга SRP-PHAT: 200–3500 Гц | `doa.ArrayDOA.__init__`, `band=(200.0, 3500.0)` |
| Смуга детектора/рівня: 80–4000 Гц | `features.DRONE_BAND_HZ` |
| Аудіо-hop станції: 0.25 с; standalone `radar.py`: 0.5 с | `fusion_config.AcousticConfig.block_seconds = 0.25`; `radar.BLOCK_SEC = 0.5` |
| Вікно рішення: 2.0 с | `features.WINDOW_SEC` |
| Період опитування USB DOA: 0.35 с + час субпроцесу | `doa.HardwareDOA.__init__(interval=0.35)` |
| Максимальний вік кешу азимута: 2.0 с | `doa.HardwareDOA.read(max_age=2.0)` |

### UNVERIFIED (жодного доказу в репозиторії — і я НЕ вигадую)

| Питання | Статус |
|---|---|
| Скільки каналів XVF3800 реально віддає на вашому Pi | **UNVERIFIED.** Визначається в runtime і НІДЕ не збережено. Немає жодного лог-файлу з `inp.describe()` у репозиторії (`logs/station.log` — див. TEST-02). |
| Який фізичний мікрофон відповідає кожному каналу | **UNVERIFIED.** `audio_io.py:42-55` сам це декларує як «PHYSICAL HARDWARE VERIFICATION REQUIRED». |
| Чи 6-канальний потік = «4 мікрофони + 2 опорні» | **UNVERIFIED.** `audio_io.py:117` — це коментар-припущення, і код це визнає у рядках 124-127. |
| Реальна геометрія масиву (відстані, порядок, форма) | **UNVERIFIED.** `calibration.py:44-47` прямо каже «Це ПРИПУЩЕННЯ про плату, а не вимір». |
| Семантика `AEC_AZIMUTH_VALUES` (чи це справді 4 азимути 4 променів) | **UNVERIFIED.** Ніде в репозиторії немає документації прошивки. Уся логіка `select_azimuth` побудована на цьому припущенні. |
| Одиниці, які віддає прошивка (радіани чи градуси) | **UNVERIFIED.** `azimuths_to_degrees` вгадує це евристикою по діапазону. |
| Нуль і напрямок обертання азимута XVF3800 | **UNVERIFIED у коді.** `bearing_frame.py:71-74` каже це прямо. Але `checker.md:527-529` фіксує ваш власний 5-точковий вимір → CCW. Див. Finding #1. |
| Чи виконує DSP власний beamforming до того, як віддає канали | **UNVERIFIED.** Код лише ДЕТЕКТУЄ це постфактум (`ArrayDOA.usable_pairs`: якщо всі пари вироджені — це оброблене стерео). |

**Я НЕ можу підтвердити жодного специфічного факту про XVF3800 з зовнішньої
документації в межах цього аудиту** — усе, що нижче, спирається на код
репозиторію та на попередній аудит `checker.md`, який сам посилається на ваші
вимірювання на Pi.

---

## 4. Microphone Channel Mapping Investigation

### Фактичний ланцюг

```
sd.InputStream(channels=N)  →  block [samples, N]
   ↓  audio_io.resolve_input
mic_channels = cfg["mic_channels"] або tuple(range(min(4, N)))
   ↓  radar.RadarEngine._mic_block
block[:, mic_channels]   →  ArrayDOA.estimate
   ↓  ArrayDOA
канал 0 підблоку ↔ mic_positions_m[0]
канал 1 підблоку ↔ mic_positions_m[1]  ... і т.д. ПО ПОРЯДКУ
```

### Що встановлено

**FACT.** `radar_calibration.json` **не містить** ключа `mic_channels`.
`calibration.load()` підставляє `DEFAULTS["mic_channels"] = []` →
`audio_io.py:133` виконує `mic_channels = tuple(range(min(n_mics, channels)))`,
тобто **канали 0, 1, 2, 3**.

**FACT.** Відповідність «індекс каналу ↔ фізичний мікрофон» у коді
встановлюється **виключно порядком** у `mic_positions_m`. Немає жодного
механізму перевірки.

**FACT.** Будь-яка перестановка каналів ЗМІНЮЄ результат SRP-PHAT, бо
`expected_tau` рахується для пар `(i, j)` за координатами `mics[i] - mics[j]`.

### ЧИ МОЖЕ НЕПРАВИЛЬНИЙ CHANNEL ORDER ВИКЛИКАТИ LEFT/RIGHT MIRROR?

**Відповідь: ТАК — але ТІЛЬКИ на SRP-гілці, і не на будь-якій перестановці.**

Геометрія за замовчуванням (`calibration.py:48`), квадрат 43 мм:

```
mic 0 = (-0.0215, +0.0215)      mic 1 = (+0.0215, +0.0215)
mic 3 = (-0.0215, -0.0215)      mic 2 = (+0.0215, -0.0215)
```

- Перестановка **0↔1 і 3↔2** (обмін лівої та правої колонок) еквівалентна
  дзеркаленню геометрії відносно осі y, тобто `x → -x`. У кадрі SRP
  (`θ = atan2(y, x)`) це дає `θ → 180° - θ`. Це **дзеркало ВПЕРЕД/НАЗАД**
  у власному кадрі масиву.
- Перестановка **0↔3 і 1↔2** (обмін верхнього та нижнього рядів) дає
  `y → -y`, тобто `θ → -θ`. Після канонізації SRP (`canonical = zero - raw`)
  це стає `canonical → 2·zero - canonical` — при `srp_zero_deg = null` (тобто
  zero = 0) це **точне дзеркало LEFT↔RIGHT на екрані**.
- Циклічний зсув 0→1→2→3→0 = поворот геометрії на 90°, тобто ПОСТІЙНИЙ
  зсув 90°, а не дзеркало. Такий дефект **не** дає симптом «іноді ліво, іноді
  право» — він дає сталу помилку.

**АЛЕ:** ця гілка виконується лише тоді, коли `HardwareDOA` не дав кута
(`doa.py:847`). Якщо `xvf_host.py` знайдено і працює — SRP-PHAT **ніколи не
викликається**, і channel order на напрямок **не впливає взагалі**.

### Додаткова знахідка (не напрямок, але аудіо)

**FACT.** `audio_io.to_mono(block, channels)` усереднює лише `mic_channels`,
але **лише якщо** `len(valid) != block.shape[1]` (`audio_io.py:188`). Тобто
при 4-канальному пристрої з `mic_channels = (0,1,2,3)` умова хибна і
усереднюються всі канали — що тут еквівалентно і тому нешкідливо.
При 6-канальному — усереднюються тільки 0-3. Логіка коректна **за умови**,
що канали 0-3 справді мікрофони. Це і є неперевірене припущення.

**Статус розділу: NOT PROVEN — REQUIRES HARDWARE TEST.** Див. TEST-02, TEST-03.

---

## 5. Microphone Geometry Investigation

### Єдине джерело геометрії

`calibration.py:48-49`:
```python
"mic_positions_m": [[-0.0215, 0.0215], [0.0215, 0.0215],
                    [0.0215, -0.0215], [-0.0215, -0.0215]],
```
`radar_calibration.json` цього ключа **не містить** → використовується DEFAULT.

Використовується у двох місцях:
- `doa.DOAProvider.__init__:821` → `ArrayDOA(cfg.get("mic_positions_m"), sr)`
- `calibrate.py:345` → `ArrayDOA(cfg["mic_positions_m"], ...)`

### Перевірка властивостей

| Питання | Відповідь | Доказ |
|---|---|---|
| Одиниці справді метри? | ТАК, за конструкцією | `SPEED_OF_SOUND = 343.0` м/с ділиться на цю величину у `expected_tau`; 0.0215 м = 21.5 мм узгоджується з коментарем «квадрат 43 мм» |
| Origin у центрі? | ТАК | сума координат = 0 |
| X/Y переплутані? | **НЕВІДОМО** — квадрат симетричний, тому обмін X↔Y = дзеркало відносно діагоналі, тобто `θ → 90° - θ`. Код цього виявити не може | — |
| Порядок обходу | 0(верх-ліво) → 1(верх-право) → 2(низ-право) → 3(низ-ліво) = **ЗА годинниковою** у стандартній математичній системі (y вгору) | обчислено з координат |
| Чи відповідає реальній платі | **NOT PROVEN — REQUIRES HARDWARE TEST** | `calibration.py:44` сам це декларує |

### ЧИ МОЖЕ ГЕОМЕТРІЯ ДАТИ RIGHT → LEFT або FRONT → BACK?

**ТАК, але тільки на SRP-гілці.** Довільна перестановка/дзеркалення
`mic_positions_m` — це ортогональне перетворення площини; половина таких
перетворень має визначник −1, тобто є дзеркальними, і жодний
`srp_zero_deg` їх не виправить (доведення — `bearing_frame.py:31-58`,
відтворене нижче виконанням).

**Стан на цій установці:** `srp_zero_deg = null` → SRP-PHAT **у будь-якому разі
некалібрований** (`zero_measured=False`), і його кут повернутий на невідому
величину. Це підтверджено виконанням:

```
srp: zero=0.0 hand=Handedness.COUNTER_CLOCKWISE assumed=COUNTER_CLOCKWISE
     calibrated=False
     "SRP-PHAT: CCW (proven), but ZERO NOT MEASURED — direction is rotated
      by an unknown amount. REQUIRES PHYSICAL CALIBRATION"
```

### Розрахункові межі масиву (обчислено з DEFAULT-геометрії, c = 343 м/с)

| Пара | База | Макс. затримка | У семплах @16 кГц | Просторовий Найквіст |
|---|---|---|---|---|
| 0-1 | 43.0 мм | 125.4 мкс | 2.01 | ≤ 3988 Гц |
| 0-2 (діагональ) | 60.8 мм | 177.3 мкс | **2.84** | **≤ 2820 Гц** |
| 0-3 | 43.0 мм | 125.4 мкс | 2.01 | ≤ 3988 Гц |
| 1-2 | 43.0 мм | 125.4 мкс | 2.01 | ≤ 3988 Гц |
| 1-3 (діагональ) | 60.8 мм | 177.3 мкс | **2.84** | **≤ 2820 Гц** |
| 2-3 | 43.0 мм | 125.4 мкс | 2.01 | ≤ 3988 Гц |

**Смуга, яку реально використовує `ArrayDOA`: 200–3500 Гц.**
3500 > 2820 → **діагональні пари працюють ВИЩЕ свого просторового Найквіста.**
Це фізична передумова для grating lobes (побічних максимумів) у просторовому
спектрі. Див. §10-C і §11.

---

## 6. DOA Algorithm Investigation

### 6.1 Гілка USB (XVF3800 DSP) — фактично активна на вашій установці

**Крок 1.** `subprocess.run([sys.executable, xvf_host.py, "AEC_AZIMUTH_VALUES"])`
у власному потоці, `cwd = script_path.parent`, timeout 3.0 с.

**Крок 2. `parse_azimuth_values`** — regex по всіх числах у рядку, що містить
ключ. Формат-агностично. **Коректно.**

**Крок 3. `azimuths_to_degrees`** — ЕВРИСТИКА:
```python
if np.max(np.abs(arr)) <= 2.0 * math.pi + 0.2:   # ≤ 6.483
    arr = np.degrees(arr)
```
Рішення «радіани чи градуси» приймається **окремо для кожного зчитування**
по максимуму з чотирьох значень. Ризик описано у §9 Finding #6.

**Крок 4. `select_azimuth(degrees, beam_index=-1, cluster_deg=25.0)`**
Для кожного променя будується група з усіх променів у межах ±25°.
Береться найбільший розмір групи; групи з однаковими центрами злипаються;
якщо лишилось >1 різних груп → `ambiguous = True`, і вибирається група
за ключем `(spread, circular_mean)`.

**Реплікація (`bearing_frame.circular_mean_deg`, numpy недоступний):**

| Вхід (4 промені) | Обраний кут | conf | ambiguous |
|---|---|---|---|
| `[140, 142, 138, 141]` | 140.3° | 1.00 | False |
| `[300, 302, 60, 62]` | **61.0°** | 0.25 | **True** |
| `[60, 300, 62, 302]` (той самий, інший порядок) | **61.0°** | 0.25 | True |
| `[10, 12, 190, 192]` | 11.0° | 0.25 | True |
| `[350, 352, 170, 172]` | **171.0°** | 0.25 | True |
| `[300, 302, 301, 60]` | 301.0° | 0.75 | False |
| `[0, 90, 180, 270]` | 0.0° | 0.12 | True |

**Ключове спостереження:** порядок променів справді більше не впливає (це було
виправлено), **але при нічиї завжди перемагає група з МЕНШИМ циркулярним
середнім** — бо `spread` в обох групах однаковий (0.0…2.0°), і ключ падає на
другий елемент `circular_mean(g)`. Жодного акустичного підґрунтя віддавати
перевагу меншому азимуту немає. Див. §9 Finding #4.

**Крок 5.** `DOAReading(angle, confidence=conf, source="usb", raw=degrees, ambiguous=amb)`

### 6.2 Гілка SRP-PHAT — резервна

Математика перевірена **читанням**, не виконанням (numpy недоступний):

- Час приходу на мікрофон `k`: `t_k = -p_k·u/c` (ближчий до джерела чує раніше).
- `expected_tau[pair] = -((p_i - p_j)·u)/c = t_i - t_j` — **узгоджено**.
- GCC: `rfft(x_i) * conj(rfft(x_j))` → пік на лагу `t_i - t_j` — **узгоджено**.
- PHAT-нормалізація `spec /= |spec| + 1e-12` — стандартна.
- Інтерполяція: `irfft(spec, n*interp)` — numpy доповнює спектр нулями, це
  коректна sinc-інтерполяція; масштаб змінюється рівномірно і на `argmax` не
  впливає.
- `max_lag = ceil(2.84) + 2 = 5` семплів; `shift = 5·8 = 40`; максимальний
  `|idx| = 2.84·8 = 22.7 < 40` → **обрізання (`np.clip`) не спрацьовує**, вибірка
  завжди в межах. **Коректно.**
- `angles = np.arange(0, 360, 360 // 72)` = крок 5°.
  ⚠ Ділення цілочисельне: при `n_angles`, що не ділить 360, кількість
  напрямків мовчки зміниться. За замовчуванням безпечно.
- `ambiguous = n_ch < 3` — прапорець виставляється **тільки** за кількістю
  каналів, ніколи за формою спектра. Тобто справжня мультипікова
  неоднозначність SRP-PHAT **не детектується взагалі**.

**Висновок:** внутрішня математика SRP-PHAT самоузгоджена. Проблема не в ній,
а в (a) некаліброваному нулі, (b) неперевіреній геометрії/каналах,
(c) перевищенні просторового Найквіста діагональними парами.

### 6.3 Арбітраж і згладжування

`DOAProvider.update` — **канонізація виконується ДО згладжування** (`doa.py:876`).
Це правильно і виправляє відомий раніше дефект змішування кадрів. **Працює
коректно.**

`DOATracker` — див. §9 Finding #2 і Finding #3: тут два реальні дефекти.

---

## 7. Calibration Investigation

### Фактичний вміст `radar_calibration.json` на цій машині

```json
{
  "range_ref_distance_m": 3.0,
  "range_ref_level_dbfs": -22.0,
  "noise_floor_dbfs": -55.0,
  "doa_offset_deg": 180.0,
  "doa_invert": false,
  "input_device": "ReSpeaker"
}
```

Після `calibration.load()` (злиття з DEFAULTS) реально діють:

| Параметр | Значення | Де задається | Де використовується | Математичний вплив | Може дати LEFT/RIGHT? |
|---|---|---|---|---|---|
| `doa_offset_deg` | **180.0** | JSON | `bearing_frame.source_convention("usb")` → `SourceConvention.zero_deg` | `canonical = 180 ± raw` | НІ сам по собі (це поворот, не дзеркало) |
| `doa_handedness` | **ВІДСУТНІЙ → None** | DEFAULTS | там само → `Handedness.UNKNOWN` → `assumed_handedness = CLOCKWISE` | обирає `+raw` замість `−raw` | **ТАК — це і є дзеркало** |
| `doa_invert` | `false` | JSON | **НІДЕ в runtime.** Читається лише `calibration.describe()` (банер) і `respeaker_led.py:1168` (діагностичний режим `--bearing`) | **ЖОДНОГО** | НІ — параметр мертвий |
| `doa_beam_index` | `-1` (DEFAULT) | DEFAULTS | `HardwareDOA.beam_index` | `-1` → автовибір групи; `≥0` → жорстко цей промінь, conf=0.5, ambiguous=False | НІ, але `≥0` вимкнув би кластеризацію |
| `mic_positions_m` | DEFAULT 43 мм квадрат | DEFAULTS | `ArrayDOA` | вся геометрія SRP | ТАК, але лише на SRP-гілці |
| `mic_channels` | `[]` → «перші N» | DEFAULTS | `audio_io.resolve_input` | порядок каналів у SRP | ТАК, але лише на SRP-гілці |
| `srp_zero_deg` | `null` | DEFAULTS | `source_convention("srp")` | zero=0, `zero_measured=False` | НІ (це поворот) |
| `led_zero_offset_deg` | `0.0` | `fusion_config.json` | `bearing_to_led_index` | поворот індексу LED | стосується лише кільця |
| `camera_boresight_deg` | `{0:0.0, 1:0.0}` | `fusion_config.json` | `camera_cue` | зсув для камери | стосується лише камери |
| `boresight_calibrated_at_doa_offset_deg` | не задано | — | `check_bearing_frames` | **перевірка мовчить** | — |
| `boresight_calibrated_handedness` | не задано | — | `check_bearing_frames` | **перевірка мовчить** | — |

### Чи застосовується щось ДВІЧІ? Чи є подвійна інверсія?

**НІ. Перевірено вичерпно.**

- `doa.apply_orientation` / `unapply_orientation` існують, але
  **у runtime-шляху не викликаються жодного разу** (grep по всіх `.py`:
  єдині згадки — визначення в `doa.py`, коментарі, і рядок імпорту
  в `calibrate.py`, який їх не використовує у `cmd_doa`).
- `bearing_frame.to_radar_screen_deg` — тотожність (`wrap360`).
- `radar_overlay._draw_target` бере `acoustic.bearing_deg` напряму, без корекцій.
- `respeaker_led.bearing_to_led_index` бере канонічний кут і застосовує
  ТІЛЬКИ `led_zero_deg` / `clockwise` — власні параметри кільця.
- `camera_cue.project` застосовує ТІЛЬКИ `camera_boresight_deg`.
- `web_server` не торкається кутів взагалі.

**Кожен вихідний шар виконує рівно ОДНЕ власне перетворення. Double
transformation НЕ ЗНАЙДЕНО.** Це — те, що працює правильно.

### Головна проблема калібрування

`doa_offset_deg = 180.0` **без** `doa_handedness` — це рівно та конфігурація,
про яку `bearing_frame.py:266-280` пише: «`doa_invert` не може бути доказом
калібрування». Виконання підтверджує:

```
usb: zero=180.0 hand=Handedness.UNKNOWN assumed=Handedness.CLOCKWISE
     calibrated=False
     "XVF3800 USB: HANDEDNESS UNKNOWN — may be MIRRORED, assuming CW."
```

Тобто **система зараз ЗДОГАДУЄТЬСЯ про напрямок обертання**, і здогадка — CW.

---

## 8. Coordinate System Investigation

### Канонічний кадр (визначений у `bearing_frame.py`)

```
0°   = напрямок, куди ДИВИТЬСЯ УСТАНОВКА (front / north)   → ВГОРУ на радарі
90°  = EAST  = ПРАВОРУЧ
180° = SOUTH = НАЗАД
270° = WEST  = ЛІВОРУЧ
Зростання — ЗА ГОДИННИКОВОЮ СТРІЛКОЮ (згори)
Діапазон [0, 360)
```

### ДОКАЗ ВИКОНАННЯМ (запущено `python bearing_frame.py` на цій машині)

```
DETERMINISTIC CHAIN CHECK — one bearing in, three layers out
 bearing  expected   radar     LED  camera  agree
       0     FRONT   FRONT   FRONT   FRONT  YES
      90     RIGHT   RIGHT   RIGHT   RIGHT  YES
     180    BEHIND  BEHIND  BEHIND  BEHIND  YES
     270      LEFT    LEFT    LEFT    LEFT  YES

And with a MIRRORED ring, the check catches it:
      90     RIGHT   RIGHT    LEFT   RIGHT  NO
     270      LEFT    LEFT   RIGHT    LEFT  NO
```

Тобто **радар, LED і камера ОДНОСТАЙНІ**: якщо канонічний кут правильний,
270° буде ЗЛІВА на всіх трьох. Перевірка має «зуби» — вона ловить
дзеркальне кільце.

### Таблиця конвенцій усіх шарів

| Шар | Нуль | Додатній напрямок | Одиниці | Обгортання | Підстава |
|---|---|---|---|---|---|
| Фізичні мікрофони | **UNKNOWN** | **UNKNOWN** | ° | — | `mic_positions_m` — DEFAULT |
| SRP-PHAT вихід | вісь +x з `mic_positions_m` | **проти годинникової** | ° | 0..360 | ДОВЕДЕНО: `unit = (cosθ, sinθ)` = atan2 |
| XVF3800 USB | **UNKNOWN** | **UNKNOWN** | ° | 0..360 | прошивка Seeed, документації немає |
| Канонічний | фронт установки | **за годинниковою** | ° | 0..360 | ВИЗНАЧЕНО в `bearing_frame` |
| Екран радара | вгору | за годинниковою | ° | 0..360 | ДОВЕДЕНО: `(cos(b−90), sin(b−90))`, y вниз |
| LED-кільце | `led_zero_offset_deg` = 0.0 | `led_index_clockwise` | індекс | mod 12 | ПОТРЕБУЄ КАЛІБРУВАННЯ |
| Камера | `camera_boresight_deg` = 0.0 | вправо у кадрі | px | ±HFOV/2 | ДОВЕДЕНО: `x = W/2 + f·tan(rel)` |
| Екран OpenCV | верх-ліво | x вправо, y вниз | px | — | OpenCV |

### Арифметика кутів

Перевірено виконанням: `angular_distance(359, 1) = 2`,
`circular_mean_deg([350, 10]) = 0`, `wrap180`, `wrap360` централізовані у
`bearing_frame`; `camera_cue.wrap_signed_deg` — псевдонім `wrap180`;
`doa.circular_mean` округлює до 6 знаків, щоб прибрати похибку float.

**ЖОДНОГО дефекту обгортання, знаку, radians/degrees або modulo у
координатному ланцюгу НЕ ЗНАЙДЕНО.**

### ЧИ Є КОМПОНЕНТ, ЯКИЙ КАЖЕ «270° = RIGHT»?

**НІ.** Усі чотири споживачі (радар, LED, камера, UART) читають один і той
самий `AcousticObservation.bearing_deg` і трактують 270° як ЛІВО.
**Проблема НЕ у розбіжності між компонентами — вона в тому, ЯКЕ ЧИСЛО
потрапляє в `bearing_deg`.**

---

## 9. LEFT / RIGHT INVERSION Investigation

---

### Finding #1 — Напрямок обертання XVF3800 не записаний у конфігурації; система ЗДОГАДУЄТЬСЯ, і здогадка суперечить вашому власному вимірюванню

**Статус: CONFIRMED BUG (у конфігурації, яку я бачу) / HIGHLY LIKELY на Pi — потребує TEST-01**

#### Evidence from code

`radar_calibration.json` (ця машина) — ключа `doa_handedness` немає.

`bearing_frame.py:281-289`:
```python
explicit = cfg.get("doa_handedness")
zero = float(cfg.get("doa_offset_deg", 0.0))
if explicit is None:
    return SourceConvention(zero, Handedness.UNKNOWN, "XVF3800 USB",
                            zero_measured=False)
```

`bearing_frame.py:199-201`:
```python
if self.handedness is Handedness.UNKNOWN:
    return Handedness.CLOCKWISE      # ← ЗДОГАДКА
```

`bearing_frame.py:206-208`:
```python
if self.assumed_handedness is Handedness.COUNTER_CLOCKWISE:
    return wrap360(self.zero_deg - float(raw_deg))
return wrap360(self.zero_deg + float(raw_deg))
```

`checker.md:527-529` (попередній аудит, посилається на ВАШ вимір):
> «The user's own five-point calibration confirmed **CCW** for the XVF3800
> (mean error **7.9°** vs **70.9°** for CW) — but that result lives only on
> the Pi (BUG-022).»

`checker.md:863-864`:
> `doa_offset_deg` — MEASURED on the Pi (**353.7**) — Not in git
> `doa_handedness` — MEASURED on the Pi (**CCW**, residual 7.9°) — Not in git

`.gitignore` містить `radar_calibration.json` → файл у git не оновлюється.

#### Technical explanation

Якщо масив реально CCW, а система застосовує CW, то:
```
правильно:  canonical = zero − raw
фактично:   canonical = zero + raw
похибка   = 2 · raw
```
Похибка **нульова при raw = 0° і raw = 180°**, і **рівно 180° (повний обмін
ЛІВО↔ПРАВО) при raw = 90° і raw = 270°**.

Відтворено виконанням (zero = 180, як у файлі):

| raw | як налаштовано (CW) | якщо реально CCW | розбіжність | сторона (cfg) | сторона (CCW) |
|---|---|---|---|---|---|
| 0 | 180 | 180 | 0 | BEHIND | BEHIND |
| 30 | 210 | 150 | 60 | BEHIND | BEHIND |
| 60 | 240 | 120 | 120 | **LEFT** | **RIGHT** |
| **90** | **270** | **90** | **180** | **LEFT** | **RIGHT** |
| 120 | 300 | 60 | 120 | **LEFT** | **RIGHT** |
| 150 | 330 | 30 | 60 | FRONT | FRONT |
| 180 | 0 | 0 | 0 | FRONT | FRONT |
| 240 | 60 | 300 | 120 | **RIGHT** | **LEFT** |
| **270** | **90** | **270** | **180** | **RIGHT** | **LEFT** |
| 300 | 120 | 240 | 120 | **RIGHT** | **LEFT** |

#### Can explain the observed behaviour?

**YES — і це найточніший збіг із вашим описом з усіх знайдених причин.**

- «дрон справа, а показує зліва» — так, при raw ≈ 240–300°.
- «але не абсолютно постійно» — так: при raw ≈ 0° і 180° помилки НЕМАЄ взагалі,
  біля них вона мала. Симптом З'ЯВЛЯЄТЬСЯ І ЗНИКАЄ залежно від того, під яким
  азимутом стоїть дрон.
- Помилка **не залежить від відстані** напряму — але виглядатиме
  «відстань-залежною», якщо ближні перевірки ви робили перед станцією
  (raw ≈ 0/180, помилка 0), а дальні — збоку.

#### Confidence

**HIGH.** У конфігурації на цій машині це доведено виконанням. Єдина
невідома — чи має файл на Pi ключ `doa_handedness`.

#### How to prove or disprove

**TEST-01** (див. §17). Одна команда на Pi.

#### Missing evidence

Вміст `radar_calibration.json` на Raspberry Pi.

---

### Finding #2 — Одне апаратне зчитування азимута зараховується трекеру кілька разів, і 3 «підтвердження» стрибка можуть надійти з ОДНОГО виміру

**Статус: CONFIRMED BUG**

#### Evidence from code

`doa.HardwareDOA.__init__`: `interval = 0.35` — період опитування DSP.
`doa.HardwareDOA.read(max_age=2.0)` — повертає **кеш**, дійсний до 2.0 с.
`fusion_config.AcousticConfig.block_seconds = 0.25` — період аудіоблоку.
`radar.RadarEngine.process_block:596`: `self.doa.update(self._mic_block(block))`
— викликається **на кожному** аудіоблоці.
`doa.DOAProvider.update:845`: `reading = self.hardware.read()`.

`DOAReading` **не має номера послідовності**. `DOATracker` **не має жодної
дедуплікації**.

`doa.DOATracker`:
```python
JUMP_CONFIRMATIONS = 3
...
self._pending.append(angle)
if len(self._pending) >= self.JUMP_CONFIRMATIONS:
    recent = self._pending[-self.JUMP_CONFIRMATIONS:]
    centre = circular_mean(recent)
    if all(angular_diff(a, centre) <= self.OUTLIER_DEG for a in recent):
        self._history.clear(); self._pending.clear()
        self._accept(centre, reading.confidence, now)   # ← СТРИБОК
```

#### Technical explanation

Аудіоцикл — 4 Гц (hop 0.25 с). Опитування DSP — не частіше 2.9 Гц
(0.35 с), а реально повільніше: `doa.py:33-36` сам пише, що запуск
python-інтерпретатора коштує **200–500 мс**, тобто фактичний період
0.55–0.85 с ≈ 1.2–1.8 Гц.

Отже **кожне апаратне зчитування споживається 2–4 рази**.

`_pending` рахує **виклики**, а не окремі виміри. Тому:
- «3 послідовні підтвердження нового напрямку» реально досягаються
  **1–2 фізичними зчитуваннями**;
- у гіршому випадку (потік опитування пригальмував, кеш живе до 2.0 с =
  8 аудіоблоків) **одного-єдиного зчитування достатньо**, щоб трек
  стрибнув на протилежний бік;
- те саме псує `_history` (вікно 8): у ньому може бути лише 2-3 незалежні
  виміри, роздуті до 8 записів, і зважене циркулярне середнє більше не
  усереднює шум.

Захист від неоднозначних вимірів (`if reading.ambiguous: return`) тут **не
допомагає**: неоднозначні виміри блокуються, а от **одне НЕоднозначне, але
хибне зчитування** (наприклад, 3 з 4 променів дивляться на стіну) проходить і
перекидає трек.

#### Can explain the observed behaviour?

**YES, PARTIALLY** — пояснює саме «раптовий стрибок стрілки на протилежний
бік», а не сталу дзеркальність. Разом із Finding #1 і Finding #4 дає повну
картину «іноді ліво, іноді право».

#### Confidence

**HIGH** — механізм прямо видно з коду; невідома лише фактична частота
опитування на вашому Pi.

#### How to prove

**TEST-04** (лог унікальності азимутів).

---

### Finding #3 — Неоднозначний вимір ПРИЙМАЄТЬСЯ при ініціалізації треку, після чого хибний бік самопідтверджується

**Статус: CONFIRMED BUG**

#### Evidence from code

`doa.DOATracker.update`, порядок перевірок (рядки 729–760):
```python
if not reading.ok or reading.confidence <= 0.0: ... return
angle = float(reading.angle_deg)
self.ambiguous = reading.ambiguous        # (1) прапорець виставлено ЗАВЖДИ
...
if self.angle is None:
    self._accept(angle, reading.confidence, now)   # (2) БЕЗ перевірки ambiguous
    return
if angular_diff(angle, self.angle) <= self.OUTLIER_DEG:
    self._pending.clear()
    self._accept(angle, reading.confidence, now)   # (3) теж БЕЗ перевірки
    return
if reading.ambiguous:                              # (4) перевірка ЛИШЕ тут
    return
```

`select_azimuth` при нічиї повертає `confidence = 0.25` — це **> 0**, тому
рання перевірка `confidence <= 0.0` його не відсіює.

#### Technical explanation

Правило «нічия не рухає трек» реалізовано **тільки для викидів (>35°)**.
Наслідки:

1. **Захоплення цілі.** Перший вимір після тиші, навіть якщо це нічия
   2-проти-2, **приймається** і задає напрямок треку. Якщо `select_azimuth`
   обрав не той кластер (див. Finding #4), трек стартує з неправильного боку.
2. **Самопідтвердження.** Далі неоднозначні виміри **на тому самому (хибному)
   боці** проходять гілку (3) і **підживлюють** трек. Неоднозначні виміри
   на правильному боці — викиди — блокуються гілкою (4) і **не можуть
   перетягнути** напрямок назад.
3. Трек лишається на хибному боці, доки не з'явиться серія з ≥3
   **неоднозначних** вимірів на правильному боці.

Це **фіксатор (latch)** хибного напрямку.

#### Can explain the observed behaviour?

**YES.** Пояснює, чому напрямок «залипає» на неправильному боці і чому
симптом нестабільний: усе залежить від того, який саме кадр був першим
після появи дрона.

#### Confidence

**HIGH** — читається безпосередньо з порядку `if`-ів.

#### How to prove

**TEST-05**.

---

### Finding #4 — При нічиї 2-проти-2 завжди перемагає кластер із МЕНШИМ азимутом

**Статус: CONFIRMED BUG**

#### Evidence from code

`doa.select_azimuth:206-211`:
```python
ambiguous = len(distinct) > 1
if ambiguous:
    distinct.sort(key=lambda g: (_spread_deg(g), circular_mean(g)))
best_members = distinct[0]
```

`_spread_deg` для двох майже однакових променів — 0.0…2.0°. При симетричному
розкладі 2+2 обидва `spread` практично рівні, і ключ падає на другий елемент —
`circular_mean(g)`, тобто **сортування за величиною кута**.

#### Reproduction (реплікація алгоритму)

| Промені DSP | Обрано | Правильно було б | Помилка |
|---|---|---|---|
| `[300, 302, 60, 62]` | **61.0°** | 301.0° (якщо джерело там) | 240° |
| `[350, 352, 170, 172]` | **171.0°** | 351.0° | 180° — **точне дзеркало** |
| `[10, 12, 190, 192]` | 11.0° | 191.0° | 180° — **точне дзеркало** |

#### Technical explanation

Детермінізм — це добре (порядок променів більше не впливає, це виправлення
працює). Але **детермінований ≠ правильний**: правило «менший азимут
перемагає» не має жодного акустичного сенсу. Ані рівень, ані узгодженість,
ані історія треку в рішенні не беруть участі.

У випадках, де два кластери рознесені приблизно на 180°, це дає **точну
дзеркальну помилку** — рівно ваш симптом.

#### Can explain the observed behaviour?

**YES.** І пояснює «частіше на далеких дистанціях»: чим слабший прямий
сигнал, тим імовірніше, що DSP розділить промені між джерелом і відбиттям.

#### Confidence

**HIGH** — відтворено.

#### Missing evidence

Що саме означають 4 значення `AEC_AZIMUTH_VALUES` у прошивці. Якщо це не
«азимут кожного з 4 променів», то вся ця логіка не має підстав узагалі.

#### How to prove

**TEST-06** (записати сирі `raw` разом із реальним положенням дрона).

---

### Finding #5 — `doa_invert` — мертвий параметр, але банер станції його друкує

**Статус: CONFIRMED BUG (вводить оператора в оману; сам по собі напрямок не змінює)**

#### Evidence

Grep по всіх `.py`: `doa_invert` читається у
`calibration.describe()` (рядок 136 — друкує «дзеркально») і в
`respeaker_led.py:1168` (діагностичний режим `--bearing`). У
**runtime-шляху DOA він не читається жодного разу** —
`bearing_frame.source_convention` дивиться **тільки** на `doa_handedness`.

#### Technical explanation

Оператор, який побачить у `radar_calibration.json` рядок `"doa_invert": true`
і банер «кут: зсув +180°, дзеркально», обґрунтовано вважатиме, що дзеркалення
застосовано. Воно не застосовується. І навпаки: виставлення `doa_invert`
вручну **не виправить** дзеркальність.

#### Can explain LEFT/RIGHT?

**NO** безпосередньо — **YES опосередковано**: цей параметр маскує Finding #1,
бо створює враження, що дзеркалення налаштоване.

#### Confidence

**HIGH (CONFIRMED).**

---

### Finding #6 — `calibrate.py doa` з ОДНІЄЮ точкою завжди записує `CW` і при цьому позначає джерело як ПОВНІСТЮ калібровану

**Статус: CONFIRMED BUG**

#### Evidence from code

`calibrate.py:424-433`:
```python
best = None
for hand in (Handedness.CLOCKWISE, Handedness.COUNTER_CLOCKWISE):
    ...
    if best is None or residual < best[1]:   # ← СУВОРЕ <
        best = (conv, residual)
```
Із однією точкою обидві гіпотези дають residual **рівно 0.0**, тому суворе `<`
залишає ту, що перевірялась першою — **CLOCKWISE**.

`calibrate.py:458-462` записує `doa_handedness: "CW"`.

`bearing_frame.source_convention("usb")` бачить `doa_handedness` не-None →
`zero_measured=True`, `handedness=CW` → `calibrated == True`.

#### Reproduction

| справжній напрямок | вимір масиву | що записується | runtime.calibrated |
|---|---|---|---|
| 90° | 30° | `CW`, zero=60.0, residual=0.0 | **True** |
| 270° | 30° | `CW`, zero=240.0, residual=0.0 | **True** |
| 0° | 123° | `CW`, zero=237.0, residual=0.0 | **True** |

#### Technical explanation

Скрипт **друкує** попередження «ОДНА ТОЧКА НЕ ВИЗНАЧАЄ НАПРЯМОК ОБЕРТАННЯ»,
але **все одно записує** `doa_handedness`. Після цього:
- напис `UNCAL` на радарі та HUD **зникає**;
- рядок «BEARING UNVERIFIED — run calibrate.py doa» у `hud.py:565` **зникає**;
- `_warn_unverified` у `respeaker_led` **замовкає**;
- система стверджує, що конвенція виміряна, хоча половину її вгадано.

Якщо оператор колись зробив одноточкове калібрування, він отримав **той самий
дзеркальний дефект, що й у Finding #1, але БЕЗ жодного попередження**.

#### Can explain LEFT/RIGHT?

**YES** — тим самим механізмом, що й Finding #1, але прихованим.

#### Confidence

**HIGH (CONFIRMED, відтворено).**

---

### Finding #7 — Евристика «радіани чи градуси» приймається окремо на кожному зчитуванні

**Статус: POSSIBLE PROBLEM**

#### Evidence

`doa.azimuths_to_degrees:144`:
```python
if np.max(np.abs(arr)) <= 2.0 * math.pi + 0.2:   # ≤ 6.483
    arr = np.degrees(arr)
```

#### Technical explanation

Рішення приймається **за максимумом з чотирьох значень цього зчитування**.
Якщо прошивка віддає ГРАДУСИ, і всі чотири промені опинилися нижче 6.48°
(дрон майже точно на нулі масиву), значення будуть помножені на 57.3:
`[3, 5, 6, 6.4]` → `[172°, 286°, 344°, 367°→7°]`.
Наступного зчитування один промінь перевищить 6.48 → інтерпретація
переключиться назад. Тобто **та сама фізична ціль давала б стрибки по всьому
колу**.

Симетрично: якщо прошивка віддає радіани, помилки не буде ніколи (0..2π завжди
≤ 6.48).

#### Can explain LEFT/RIGHT?

**PARTIALLY** — і лише якщо прошивка віддає градуси, чого **я підтвердити не
можу**. Приклад у docstring (`0.273 3.688 2.993 2.993`) виглядає як радіани.

#### Confidence

**LOW-MEDIUM.** Потребує TEST-06 (сирі значення).

---

### Finding #8 — Порядок каналів і геометрія масиву неперевірені

**Статус: HIGHLY LIKELY PROBLEM (але релевантний лише коли активний SRP-PHAT)**

Повний розбір — §4 і §5. Коротко:

- `mic_channels` = «перші 4 канали» — припущення, задокументоване як
  припущення (`audio_io.py:42-55`).
- `mic_positions_m` = квадрат 43 мм — DEFAULT, ніколи не звірявся з платою
  (`calibration.py:44-47`).
- Перестановка «верхній ряд ↔ нижній ряд» дає `θ → −θ` = **точне дзеркало
  ЛІВО/ПРАВО**, яке жоден `srp_zero_deg` не виправить.

**АЛЕ:** `DOAProvider.update:847` викликає SRP **лише коли USB не дав кута**.
Якщо `xvf_host.py` знайдено і працює, ця гілка мертва, і channel order на
ваш симптом **не впливає взагалі**.

**Required test:** TEST-02 + TEST-03.

---

## 10. ONE DRONE → TWO TARGETS Investigation

### A. MULTIPLE DOA PEAKS

**Не є джерелом другої цілі.**

- `ArrayDOA.estimate` бере `int(np.argmax(scores))` — **рівно один** максимум.
  Другорядні піки не виявляються і не повертаються взагалі.
- `select_azimuth` повертає **один** кут. Другий кластер, якщо він є,
  **відкидається і ніде не зберігається**.
- `DOAReading` має поле `angle_deg: float | None` — **одне** число.
- `AcousticObservation.bearing_deg` — **одне** число.
- `FusedTarget` містить **одну** акустичну обсервацію.

**Висновок: у backend НЕМАЄ структури даних, здатної представити дві акустичні
цілі.** Отже, друга ціль може народитися ТІЛЬКИ на рендері. Так воно і є —
див. E і F.

---

### B. REFLECTIONS / MULTIPATH

**Статус: HARDWARE / ENVIRONMENT POSSIBILITY — вплив на код доведений**

Прямого коду «відбиття» немає, але механізм зв'язку є і він у коді:

1. Відбиття від стіни/будівлі/землі приходить із іншого напрямку.
2. DSP розподіляє свої 4 промені між прямим і відбитим шляхом →
   `AEC_AZIMUTH_VALUES` дає розклад 2-проти-2 або 3-проти-1.
3. `select_azimuth` виставляє `ambiguous = True` і **обирає кластер із меншим
   азимутом** (Finding #4) — не обов'язково прямий шлях.
4. `ambiguous = True` доходить до `radar_overlay._draw_target` і **малює
   другий маркер** (див. E).

Тобто відбиття не створює другої цілі саме по собі — воно **вмикає механізм,
який її малює**.

**PHAT-нормалізація** (`spec /= |spec|`) у SRP-гілці навмисно робить пік
вужчим і стійкішим до реверберації, але **не усуває** відбиття як окремий пік
— вона лише зменшує його вплив на форму.

---

### C. SIDE LOBES / GRATING LOBES

**Статус: POSSIBLE PROBLEM (обчислено, не виміряно)**

Обчислено з DEFAULT-геометрії:
```
діагональні пари 0-2 і 1-3: база 60.8 мм → просторовий Найквіст 2820 Гц
смуга, яку використовує ArrayDOA:                        200–3500 Гц
```
**Верхні 680 Гц смуги перевищують просторовий Найквіст діагональних пар.**
Це створює побічні (grating) максимуми у просторовому спектрі `scores`.

Наслідки в коді:
- `argmax` може вибрати grating lobe замість головного піка → **стрибок кута**;
- `confidence = (peak − median) / (max − min)` при двох близьких за висотою
  піках падає, але кут усе одно віддається як єдиний результат;
- SRP **ніколи не позначає це як ambiguous** (`ambiguous = n_ch < 3` —
  тільки за кількістю каналів).

**Важливо:** це стосується **лише SRP-гілки**, яка на вашій установці,
найімовірніше, не активна. Для XVF3800 DSP аналогічний ефект можливий, але
**UNVERIFIED** — його алгоритм невідомий.

---

### D. FRONT/BACK AMBIGUITY

**Статус: POSSIBLE PROBLEM**

Плоский 2-D масив із 4 мікрофонів у горизонтальній площині:
- **По азимуту front/back неоднозначності НЕМАЄ** — квадратна апертура
  розрізняє всі 360° азимуту (на відміну від лінійного 2-мікрофонного).
- **По ЕЛЕВАЦІЇ неоднозначність повна**: масив плоский, тому джерело під
  кутом +30° над горизонтом і −30° під ним дають однакові часові зсуви.
  Код це усвідомлює і ніде елевацію не стверджує
  (`camera_cue.py:51-56`, `hud.py:568` «ELEV NOT MEASURED»). **Працює правильно.**
- Дрон **високо над станцією** (майже в зеніті) дає всі затримки ≈ 0 →
  просторовий спектр майже плоский → азимут стає майже випадковим.
  Це реальний механізм нестабільності, але він дає **один** блукаючий кут,
  а не дві цілі.

---

### E. ⚠️ ГОЛОВНА ПРИЧИНА ДРУГОЇ ЦІЛІ: РАДАР МАЛЮЄ «ГОСТА» НА ВИГАДАНОМУ КУТІ

**Статус: CONFIRMED BUG**

#### Evidence from code

`radar_overlay.py:331-339`:
```python
# ── Mirror ghost for a 2-mic ambiguity ──
if acoustic.bearing_ambiguous:
    mirror = (180.0 - bearing) % 360.0
    mr = (min(distance * px_per_m, R - 2) if distance is not None
          else R * 0.55)
    mx, my = _polar_to_xy(cx, cy, mirror, mr)
    cv2.circle(tile, (mx, my), 5, colour, 1, cv2.LINE_AA)
    cv2.line(tile, (mx - 3, my - 3), (mx + 3, my + 3), colour, 1, cv2.LINE_AA)
```

Той самий дефект продубльовано у застарілому GUI —
`radar_gui.py:238-240`:
```python
if st.angle_ambiguous:
    mx, my = to_screen(cx, cy, (180.0 - angle) % 360.0, ...)
```

#### Технічне пояснення — ЧОМУ ЦЕЙ КУТ ВИГАДАНИЙ

Коментар у `radar_overlay.py:57-58` стверджує, що гост — це «альтернативний
розв'язок для 2-мікрофонного масиву». Це невірно у двох сценаріях із двох:

**Сценарій 1 — USB (ваша основна гілка).** `ambiguous` тут означає
**нічию між двома виміряними кластерами променів**. Другий кластер — це
конкретне число, яке `select_azimuth` **відкидає і нікуди не передає**.
Радар його не має і малює замість нього `180 − bearing`.

Відтворено:

| Промені DSP | Обраний bearing | Гост намальовано на | Реальний другий кластер |
|---|---|---|---|
| `[300, 302, 60, 62]` | 61.0° | **119.0°** | **301.0°** |
| `[10, 12, 190, 192]` | 11.0° | **169.0°** | **191.0°** |
| `[350, 352, 170, 172]` | 171.0° | **9.0°** | **351.0°** |
| `[0, 90, 180, 270]` | 0.0° | 180.0° | 90/180/270° |

**Жодного разу гост не збігся з реальним другим кластером.**

**Сценарій 2 — SRP із 2 каналами.** Тут `ambiguous = (n_ch < 3)` справді
означає дзеркальну неоднозначність лінійної бази. Але правильне дзеркало для
бази вздовж осі x — це `raw → −raw`, що в канонічному кадрі
(`canonical = zero − raw`) дає **`ghost = (2·zero − bearing) mod 360`**.
Формула `180 − bearing` збігається з нею **лише якщо `zero = 90°`**.
При `srp_zero_deg = null` (zero = 0) правильний гост був би
`360 − bearing` — а малюється `180 − bearing`. **Розбіжність 180°.**

#### Can explain "one drone → two targets"?

**YES — ПОВНІСТЮ.** Це буквально другий маркер, намальований на радарі для
однієї фізичної цілі, на куті, який ніхто не вимірював.

#### Чому «частіше на далеких дистанціях»

`ambiguous` вмикається саме тоді, коли промені DSP розповзаються, тобто при
низькому SNR — тобто на великих відстанях. Отже, гост з'являється саме тоді,
коли ви його й бачите.

#### Confidence

**HIGH (CONFIRMED).** Формула, її вхід і її розбіжність із реальним другим
кластером — усі відтворені.

---

### F. STALE TARGETS / STICKY AMBIGUOUS

**Статус: CONFIRMED BUG (підтримує другу ціль довше, ніж вона існує)**

#### Evidence from code

`doa.DOATracker.update:736`: `self.ambiguous = reading.ambiguous` —
виставляється **до** будь-якої перевірки і **навіть тоді, коли вимір буде
відкинуто** гілкою `if reading.ambiguous: return`.

`doa.DOATracker.reset()` очищає `angle`, `confidence`, `source`, `calibrated`,
`convention` — але **НЕ `self.ambiguous`**.

`doa.DOAProvider.hold()` → `tracker.update(DOAReading(None))` → перша ж
перевірка `if not reading.ok: ... return` — `ambiguous` **не змінюється**.

`doa.DOAProvider._to_canonical`:
```python
ambiguous = self.tracker.ambiguous or reading.ambiguous
```

#### Наслідок

Досить **одного** неоднозначного зчитування, щоб `ambiguous` став `True` і
залишався таким, доки не надійде однозначний вимір, який пройде до `_accept`.
Під час `ALARM_COASTING` (до 3.0 с) вимірів **немає взагалі** — прапорець
заморожується разом із кутом, і **гост продовжує малюватися** весь час
утримання.

При `reset()` (втрата треку на 6 с) прапорець теж не скидається — наступна
ціль стартує з успадкованим `ambiguous = True`.

---

### G. TEMPORAL FILTERING / ТРЕЙЛ

**Статус: HIGHLY LIKELY PROBLEM (візуально виглядає як дві цілі)**

`fusion_config.UIConfig.radar_trail_s = 8.0`
`target_state.AcousticTrackHistory.trail()` віддає **всі** точки за останні 8 с.
`radar_overlay._draw_trail` малює кожну як точку на її власному bearing.

Якщо трек стрибнув (Finding #2, #3, #4), **8 секунд** на радарі одночасно
присутні:
- згасаюча група точок на СТАРОМУ напрямку,
- свіжий бліп на НОВОМУ напрямку,
- (плюс гост із E).

Оператор бачить **до трьох** позначок для одного дрона.

Додатково: `AcousticTrackHistory.add` викликається лише коли
`acoustic.detected and not acoustic.coasting` (`sensor_fusion.py:428-433`) —
це коректно; історія **не** засмічується під час coasting. Це працює правильно.

Трейл очищається у `SensorFusion._transition` **тільки** при переході в
`SEARCHING`, не в `TARGET_LOST` (`sensor_fusion.py:811-818`). У стані
`TARGET_LOST` (утримується `target_lost_display_s`) старі точки ще на екрані.

---

### H. THREADING / RACE CONDITIONS

**Статус: перевірено — критичних гонок у шляху даних НЕ ЗНАЙДЕНО, але є одна
реальна конкуренція за пристрій**

Що **працює правильно**:
- `HardwareDOA` захищає `_reading`/`_stamp` одним `threading.Lock`.
- `LatestValue` (target_state) — потокобезпечна публікація знімків.
- `AcousticObservation` — `frozen` dataclass; `acoustic_worker._to_observation`
  робить копію, бо `RadarStatus` мутується на місці. Задокументовано і коректно.
- `SensorFusion.update` викликається з одного (UI) потоку.
- Один аудіоблок обробляється рівно один раз (`_loop` послідовний).
- Дедуплікація по `seq` у `sensor_fusion.py:418` не дає UI вставити той самий
  акустичний кадр в історію двічі. **Коректно.**

Що є проблемою:

**ДВА ПОТОКИ НЕЗАЛЕЖНО ЗАПУСКАЮТЬ `xvf_host.py` ПРОТИ ОДНОГО USB-ПРИСТРОЮ,
БЕЗ СПІЛЬНОГО БЛОКУВАННЯ:**

- `doa.HardwareDOA._loop` — потік `Thread(target=self._loop)`, кожні 0.35 с,
  команда `AEC_AZIMUTH_VALUES` (читання).
- `respeaker_led.RespeakerLed` — власний робочий потік, команди
  `LED_RING_COLOR` тощо (запис), плюс `discover_commands` на старті.

Спільного м'ютекса між `doa.py` і `respeaker_led.py` немає (перевірено grep'ом).

**Статус: POSSIBLE PROBLEM.** Чи призводить одночасний доступ до контрольного
endpoint XVF3800 до збоїв читання азимута — **UNVERIFIED**, це залежить від
прошивки та драйвера. Симптоми були б: таймаути, нерозпізнаний вивід, або —
найгірше — **прочитане не те значення**.

---

## 11. Distance / Low SNR Investigation

Тут я НЕ роблю висновку «відстань є причиною». Я показую лише те, що
**прослідковується у коді або в фізиці**.

### Що ДІЙСНО залежить від відстані

| Механізм | Ланцюг | Наслідок |
|---|---|---|
| Розкид променів DSP | ↓SNR → промені розходяться → розклад 2-проти-2 → `select_azimuth` → `ambiguous=True` + вибір кластера з меншим кутом | **ЛІВО/ПРАВО (Finding #4) І ДРУГА ЦІЛЬ (§10-E)** — обидва симптоми одночасно |
| Відносна вага відбиття | Прямий шлях слабшає як 1/r; відбита складова від великої стіни слабшає повільніше → на великій відстані відбиття стає порівнянним | живить попередній рядок |
| Частотне згасання | Атмосферне поглинання росте з частотою; на відстані спектр дрона зсувається вниз → у SRP-гілці більше енергії лишається у 200–1500 Гц, де база 43 мм дає мізерну різницю фаз | ↓ точність SRP, не дзеркалення |
| Роздільна здатність TDOA | Максимальна затримка для бази 43 мм — **2.01 семпла** @16 кГц. Уся інформація про напрямок вміщена в ±2 семпли. При ↓SNR похибка оцінки піка ±0.5 семпла = **±25° по кутy** | великий джиттер кута на межі чутності |
| Шумовий гейт і стан детектора | Кут вимірюється **тільки** у станах TRACK/ALARM (`radar.py:592`). Далекий дрон частіше провалюється в ALARM_COASTING, де кут **заморожується** | старий кут тримається до 3.0 с; при поверненні сигналу з іншого боку — видимий стрибок |
| Кешування азимута | Finding #2: одне зчитування рахується кілька разів. На межі чутності частка ХИБНИХ зчитувань зростає, а множник лишається тим самим | ↑ ймовірність стрибка треку |

### Що НЕ залежить від відстані

- **Помилка напряму обертання (Finding #1)** — вона залежить від **азимуту**
  (`похибка = 2·raw`), а не від дальності. Може виглядати дистанційно
  залежною, якщо ближні перевірки робилися перед станцією, а дальні — збоку.
- **Формула госта `180 − bearing`** — вигадана однаково на будь-якій відстані;
  залежить від відстані лише **частота вмикання** прапорця `ambiguous`.

### Чого перевірити не можу

- Реальний акустичний спектр вашого дрона — **UNVERIFIED**.
- Фонові рівні на майданчику: `noise_floor_dbfs = -55.0`, а
  `range_ref_level_dbfs = -22.0 @ 3.0 м` — обидва **круглі числа без
  `range_spreading_db`**, тобто `acoustic_worker._warn_if_range_calibration_
  looks_synthetic` (рядки 224-277) **спрацює і назве це калібрування
  написаним від руки**. Усі метри на екрані — довільні. Це не впливає на
  напрямок, але впливає на радіус, на якому малюється бліп і гост.
- Вплив вітру на мікрофон — **UNVERIFIED**.

---

## 12. Confirmed Bugs

Тільки те, що доведено кодом або відтворено.

| # | Bug | Файл:рядок | Симптом |
|---|---|---|---|
| **C-1** | `doa_handedness` відсутній у конфігурації → `Handedness.UNKNOWN` → мовчазна здогадка `CLOCKWISE`; вимір користувача каже CCW | `bearing_frame.py:281-289, 199-201`; `radar_calibration.json` | **LEFT/RIGHT** — похибка `2·raw`, рівно 180° при raw 90°/270° |
| **C-2** | Радар малює другий маркер на **вигаданому** куті `(180 − bearing)`, який не є ані виміряним другим кластером, ані правильним дзеркалом SRP | `radar_overlay.py:331-339`; дубль у `radar_gui.py:238-240` | **ДВІ ЦІЛІ** |
| **C-3** | Одне апаратне зчитування зараховується трекеру 2-4 (до 8) разів; `JUMP_CONFIRMATIONS=3` рахує виклики, а не виміри | `doa.py:478-483, 760-770`; `radar.py:596` | **раптовий стрибок на протилежний бік** |
| **C-4** | Неоднозначний вимір приймається при ініціалізації треку і в межах ±35°; блокується лише як викид → хибний бік фіксується | `doa.py:729-760` | **залипання на хибному боці** |
| **C-5** | Нічия 2-проти-2 завжди виграється кластером із МЕНШИМ азимутом (`sort` по `circular_mean`) | `doa.py:206-211` | **дзеркальний вибір** при рознесенні ≈180° |
| **C-6** | `calibrate.py doa` з однією точкою завжди пише `CW` (суворе `<` на нульовому residual) і робить джерело «каліброваним», прибираючи попередження `UNCAL` | `calibrate.py:424-433, 458-462` | **прихована дзеркальність** |
| **C-7** | `DOATracker.ambiguous` — «липкий»: не скидається ані в `reset()`, ані в `hold()`, ані при відкиданні виміру | `doa.py:711-718, 736; 925-938` | **гост тримається довше, ніж неоднозначність** |
| **C-8** | `doa_invert` мертвий у runtime, але друкується в банері як «дзеркально» | `calibration.py:136`; `bearing_frame.py:281` | **маскує C-1** |

---

## 13. Highly Likely Problems

| # | Проблема | Підстава | Потрібен тест |
|---|---|---|---|
| **H-1** | Трейл 8 с зберігає точки на СТАРОМУ напрямку після стрибка → на екрані до трьох позначок для одного дрона | `fusion_config.py:556`; `radar_overlay._draw_trail` | TEST-08 |
| **H-2** | `mic_channels` = «перші 4» — припущення. Хибне → SRP рахує геометрію не по тих мікрофонах | `audio_io.py:128-133`, і сам код це декларує | TEST-02, TEST-03 |
| **H-3** | `mic_positions_m` — DEFAULT 43 мм, ніколи не звірявся з платою. Перестановка «верх↔низ» = точне дзеркало ЛІВО/ПРАВО | `calibration.py:44-49` | TEST-03 |
| **H-4** | Субпроцес python кожні 0.35 с (200-500 мс на старт інтерпретатора) конкурує з аудіобюджетом 250 мс на Pi 5 | `doa.py:33-36`; `acoustic_worker.py:386` | TEST-09 |
| **H-5** | Калібрування дальності виглядає написаним від руки (круглі −22.0 dBFS @ 3.0 м, немає `range_spreading_db`) → усі метри довільні, отже радіус госта і бліпа теж | `radar_calibration.json`; `acoustic_worker.py:224-277` | запустити `calibrate.py range` |
| **H-6** | `boresight_calibrated_at_doa_offset_deg` і `boresight_calibrated_handedness` не задані → перевірка застарілості кадру **мовчить**, хоча `doa_offset_deg` = 180 | `fusion_config.py:865-867`; `fusion_config.json` | — |

---

## 14. Possible Problems

| # | Проблема | Чому не доведено |
|---|---|---|
| **P-1** | Grating lobes SRP: діагональні пари працюють вище просторового Найквіста (2820 Гц) у смузі до 3500 Гц | обчислено з DEFAULT-геометрії, яка сама неперевірена; SRP-гілка, ймовірно, неактивна |
| **P-2** | Якщо пристрій віддає лише 2 канали і USB DOA недоступний → `ambiguous = (n_ch < 3)` = **завжди True** → гост малюється **постійно** | невідомо, скільки каналів віддає ваш пристрій |
| **P-3** | Евристика радіани/градуси може перемикатися між зчитуваннями | семантика прошивки невідома |
| **P-4** | Два потоки одночасно ганяють `xvf_host.py` проти одного USB-пристрою без спільного блокування | наслідки залежать від прошивки/драйвера |
| **P-5** | `AEC_AZIMUTH_VALUES` може означати не «4 азимути 4 променів» — тоді вся логіка `select_azimuth` не має підстав | документації прошивки в репозиторії немає |
| **P-6** | Дрон майже в зеніті → всі затримки ≈0 → азимут майже випадковий (плоский масив не міряє елевацію) | фізика підтверджує, вимірювання немає |
| **P-7** | `ArrayDOA.angles = arange(0, 360, 360 // n_angles)` — цілочисельне ділення; при зміні `n_angles` кількість напрямків зміниться мовчки | за замовчуванням (72) безпечно |

---

## 15. Things That Are Working Correctly

Це важливо не менше за дефекти. Наступне перевірено і **справне** —
шукати тут баги не треба.

1. **Канонічний координатний кадр самоузгоджений.** Доведено **виконанням**
   `python bearing_frame.py`: 0=FRONT, 90=RIGHT, 180=BEHIND, 270=LEFT
   однаково для радара, LED і камери. Перевірка має «зуби» — вона ловить
   дзеркальне кільце.
2. **`radar_overlay._polar_to_xy` правильний.** `(cx + r·cos(b−90), cy + r·sin(b−90))`
   при y вниз дає 0=вгору, 90=вправо. Радар **не** дзеркалить.
3. **Подвійних перетворень НЕМАЄ.** `apply_orientation`/`unapply_orientation`
   у runtime не викликаються; кожен вихідний шар робить рівно одне власне
   перетворення. Це раніше було проблемою і зараз виправлено.
4. **Канонізація виконується ДО згладжування** (`doa.py:876`). Трекер
   згладжує однорідні числа; перемикання USB↔SRP більше не рухає трек стрибком.
5. **У кожного джерела власна конвенція** (`source_convention`). Спільний
   `doa_offset_deg` для USB і SRP більше не застосовується.
6. **Циркулярна арифметика правильна всюди.** `circular_mean`, `wrap180`,
   `wrap360`, `angular_distance` централізовані й перевірені виконанням.
   Жодної помилки `(a+b)/2` на кутах не знайдено.
7. **Математика SRP-PHAT самоузгоджена.** Знак `expected_tau` відповідає
   порядку аргументів GCC; інтерполяція коректна; вибірка ніколи не виходить
   за межі кореляційного вікна.
8. **Вироджені пари виключаються, а не вимикають SRP назавжди**
   (`usable_pairs` + `ARRAY_DISABLE_AFTER = 20`). Сліпа смуга ±90° усунена.
9. **Fusion-шар не створює другої цілі.** `SensorFusion` тримає рівно один
   `FusedTarget` з одним акустичним `AcousticObservation`. Багатоцільового
   трекера в системі немає взагалі.
10. **Дедуплікація акустичних кадрів по `seq`** (`sensor_fusion.py:418`) не дає
    UI вставити той самий вимір в історію двічі. Працює.
11. **Історія не засмічується під час coasting** (`sensor_fusion.py:428`) —
    заморожений кут не потрапляє у трейл як новий вимір.
12. **Кут не вимірюється на порожньому фоні** (`radar.py:592`) — тільки
    в TRACK/ALARM. Це правильно: SRP на шумі дав би випадкові кути.
13. **`to_mono` обмежений мікрофонними каналами** — опорні канали не
    домішуються у вхід класифікатора.
14. **Потокова безпека шляху даних коректна:** `Lock` у `HardwareDOA`,
    `LatestValue`, immutable `AcousticObservation`, копія `RadarStatus`.
15. **Часові інтервали рахуються по `time.monotonic()`** у `DOATracker`,
    `HardwareDOA`, `target_state.now()`, `acoustic_worker` — стрибок NTP на
    Pi без RTC не зламає таймаути.
16. **Система чесно каже, що не виміряно.** `UNCAL`, `BEARING UNVERIFIED`,
    «н/д» замість вигаданих метрів, «HANDEDNESS UNKNOWN» у банері — усе це
    працює і виводиться. Проблема C-6 полягає саме в тому, що одноточкове
    калібрування ці попередження **вимикає**.
17. **`web_server.py` не торкається кутів.** Він приймає готовий BGR-кадр і
    кодує JPEG. Дзеркалення на шляху до браузера **виключено**.
18. **LED-кільце малює рівно ОДИН сектор** і при відсутності bearing світить
    усе кільце (а не один довільний світлодіод). Другої цілі не створює.

---

## 16. Missing Evidence

Що **неможливо** довести самим кодом:

1. **Вміст `radar_calibration.json` на Raspberry Pi.** Файл у `.gitignore`.
   Від нього залежить, чи є C-1 активним прямо зараз.
2. **Скільки каналів реально віддає ваш XVF3800** і які з них мікрофони.
3. **Реальна геометрія плати** — розміри, порядок, орієнтація.
4. **Нуль і напрямок обертання азимута прошивки XVF3800.**
5. **Семантика `AEC_AZIMUTH_VALUES`** — що саме означають ці 4 числа.
6. **Одиниці, які віддає прошивка** (радіани чи градуси).
7. **Чи виконує DSP beamforming перед видачею каналів.**
8. **Чи конфліктують два потоки за USB-контроль пристрою.**
9. **Реальна частота опитування DSP на вашому Pi** (від неї залежить
   множник дублювання в C-3).
10. **Акустичне середовище майданчика** — де стіни, який фон, який вітер.
11. **Спектр вашого дрона** і реальна дальність виявлення.
12. **Фізичне розташування LED 0** на кільці (`led_zero_offset_deg` = 0.0 —
    це DEFAULT, не вимір).

---

## 17. REQUIRED DIAGNOSTIC TESTS

**Це тести, а не виправлення. Нічого не змінюйте за їх результатами без
окремого рішення.**

---

### TEST-01 — Чи записаний напрямок обертання масиву на Pi (НАЙВАЖЛИВІШИЙ)

**Purpose:** з'ясувати, чи активний C-1 прямо зараз.

**What to do (на Raspberry Pi):**
```bash
cd ~/…/acoustic_radar
cat radar_calibration.json
python -c "import calibration, bearing_frame; c=calibration.load(); print(bearing_frame.source_convention('usb', c).describe())"
```

**Expected result:**
`XVF3800 USB: zero at +354deg, CCW` — конвенція виміряна повністю.

**What different outcomes mean:**
- Рядок містить **`HANDEDNESS UNKNOWN — may be MIRRORED, assuming CW`**
  → **C-1 АКТИВНИЙ.** Ваше ліво/право дзеркальне на 2·raw. Це головна причина.
- Рядок містить **`CW (proven)`** → конвенція записана як CW, але ваш
  попередній 5-точковий вимір дав CCW (residual 7.9° проти 70.9°) →
  **або запис зроблено ОДНІЄЮ точкою (C-6), або він застарілий.** Переходьте
  до TEST-07.
- Рядок містить **`CCW`** → C-1 і C-6 виключені. Причину слід шукати у
  C-3/C-4/C-5.

---

### TEST-02 — Скільки каналів і які з них живі

**Purpose:** встановити фактичний формат аудіопотоку.

**What to do:**
```bash
python radar.py --list-devices
python -c "import calibration, audio_io, features; print(audio_io.resolve_input(calibration.load(), features.SAMPLE_RATE).describe())"
```

**Expected result:** рядок на кшталт
`ReSpeaker … — 6 кан. @ 16000 Гц (сирі канали масиву)`.

**What different outcomes mean:**
- **2 канали** → SRP-PHAT неможливий; **уся SRP-гілка мертва**, тому H-2, H-3,
  P-1 до вашого симптому не стосуються. Але тоді, якщо USB DOA колись відмовить,
  спрацює P-2 і гост малюватиметься **постійно**.
- **6 каналів** → перевірте TEST-03, бо `mic_channels` = «перші 4».
- **1 канал** → напрямку немає в принципі; система має писати «н/д».

---

### TEST-03 — Постукування по кожному мікрофону (перевірка порядку каналів)

**Purpose:** довести або спростувати H-2 і H-3.

**What to do:**
1. Позначте мікрофони на платі за годинниковою: A, B, C, D, почавши з того,
   що ближчий до USB-роз'єму.
2. `python calibrate.py check` — він друкує рівні по каналах.
3. Для кожного мікрофона по черзі: акуратно постукайте **тільки по ньому**
   (нігтем, не по платі) і запишіть, у якому каналі стрибнув рівень.
4. Складіть таблицю A→канал?, B→канал?, C→канал?, D→канал?

**Expected result:** A→0, B→1, C→2, D→3 у порядку, що відповідає
`mic_positions_m` (верх-ліво, верх-право, низ-право, низ-ліво).

**What different outcomes mean:**
- Порядок збігається → H-2 і H-3 знято.
- Порядок є **перестановкою верхнього і нижнього рядів** (0↔3, 1↔2) →
  **точне дзеркало ЛІВО/ПРАВО на SRP-гілці.**
- Порядок є **циклічним зсувом** → стала помилка 90°/180°/270°, не дзеркало.
- Стрибок видно в каналах 4 або 5 → мікрофони **не** в перших чотирьох
  каналах; `mic_channels` неправильний.

---

### TEST-04 — Скільки РІЗНИХ азимутів насправді бачить трекер

**Purpose:** виміряти множник дублювання з C-3.

**What to do (тимчасовий діагностичний скрипт, НЕ зміна проєкту):**
```bash
python - <<'EOF'
import time, collections
from doa import HardwareDOA
hw = HardwareDOA(); hw.start(); time.sleep(1)
seen, samples = [], []
t0 = time.monotonic()
while time.monotonic() - t0 < 30:      # 30 с, читаємо як аудіоцикл — 4 Гц
    r = hw.read()
    samples.append(None if r.angle_deg is None else round(r.angle_deg, 3))
    time.sleep(0.25)
hw.stop()
c = collections.Counter(samples)
print("зчитувань аудіоциклом:", len(samples))
print("РІЗНИХ значень:", len(c))
print("множник дублювання:", len(samples)/max(len(c),1))
print("топ-5:", c.most_common(5))
EOF
```

**Expected result:** множник ≈ 1.0–1.4 (кожен вимір споживається один раз).

**What different outcomes mean:**
- **Множник ≥ 2.0** → C-3 підтверджено: 3 «підтвердження» стрибка можуть
  надійти з 1-2 фізичних вимірів.
- **Множник ≥ 4.0** → одне зчитування само по собі здатне перекинути трек.
- **Багато `None`** → USB DOA нестабільний; система переходить на SRP, і
  тоді релевантні H-2/H-3/P-1/P-2.

---

### TEST-05 — Чи фіксується трек на хибному боці (C-4)

**Purpose:** довести latch неоднозначного захоплення.

**What to do:** без заліза, чистий Python на Pi (numpy там є):
```bash
python - <<'EOF'
from doa import DOATracker, DOAReading
tr = DOATracker()
# перший вимір — НЕОДНОЗНАЧНИЙ, хибний бік
tr.update(DOAReading(90.0, confidence=0.25, source="usb", ambiguous=True), now=0.0)
print("після хибного неоднозначного захоплення:", tr.format())
# далі 10 ОДНОЗНАЧНИХ вимірів на правильному боці 270
for i in range(10):
    tr.update(DOAReading(270.0, confidence=0.9, source="usb"), now=0.5*(i+1))
    print(f"  крок {i+1}: {tr.format()}  ambiguous={tr.ambiguous}")
EOF
```

**Expected result:** трек має перейти на 270° після 3 підтверджень.

**What different outcomes mean:**
- Якщо трек **тримається біля 90°** довше 3 кроків → C-4 підтверджено.
- Якщо `ambiguous` лишається `True` після переходу на 270° → **C-7
  підтверджено**, гост малюватиметься на однозначній цілі.

---

### TEST-06 — Сирі промені DSP проти реального положення дрона

**Purpose:** одночасно перевірити C-5, Finding #7 і P-5 — три невідомі одним
експериментом.

**What to do:**
1. Поставте гучне джерело (дрон у зависанні або динамік із записом дрона)
   на **відомому** азимуті, для якого ви вже знаєте, що система показує
   дзеркально — наприклад, справа від станції.
2. Запустіть:
```bash
python - <<'EOF'
import time
from doa import HardwareDOA, parse_azimuth_values, azimuths_to_degrees, select_azimuth
hw = HardwareDOA()
for _ in range(40):
    r = hw._query_once()
    if r.ok:
        print(f"raw_deg={[round(x,1) for x in r.raw]}  ->  обрано {r.angle_deg:6.1f}  "
              f"conf={r.confidence:.2f}  ambiguous={r.ambiguous}")
    else:
        print("ERR:", r.error)
    time.sleep(0.4)
EOF
```
3. Запишіть 40 рядків разом із фізичним положенням дрона.

**Expected result:** усі чотири `raw_deg` кластеризуються навколо ОДНОГО
значення, `ambiguous=False`, `conf ≈ 1.0`.

**What different outcomes mean:**
- **Регулярний розклад 2-проти-2** → C-5 активний, і система систематично
  обирає менший азимут. Занотуйте, чи правильний кластер — той, що більший.
- **`ambiguous=True` частіше на далеких дистанціях** → доведено зв'язок
  «відстань → друга ціль» (§10-E, §11).
- **Значення виглядають як 0…360 (не 0…6.28)** → прошивка віддає **градуси**,
  і Finding #7 стає реальним ризиком.
- **Значення завжди 0…6.28** → прошивка віддає радіани, Finding #7 знято.
- **Значення не схожі на азимути взагалі** (наприклад, одне велике і три нулі,
  або значення поза 0…360) → **P-5 підтверджено: `AEC_AZIMUTH_VALUES` означає
  не те, що припускає код**, і вся логіка `select_azimuth` не має підстав.

---

### TEST-07 — Двоточкове визначення напрямку обертання (вирішальний)

**Purpose:** остаточно закрити питання ЛІВО/ПРАВО.

**What to do:**
1. Поставте джерело **точно спереду** станції — це 0° у тому кадрі, який ви
   хочете бачити на екрані.
2. Поставте те саме джерело **точно праворуч** — це 90°.
3. Запустіть `python calibrate.py doa` і введіть **обидві** точки
   (0, потім 90). **Не завершуйте після першої** — одна точка дає C-6.
4. Прочитайте рядок `XVF3800 USB: zero at …, CW|CCW` і середню похибку.

**Expected result:** residual < 15°, і напрямок обертання визначений.

**What different outcomes mean:**
- **CCW із малим residual** → підтверджує ваш попередній 5-точковий вимір і
  доводить, що поточна CW-здогадка була причиною дзеркалення (C-1).
- **CW із малим residual** → C-1 знято; причина в C-3/C-4/C-5.
- **residual > 30° для ОБОХ гіпотез** → масив віддає кут нестабільно;
  переходьте до TEST-06.
- **Обидві гіпотези дають residual 0.0** → ви ввели лише одну точку;
  результат нічого не доводить (це і є C-6).

---

### TEST-08 — Чи є друга ціль ГОСТОМ, трейлом, чи чимось третім

**Purpose:** розрізнити C-2, C-7 і H-1 на екрані.

**What to do:** коли на радарі видно дві позначки, роздивіться їх ФОРМУ
(`radar_overlay.py`):
- **суцільне коло Ø5 + обвідне коло Ø8** = основний бліп (одна штука);
- **порожнє коло Ø5 з ПЕРЕКРЕСЛЕННЯМ** (діагональна риска) = **ГОСТ (C-2)**;
- **дрібні точки Ø1, що згасають** = **ТРЕЙЛ (H-1)**;
- **штрихований промінь від центру** = bearing без дальності.

Додатково запишіть кути обох позначок і перевірте: якщо
`друга ≈ (180 − перша) mod 360` → це точно гост.

**Expected result / meaning:**
- Гост підтверджено → C-2 і є причиною «двох цілей».
- Дрібні згасаючі точки на старому напрямку → H-1; корінь у стрибку треку
  (C-3/C-4/C-5), а не в рендері.
- Дві **однакові суцільні** позначки → у системі з'явилося щось, чого я в коді
  не знайшов; повідомте, це змінює висновок.

---

### TEST-09 — Чи встигає аудіоцикл (H-4)

**Purpose:** перевірити, чи субпроцес DOA не з'їдає аудіобюджет.

**What to do:** запустіть станцію на 10 хвилин і подивіться `logs/station.log`:
```bash
grep -E "audio-overflow|audio-slow|block processing" logs/station.log
```

**Expected result:** жодного `audio-overflow`, mean block ≪ 250 мс.

**What different outcomes mean:**
- Є `audio buffer overflow` → аудіо **втрачається**, і разом із ним частина
  сигналу дрона; кут рахується по неповних даних.
- Є `acoustic processing took … of its 250 ms budget` → Pi на межі.

---

### TEST-10 — Ізолювати LED від опитування DOA (P-4)

**Purpose:** перевірити, чи конкуренція за USB псує зчитування азимута.

**What to do:** проведіть TEST-06 **двічі**: один раз зі станцією, що працює
з увімкненим кільцем, і один раз запустивши станцію з `--no-led`.
Порівняйте частку `ERR:` і частку `ambiguous=True`.

**Expected result:** різниці немає.

**What different outcomes mean:** якщо з `--no-led` помилок/неоднозначностей
помітно менше → **P-4 підтверджено**, два потоки заважають один одному на
USB-контролі.

---

## 18. ROOT CAUSE RANKING

| Rank | Possible Root Cause | Evidence | Confidence | Explains Left/Right | Explains Two Targets |
|---|---|---|---|---|---|
| **1** | `doa_handedness` не записаний → мовчазна здогадка CW проти виміряного CCW (C-1) | `bearing_frame.py:281-289,199-208`; `radar_calibration.json`; `checker.md:527-529,863-864`; **відтворено виконанням** | **HIGH** (на цій конфігурації — CONFIRMED; на Pi — TEST-01) | **YES — повністю.** Похибка `2·raw`: рівно 180° при raw 90°/270°, нуль при 0°/180° → «іноді ліво, іноді право» | NO |
| **2** | Радар малює госта на вигаданому куті `180 − bearing` (C-2) | `radar_overlay.py:331-339`; `radar_gui.py:238-240`; **відтворено**: гост 119° при реальному другому кластері 301° | **HIGH — CONFIRMED** | NO | **YES — повністю.** Це буквально друга позначка для однієї цілі |
| **3** | Нічия 2-проти-2 виграється кластером із меншим азимутом (C-5) | `doa.py:206-211`; **відтворено**: `[350,352,170,172] → 171°` замість 351° | **HIGH — CONFIRMED** | **YES** — при рознесенні ≈180° це точне дзеркало | опосередковано: вмикає `ambiguous`, який вмикає госта |
| **4** | Одне зчитування зараховується 2-4 рази; «3 підтвердження» стрибка з 1-2 вимірів (C-3) | `doa.py:478-483,760-770`; `radar.py:596`; hop 0.25 с проти interval 0.35 с + 200-500 мс субпроцес | **HIGH — CONFIRMED (механізм)**, множник — TEST-04 | **YES, PARTIALLY** — пояснює раптові стрибки, не сталу дзеркальність | опосередковано через трейл (H-1) |
| **5** | Неоднозначний вимір фіксує трек на хибному боці (C-4) | `doa.py:729-760`, порядок `if`-ів | **HIGH — CONFIRMED** | **YES** — залипання | NO |
| **6** | Одноточкове `calibrate.py doa` пише CW і глушить попередження (C-6) | `calibrate.py:424-433`; **відтворено** | **HIGH — CONFIRMED** | **YES** — тим самим механізмом, що #1, але приховано | NO |
| **7** | Трейл 8 с зберігає старий напрямок після стрибка (H-1) | `fusion_config.py:556`; `radar_overlay._draw_trail` | MEDIUM-HIGH | NO | **YES, PARTIALLY** — візуально до трьох позначок |
| **8** | «Липкий» `ambiguous` (C-7) | `doa.py:711-718,736,925-938` | **HIGH — CONFIRMED** | NO | **YES** — подовжує життя госта, зокрема на всі 3 с coasting |
| **9** | Відбиття/multipath розділяють промені DSP | фізика + `select_azimuth`; посилюється з відстанню | MEDIUM | **YES, PARTIALLY** — через #3 | **YES, PARTIALLY** — через #2 |
| **10** | Порядок каналів / геометрія масиву неперевірені (H-2, H-3) | `audio_io.py:128-133`; `calibration.py:44-49` | MEDIUM — **тільки якщо активний SRP** | **YES** для перестановки «верх↔низ» | NO |
| **11** | Grating lobes SRP (2820 Гц Найквіст проти смуги 3500 Гц) (P-1) | обчислено з DEFAULT-геометрії | LOW-MEDIUM | PARTIALLY — стрибки кута | NO (SRP віддає один пік) |
| **12** | Евристика радіани/градуси (Finding #7) | `doa.py:144` | LOW | PARTIALLY | NO |
| **13** | Конкуренція двох потоків за `xvf_host.py` (P-4) | `doa.py` + `respeaker_led.py`, спільного локу немає | LOW — UNVERIFIED | PARTIALLY | PARTIALLY |
| **14** | Дрон у зеніті → плоский просторовий спектр (P-6) | фізика плоского масиву | LOW | PARTIALLY | NO |

---

## 19. FINAL TECHNICAL CONCLUSION

### Що ми знаємо ТОЧНО

1. **Координатний ланцюг від `bearing_deg` до екрана справний.** Це доведено
   виконанням: 270° дає ліво на радарі, на LED і на камері однаково. Радар
   **не** дзеркалить. Подвійних перетворень у системі **немає**. Отже проблема
   не в тому, як число малюється, а в тому, **яке число приходить**.

2. **У backend фізично немає структури для двох акустичних цілей.** Один
   `bearing_deg`, один трек, один `FusedTarget`. Тому друга ціль на радарі —
   **завжди артефакт рендера**, а не другий вимір.

3. **Друга ціль малюється формулою `(180 − bearing)`, і цей кут ніхто не
   вимірював.** Відтворено: при променях `[300, 302, 60, 62]` система обирає
   61°, а госта малює на 119° — тоді як реальний другий кластер стоїть на 301°.
   Жодного разу гост не збігся з виміряним другим напрямком.

4. **Конфігурація, яку я бачу, НЕ містить напрямку обертання масиву.**
   `doa_handedness` відсутній → `Handedness.UNKNOWN` → мовчазна здогадка
   `CLOCKWISE`. Попередній аудит фіксує, що ваш власний 5-точковий вимір дав
   **CCW** (7.9° проти 70.9°).

5. **Якщо масив CCW, а застосовується CW — похибка дорівнює рівно `2·raw`:**
   нуль спереду і ззаду, **повний обмін ЛІВО↔ПРАВО збоку**. Це найточніший
   збіг із вашим описом «показує дзеркально, але не абсолютно постійно».

6. **`doa_invert` у runtime мертвий**, хоча банер друкує «дзеркально». Цей
   параметр не виправить нічого і маскує справжню причину.

7. **Одноточкове `calibrate.py doa` завжди записує CW** і при цьому вимикає
   всі попередження про некаліброваність. Відтворено на трьох прикладах.

### Що майже напевно є проблемою

8. **Захист «3 підтвердження перед стрибком» слабший, ніж читається.** Він
   рахує **виклики**, а не окремі виміри, а одне апаратне зчитування
   споживається аудіоциклом 2-4 рази (у гіршому випадку до 8). Одне хибне
   зчитування може перекинути стрілку на протилежний бік.

9. **Неоднозначний вимір приймається при захопленні цілі** і не блокується
   в межах ±35°. Це фіксує трек на тому боці, який `select_azimuth` обрав
   першим — а обирає він **завжди кластер із меншим азимутом**, без жодної
   акустичної підстави.

10. **Прапорець `ambiguous` липкий.** Один розклад променів 2-проти-2 вмикає
    госта, і той лишається на екрані навіть після того, як неоднозначність
    зникла, і на всі 3 секунди `ALARM_COASTING`.

11. **Трейл живе 8 секунд.** Після стрибка на екрані одночасно старий
    напрямок (точками), новий бліп і гост — до трьох позначок для одного дрона.

### Що поки НЕМОЖЛИВО довести без заліза

- Чи є `doa_handedness` у файлі на Pi (файл у `.gitignore`).
- Скільки каналів віддає ваш пристрій і які з них мікрофони.
- Чи відповідає геометрія `43 мм квадрат` реальній платі.
- Що саме означають 4 числа `AEC_AZIMUTH_VALUES` і в яких одиницях.
- Нуль і напрямок обертання азимута самої прошивки.
- Чи конфліктують потік DOA і потік LED за USB-контроль.

### ЩО ПЕРЕВІРИТИ ПЕРШИМ (3–5 речей, у цьому порядку)

1. **TEST-01** — одна команда на Pi: чи записаний `doa_handedness`.
   Це відповідь «так/ні» на головну гіпотезу ЛІВО/ПРАВО, і вона коштує
   тридцять секунд.
2. **TEST-06** — 40 рядків сирих `AEC_AZIMUTH_VALUES` разом із реальним
   положенням дрона. Один експеримент закриває три невідомі одразу:
   чи буває розклад 2-проти-2, чи це радіани, і чи взагалі ці числа є
   азимутами.
3. **TEST-08** — подивитися на ФОРМУ другої позначки на екрані.
   Перекреслене порожнє коло = гост (C-2). Це підтвердить причину «двох
   цілей» без жодного інструмента.
4. **TEST-07** — двоточкове `calibrate.py doa` (спереду і праворуч).
   **Обов'язково дві точки** — одна точка через C-6 дасть хибний результат,
   який ще й вимкне попередження.
5. **TEST-02 + TEST-03** — скільки каналів і який їх порядок. Це визначає,
   чи взагалі релевантні H-2/H-3/P-1 до вашого симптому, чи SRP-гілка мертва.

---

### Одне речення на кожен із двох ваших симптомів

**ЛІВО/ПРАВО:** найімовірніше — система **не знає** напрямку обертання
масиву, мовчки здогадується «за годинниковою», а масив, за вашим власним
попереднім виміром, рахує **проти** годинникової; похибка `2·raw` дає нуль
спереду і повне дзеркало збоку — це і є «не абсолютно постійно». Підсилюють
цей ефект три незалежні дефекти трекера (арбітражна нічия, дублювання
зчитувань, фіксація на неоднозначному захопленні).

**ДВІ ЦІЛІ:** підтверджено — це **не** другий вимір, а другий **маркер**,
який радар малює за формулою `(180 − bearing)`, ніколи не звірявши її з тим,
що насправді виміряв DSP; плюс восьмисекундний трейл, який після стрибка
залишає на екрані старий напрямок.

---

## Перевірка перед завершенням

- [x] Створено тільки `fiiiiiiixxxx_drone.md` як результат аудиту
- [x] Жоден існуючий файл проєкту не змінено (перевірено `git status`)
- [x] Не зроблено автоматичних fixes, patch не створено
- [x] Проаналізовано весь audio / microphone / DOA pipeline
- [x] Camera / YOLO / Hailo / MJPEG не аналізувалися (крім перевірки, що
      `web_server` не перетворює кути)
- [x] Кожна серйозна гіпотеза має code evidence з номерами рядків
- [x] FACT відділено від HYPOTHESIS статусами CONFIRMED / HIGHLY LIKELY /
      POSSIBLE / HARDWARE-ENVIRONMENT / UNVERIFIED
- [x] Hardware facts не вигадані; усе неперевірене позначено UNVERIFIED або
      NOT PROVEN — REQUIRES HARDWARE TEST
- [x] Знайдено всі точки, де RIGHT може стати LEFT (§9, 8 знахідок)
- [x] Знайдено всі точки, де один дрон стає двома (§10, розділи A–H)
- [x] Написано 10 конкретних diagnostic tests із очікуваними результатами та
      інтерпретацією кожного варіанта результату

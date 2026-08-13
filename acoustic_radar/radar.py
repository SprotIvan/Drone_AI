#!/usr/bin/env python3
"""
radar.py — Акустичний радар реального часу для Raspberry Pi.

Логіка роботи (як і замовлялось):
    є дрон              → ATTENTION → ALARM (SOS) + напрямок + відстань
    сторонні звуки      → нічого
    тиша                → нічого

═══════════════════════════════════════════════════════════════════
⚠️ ГОЛОВНІ ВИПРАВЛЕННЯ ПОРІВНЯНО З ПОПЕРЕДНЬОЮ ВЕРСІЄЮ
═══════════════════════════════════════════════════════════════════

1. ОЗНАКИ РАХУЮТЬСЯ ТИМ САМИМ КОДОМ, ЩО Й ПРИ НАВЧАННІ.
   Раніше radar.py мав власну реалізацію Mel-спектрограми, яка не
   збігалася з torchaudio у dataset.py (інші трикутні фільтри, інше
   вікно Ганна, відсутня нормалізація рівня). Модель на Pi отримувала
   не ті числа, на яких навчалась. Тепер обидва шляхи викликають
   features.MelFrontend.

2. КЛАСИФІКАЦІЯ БІЛЬШЕ НЕ ЗАЛЕЖИТЬ ВІД ГУЧНОСТІ.
   Датасет нормалізувався до піку 0.95, живий мікрофон — ні. Заміри
   показували: той самий сигнал, тихіший у 10 разів, давав P(дрон)
   0.047 → 0.000. Нормалізація спектрограми у features.py це усунула.

3. РІШЕННЯ ПРИЙМАЄТЬСЯ 2 РАЗИ НА СЕКУНДУ, А НЕ РАЗ НА 2 СЕКУНДИ.
   Було: блокуюче читання 2-секундного чанка, 5 підтверджень поспіль
   → тривога через 10 секунд після появи дрона. Потім: ковзне вікно
   2 с з кроком 0.5 с.

3b. ТРИВОГА ПІДНІМАЄТЬСЯ ЗА 1.0 с, А НЕ ЗА 3.5 с.
   Навіть після пункту 3 реальна затримка була 7 блоків, а не 5:
   EMA ймовірності стартувала з нуля, тому за p_drone = 1.0 згладжене
   значення давало 0.50 / 0.75 / 0.875 і перетинало поріг 0.775 аж на
   третьому блоці — і лише тоді починався відлік підтверджень.
   Тепер холодний старт EMA скориговано, а підтверджень потрібно 2:
   2 блоки × 0.5 с = 1.0 с. Див. DetectorTuning.

3c. КОРОТКА ВТРАТА СИГНАЛУ БІЛЬШЕ НЕ ГАСИТЬ ЦІЛЬ.
   З'явився явний стан ALARM_COASTING: тривога і останній достовірний
   напрямок утримуються 3.0 с після зникнення сигналу (маневр, вітер,
   перепона), і лише потім ціль скидається.

4. ПОРІГ ВЗЯТО З НАВЧАННЯ, А НЕ ЗІ СТЕЛІ.
   CONFIDENCE_MIN = 0.85 було константою «на око». Тепер поріг
   підбирається у train.py під задану частку хибних тривог і
   зберігається у model_config.json.

5. ШУМОВИЙ ГЕЙТ БІЛЬШЕ НЕ ГЛУШИТЬ ДАЛЕКИХ ДРОНІВ.
   NOISE_GATE = 0.03 по ПІКУ відкидав ціле вікно ще до інференсу.
   Тепер гейт працює по RMS у смузі дрона і стоїть на рівні цифрової
   тиші, тобто відсіює лише реально порожній сигнал.

6. НАПРЯМОК І ВІДСТАНЬ — див. doa.py та ranging.py.
   Якщо їх неможливо виміряти, показується «н/д», а не вигадане число.

Використання:
    python radar.py                 # консольний режим
    python radar.py --list-devices  # показати мікрофони
    python radar.py --debug         # додаткова діагностика
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import calibration
import features
from audio_io import InputConfig, open_stream, resolve_input, to_mono
from doa import DOAProvider, DOAReading
from features import MelFrontend
from latency import BUDGET
from ranging import RangeEstimate, RangeEstimator


# ═══════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).resolve().parent
ONNX_MODEL_PATH = str(BASE_DIR / "acoustic_radar.onnx")
MODEL_CONFIG    = BASE_DIR / "model_config.json"

CLASS_NAMES = {0: "Фон", 1: "ДРОН"}

# ── Ковзне вікно ──
BLOCK_SEC     = 0.5                                       # крок рішення
BLOCK_SAMPLES = int(features.SAMPLE_RATE * BLOCK_SEC)
WINDOW_SAMPLES = features.WINDOW_SAMPLES                  # 2 с

# ── Прийняття рішення ──
#
# ⚠️ УСІ ЧАСОВІ КОНСТАНТИ ТУТ ВИРАЖЕНІ В БЛОКАХ, А НЕ В СЕКУНДАХ.
# Один блок = BLOCK_SEC = 0.5 с. Перераховувати в секунди можна лише
# множенням на BLOCK_SEC — саме тому вони й лежать поруч із ним.
#
DEFAULT_THRESHOLD = 0.60   # використовується, лише якщо немає model_config.json
HOLD_FACTOR       = 0.70   # запасний гістерезис, якщо HOLD_THRESHOLD = None
PROB_EMA          = 0.5    # згладжування ймовірності (1.0 = без згладжування)

# P_HOLD — абсолютний поріг УТРИМАННЯ вже піднятої тривоги.
# Поріг ВХОДУ (P_START) береться з model_config.json (навчання, 0.775).
# Раніше поріг утримання рахувався як threshold·HOLD_FACTOR = 0.542;
# абсолютне значення задане явно, бо це операційне рішення, а не
# властивість моделі. None → повернутись до threshold·HOLD_FACTOR.
HOLD_THRESHOLD    = 0.50

# ⚠️ БУЛО 5 (≈2.5 с). Разом із «холодним» стартом EMA (див. Detector.update)
# реальна затримка тривоги складала 7 блоків = 3.5 с.
# 2 блоки × 0.5 с = 1.0 с — верхня межа замовленого діапазону 0.5–1.0 с.
#
# ЦІНА: поріг 0.775 підібрано під 2% хибних тривог НА ОДНЕ ВІКНО. Вимога
# двох послідовних вікон замість п'яти підвищує частоту хибних тривог.
# Сусідні вікна перекриваються на 1.5 с із 2 с, тобто вони сильно
# корельовані, і реальна частота лежить між 0.04% і 2% — виміряти її
# можна лише на реальному ефірі. Якщо хибних тривог стане забагато,
# піднімайте це число, а не поріг: поріг підібраний на валідації.
CONFIRM_THRESHOLD = 2      # підтверджень до тривоги (1.0 с при кроці 0.5 с)

# ⚠️ Тут була помилка на одиницю: перевірка `misses > MISS_TOLERANCE`
# скидала ціль на 7-му пропуску, тобто утримання тривало 3.5 с, а не 3.0 с,
# як стверджували коментарі тут і в fusion_config.AcousticConfig
# (`lost_after_s = 3.0`). Тепер перевірка `>=` і 6 блоків = рівно 3.0 с.
MISS_TOLERANCE    = 6      # пропусків до скидання (6 × 0.5 с = 3.0 с)

# ── Шумовий гейт ──
# Поріг у dBFS по RMS у смузі 80–4000 Гц. -65 дБ — це фактично цифрова
# тиша (від'єднаний мікрофон), а не «тихий двір».
GATE_DBFS = -65.0

# ── UART ──
UART_PORT = "/dev/serial0"
UART_BAUD = 115200
# Формат сумісний зі старою прошивкою приймача: "ANGLE:123\n".
UART_EXTENDED = False      # True → додатково "DRONE:conf,angle,dist\n"


# ═══════════════════════════════════════════════════════════════
#  Стан радара
# ═══════════════════════════════════════════════════════════════

#: Стани, у яких ціль вважається СУПРОВОДЖУВАНОЮ (є за чим стежити).
TRACKING_STATES = ("TRACK", "ALARM", "ALARM_COASTING")
#: Стани, у яких тривога ПІДНЯТА (свіжа або утримувана).
ALARM_STATES = ("ALARM", "ALARM_COASTING")


@dataclass
class DetectorTuning:
    """
    Налаштування автомата виявлення, у БЛОКАХ (1 блок = BLOCK_SEC = 0.5 с).

    Виділено в окремий об'єкт, щоб станція (fusion_config.AcousticConfig)
    могла їх перевизначити, не редагуючи radar.py, і щоб `python radar.py`
    без конфігурації поводився рівно так само, як станція за замовчуванням.
    """

    confirm_blocks: int = CONFIRM_THRESHOLD
    miss_tolerance: int = MISS_TOLERANCE
    prob_ema: float = PROB_EMA
    hold_factor: float = HOLD_FACTOR
    hold_threshold: float | None = HOLD_THRESHOLD

    # ⚠️ Головне джерело затримки тривоги, і воно НЕ було порогом.
    # EMA стартувала з нуля, тож навіть за p_drone = 1.0 згладжена
    # ймовірність давала 0.50 / 0.75 / 0.875 і перетинала поріг 0.775
    # лише на ТРЕТЬОМУ блоці. Разом із 5 підтвердженнями це 7 блоків =
    # 3.5 с. Тут це класичний зсув «холодного старту» EMA.
    # True → перший неглушений блок після тиші задає p_smoothed = p_drone
    # (стандартна корекція зсуву), тому лічильник підтверджень починає
    # рахувати одразу. Згладжування в усталеному режимі не змінюється:
    # від поодиноких спалахів захищає лічильник підтверджень, а не EMA.
    seed_ema_on_first_block: bool = True

    def hold_limit(self, threshold: float) -> float:
        """Поріг утримання вже піднятої тривоги (P_HOLD)."""
        if self.hold_threshold is not None:
            return float(self.hold_threshold)
        return float(threshold) * float(self.hold_factor)

    def confirm_seconds(self, block_sec: float = BLOCK_SEC) -> float:
        return self.confirm_blocks * float(block_sec)

    def coast_seconds(self, block_sec: float = BLOCK_SEC) -> float:
        return self.miss_tolerance * float(block_sec)


@dataclass
class RadarStatus:
    """Знімок стану — використовується і консоллю, і GUI."""
    state: str = "SLEEP"
    p_drone: float = 0.0
    p_smoothed: float = 0.0
    confirmations: int = 0
    misses: int = 0
    threshold: float = DEFAULT_THRESHOLD
    hold_threshold: float = 0.0
    confirm_needed: int = CONFIRM_THRESHOLD
    miss_tolerance: int = MISS_TOLERANCE
    level_dbfs: float = -120.0
    gated: bool = True
    angle_deg: float | None = None
    angle_confidence: float = 0.0
    angle_source: str = "none"
    angle_ambiguous: bool = False
    #: Конвенцію джерела виміряно, напрямок можна показувати на залізі.
    #: False = напрямок може бути дзеркальним — див. bearing_frame.
    angle_calibrated: bool = False
    angle_reason: str = ""
    range: RangeEstimate = field(default_factory=RangeEstimate)
    overflow: bool = False

    @property
    def is_alarm(self) -> bool:
        """Тривога піднята — включно з утриманням при короткій втраті."""
        return self.state in ALARM_STATES

    @property
    def is_coasting(self) -> bool:
        """Тривога утримується, але сигналу зараз немає."""
        return self.state == "ALARM_COASTING"

    @property
    def is_tracking(self) -> bool:
        return self.state in TRACKING_STATES


class Detector:
    """
    Скінченний автомат виявлення.

        SLEEP           — цифрова тиша (нижче шумового гейта)
        LISTEN          — слухаємо, дрона немає
        TRACK           — є підозра, накопичуємо підтвердження
        ALARM           — дрон підтверджено, сигнал є зараз
        ALARM_COASTING  — дрон був підтверджений, сигнал ТИМЧАСОВО зник

                     CLEAR (SLEEP/LISTEN)
                         │  p_smoothed >= P_START
                         ▼
                       TRACK  ──(confirm_blocks підтверджень)──►  ALARM
                                                                   │
                                              p_smoothed < P_HOLD  │
                                                                   ▼
                                                          ALARM_COASTING
                                                            │        │
                                        p_smoothed >= P_HOLD│        │miss_tolerance
                                                            ▼        ▼   блоків
                                                          ALARM    CLEAR

    ГІСТЕРЕЗИС. Щоб УВІЙТИ у TRACK, потрібно p_smoothed ≥ P_START (поріг
    із навчання, 0.775). Щоб УТРИМАТИ вже підтверджену ціль — досить
    P_HOLD (0.50). Без цього ціль на межі порогу блимає між станами.

    УТРИМАННЯ (coasting). Просідання ймовірності нижче P_HOLD НЕ скидає
    тривогу одразу: автомат переходить в ALARM_COASTING і тримає останній
    відомий стан ще miss_tolerance блоків. Це саме той інтервал, за який
    дрон встигає зробити маневр, сховатись за перепоною або потрапити під
    порив вітру. ALARM_COASTING — це і є те, що раніше було «ALARM з
    misses > 0»: окремого автомата не з'явилось, стан просто отримав ім'я,
    щоб LED-кільце, радар і камера могли показати його чесно.
    """

    def __init__(self, threshold: float,
                 tuning: DetectorTuning | None = None):
        self.threshold = threshold
        self.tuning = tuning if tuning is not None else DetectorTuning()
        self.state = "SLEEP"
        self.confirmations = 0
        self.misses = 0
        self.p_smoothed = 0.0
        # Чи бачив автомат хоч один неглушений блок з моменту скидання —
        # потрібно для корекції холодного старту EMA.
        self._ema_primed = False

    @property
    def hold_limit(self) -> float:
        return self.tuning.hold_limit(self.threshold)

    def reset(self) -> None:
        self.state = "LISTEN"
        self.confirmations = 0
        self.misses = 0
        self._ema_primed = False

    def update(self, p_drone: float, gated: bool) -> str:
        tun = self.tuning

        if gated:
            self.p_smoothed *= (1.0 - tun.prob_ema)
            if self.state in TRACKING_STATES:
                # Цифрова тиша — це теж пропуск. Підтверджена ціль іде в
                # утримання, непідтверджена (TRACK) просто накопичує
                # пропуски і скидається за тим самим лімітом.
                self.misses += 1
                if self.state in ALARM_STATES:
                    self.state = "ALARM_COASTING"
                else:
                    self.confirmations = 0      # див. _miss() нижче
                if self.misses >= tun.miss_tolerance:
                    self.reset()
                    self.state = "SLEEP"
            else:
                self.state = "SLEEP"
                self.confirmations = 0
                self._ema_primed = False
            return self.state

        # ── Експоненційне згладжування ──
        if tun.seed_ema_on_first_block and not self._ema_primed:
            # Корекція зсуву холодного старту: без неї перші 2-3 блоки
            # згладжена ймовірність механічно занижена, і тривога
            # затримується рівно на цей час (див. DetectorTuning).
            self.p_smoothed = float(p_drone)
        else:
            self.p_smoothed = (tun.prob_ema * p_drone
                               + (1.0 - tun.prob_ema) * self.p_smoothed)
        self._ema_primed = True

        # P_START для входу, P_HOLD для утримання вже підтвердженої цілі.
        # TRACK (ще не підтверджений) утримується на тому ж P_HOLD — так
        # було й раніше, і саме це дозволяє накопичити підтвердження на
        # сигналі, що коливається навколо порога.
        limit = (self.hold_limit if self.state in TRACKING_STATES
                 else self.threshold)

        if self.p_smoothed >= limit:
            self.confirmations += 1
            self.misses = 0
            self.state = ("ALARM" if self.confirmations >= tun.confirm_blocks
                          else "TRACK")
        else:
            if self.state in TRACKING_STATES:
                self.misses += 1
                if self.state in ALARM_STATES:
                    # Тривога вже піднята — утримуємо ціль. Лічильник
                    # підтверджень НЕ чіпаємо: він уже перевищив ліміт, і
                    # завдяки цьому повернення сигналу піднімає ALARM тим
                    # самим блоком, а не через ще confirm_blocks блоків.
                    self.state = "ALARM_COASTING"
                else:
                    # ⚠️ TRACK, тривоги ще не було. Підтвердження мають
                    # бути ПОСЛІДОВНИМИ, інакше сигнал, що стрибає навколо
                    # порога, накопичує їх через пропуски: один блок на
                    # 0.78, пропуск, ще один на 0.51 — і тривога піднята,
                    # хоча жодних двох блоків поспіль не було.
                    #
                    # Старий код теж не скидав лічильник, але там треба
                    # було 5 підтверджень, і зібрати їх «по одному через
                    # пропуск» практично не виходило. Із 2 підтвердженнями
                    # ця лазівка стає реальним джерелом хибних тривог,
                    # тому вона закрита разом зі зниженням порога.
                    self.confirmations = 0
                if self.misses >= tun.miss_tolerance:
                    self.reset()
            else:
                self.state = "LISTEN"
                self.confirmations = 0

        return self.state


# ═══════════════════════════════════════════════════════════════
#  Рушій
# ═══════════════════════════════════════════════════════════════

class RadarEngine:
    """
    Один екземпляр обробки: ознаки → модель → автомат → напрямок/відстань.

    ⚠️ Раніше цей код був продубльований у radar.py і radar_gui.py.
    Дублікати розійшлись (GUI не мав UART, інакше рахував відстань),
    тому тепер він один.
    """

    def __init__(self, model_path: str = ONNX_MODEL_PATH,
                 cfg: dict | None = None, verbose: bool = True,
                 tuning: DetectorTuning | None = None):
        import onnxruntime as ort

        self.cfg = cfg if cfg is not None else calibration.load()
        self.verbose = verbose
        self.tuning = tuning if tuning is not None else DetectorTuning()

        # ── Модель ──
        if not Path(model_path).exists():
            raise SystemExit(f"❌ Не знайдено модель: {model_path}\n"
                             f"   Спочатку навчіть її: python train.py")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2      # Pi: більше потоків не пришвидшує
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Конфіг моделі лежить поруч із самою моделлю (train.py пише їх
        # разом). Якщо модель узяли з іншої директорії — шукаємо там,
        # а не сліпо поруч із radar.py.
        sibling = Path(model_path).resolve().with_name(MODEL_CONFIG.name)
        self.model_config_path = sibling if sibling.exists() else MODEL_CONFIG

        self.threshold = self._load_threshold()
        self._check_model_matches_frontend()

        # ── Обробка ──
        self.frontend = MelFrontend()
        self.detector = Detector(self.threshold, self.tuning)
        self.ranger = RangeEstimator(self.cfg)

        self.buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._filled = 0
        self.doa: DOAProvider | None = None     # створюється у attach_input
        self.input: InputConfig | None = None

        self.status = RadarStatus(
            threshold=self.threshold,
            hold_threshold=self.detector.hold_limit,
            confirm_needed=self.tuning.confirm_blocks,
            miss_tolerance=self.tuning.miss_tolerance)

    # ── Ініціалізація ──────────────────────────────────────────

    def _load_threshold(self) -> float:
        """Читає робочий поріг, підібраний під час навчання."""
        if self.model_config_path.exists():
            try:
                cfg = json.loads(
                    self.model_config_path.read_text(encoding="utf-8"))
                return float(cfg.get("decision_threshold", DEFAULT_THRESHOLD))
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        if self.verbose:
            print(f"   ⚠️  Немає {MODEL_CONFIG.name} — поріг за замовчуванням "
                  f"{DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD

    def _check_model_matches_frontend(self) -> None:
        """
        Звіряє форму входу ONNX із поточним front-end.

        Без цієї перевірки невідповідність (наприклад, стара модель на
        32 mel проти нинішніх 64) або тихо ламала б результат, або
        падала б незрозумілою помилкою ONNX Runtime.
        """
        shape = self.session.get_inputs()[0].shape
        model_mels = shape[2] if len(shape) >= 3 else None

        if isinstance(model_mels, int) and model_mels != features.N_MELS:
            raise SystemExit(
                f"❌ Модель очікує {model_mels} Mel-каналів, а features.py "
                f"дає {features.N_MELS}.\n"
                f"   Це стара модель, навчена з іншим front-end — з нею "
                f"детекція завжди буде поганою.\n"
                f"   Перенавчіть: python mixer.py && python train.py")

        if self.model_config_path.exists():
            try:
                cfg = json.loads(
                    self.model_config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return
            mismatched = [
                (k, cfg[k], getattr(features, attr))
                for k, attr in (("n_mels", "N_MELS"), ("n_fft", "N_FFT"),
                                ("hop_length", "HOP_LENGTH"),
                                ("sample_rate", "SAMPLE_RATE"))
                if k in cfg and cfg[k] != getattr(features, attr)
            ]
            if mismatched:
                details = ", ".join(f"{k}: модель={a}, features.py={b}"
                                    for k, a, b in mismatched)
                raise SystemExit(
                    f"❌ Параметри front-end не збігаються з моделлю "
                    f"({details}).\n   Перенавчіть модель.")

    def attach_input(self, inp: InputConfig) -> None:
        """Прив'язує рушій до конкретного входу (потрібно для SRP-PHAT)."""
        self.input = inp
        self.doa = DOAProvider(self.cfg, inp.sample_rate,
                               len(inp.mic_channels) or inp.channels)
        if self.verbose:
            print(f"   {self.doa.describe()}")

    # ── Обробка одного блоку ───────────────────────────────────

    def process_block(self, block: np.ndarray) -> RadarStatus:
        """
        Args:
            block: [samples, channels] — черговий шматок аудіо (0.5 с)
        Returns:
            RadarStatus — актуальний стан
        """
        # Тільки мікрофонні канали — див. audio_io.to_mono.
        mono = to_mono(block, getattr(self.input, "mic_channels", None))

        # ── Ковзне вікно: зсуваємо буфер і дописуємо новий блок ──
        n = len(mono)
        if n >= WINDOW_SAMPLES:
            self.buffer[:] = mono[-WINDOW_SAMPLES:]
            self._filled = WINDOW_SAMPLES
        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = mono
            self._filled = min(WINDOW_SAMPLES, self._filled + n)

        st = self.status
        st.overflow = False

        # Поки вікно не заповнилось — рішення не приймаємо
        if self._filled < WINDOW_SAMPLES:
            st.state = "SLEEP"
            return st

        # ── Рівень і гейт ──
        level = self.ranger.level_dbfs(self.buffer)
        st.level_dbfs = level
        st.gated = level < GATE_DBFS

        # ── Інференс ──
        if st.gated:
            st.p_drone = 0.0
        else:
            _t = time.monotonic()
            mel = self.frontend(self.buffer)
            _t_feat = time.monotonic()
            logits = self.session.run(
                None, {self.input_name: MelFrontend.as_model_input(mel)})[0]
            exp = np.exp(logits[0] - logits[0].max())
            st.p_drone = float((exp / exp.sum())[1])
            # Measured, not estimated: these two are the only parts of the
            # chain that a faster machine would actually speed up.
            BUDGET.record("features", (_t_feat - _t) * 1000.0)
            BUDGET.record("inference", (time.monotonic() - _t_feat) * 1000.0)

        st.state = self.detector.update(st.p_drone, st.gated)
        st.p_smoothed = self.detector.p_smoothed
        st.confirmations = self.detector.confirmations
        st.misses = self.detector.misses

        # ── Напрямок ──
        # Кут вимірюється лише поки СИГНАЛ Є: рахувати напрямок на
        # порожній фон немає сенсу, а SRP-PHAT на шумі дасть випадкові
        # значення. Коли ціль втрачена, трек сам звільняється по таймауту.
        #
        # ⚠️ ALARM_COASTING сюди НЕ входить навмисно. Під час утримання
        # сигналу немає — новий вимір кута був би шумом, який зіпсував би
        # саме той останній достовірний напрямок, заради збереження якого
        # утримання й існує. Порожнє вимірювання лише «підживлює» трекер:
        # DOATracker.RELEASE_SEC = 6.0 с > 3.0 с утримання, тому останній
        # кут гарантовано доживає до кінця coasting без жодних припущень.
        if self.doa is not None:
            if st.state in ("TRACK", "ALARM"):
                # SRP-PHAT отримує САМЕ мікрофонні канали, у порядку
                # mic_positions_m. Передати сюди сирий блок означало б
                # рахувати геометрію по опорних каналах.
                self.doa.update(self._mic_block(block))
            else:
                self.doa.tracker.update(DOAReading(None))
                self.doa.canonical = self.doa._to_canonical(DOAReading(None))
            # ⚠️ КАНОНІЧНИЙ кут, а не сирий кут трекера. Перетворення
            # конвенції джерела виконує DOAProvider РІВНО ОДИН РАЗ; радар,
            # LED і камера далі не мають права нічого «довертати».
            canon = self.doa.canonical
            st.angle_deg = canon.deg
            st.angle_confidence = canon.confidence
            st.angle_source = canon.source
            st.angle_ambiguous = canon.ambiguous
            st.angle_calibrated = canon.calibrated
            st.angle_reason = canon.reason

        # ── Відстань ──
        if st.state in ("TRACK", "ALARM"):
            st.range = self.ranger.estimate(self.buffer)
        elif st.state == "ALARM_COASTING":
            # Сигнал зник — вимірювати рівень немає по чому, і оцінка
            # «раптом стало дуже далеко» була б артефактом тиші, а не
            # рухом цілі. Лишаємо ОСТАННЮ достовірну оцінку без змін;
            # те, що вона не свіжа, показує сам стан ALARM_COASTING.
            pass
        else:
            # Фон і тиша — оновлюємо оцінку шумової підлоги
            self.ranger.observe_background(self.buffer)
            self.ranger.reset_track()
            st.range = RangeEstimate(level_dbfs=level,
                                     noise_dbfs=self.ranger.noise.level_dbfs,
                                     reason="цілі немає")
        return st

    def _mic_block(self, block: np.ndarray) -> np.ndarray | None:
        """Підблок лише з мікрофонних каналів, або None якщо їх немає."""
        if block.ndim != 2:
            return None
        chans = getattr(self.input, "mic_channels", None)
        if not chans:
            return block
        valid = [c for c in chans if 0 <= c < block.shape[1]]
        if len(valid) < 2:
            return None
        return block[:, valid]

    def close(self) -> None:
        if self.doa is not None:
            self.doa.stop()


# ═══════════════════════════════════════════════════════════════
#  Консольний вивід
# ═══════════════════════════════════════════════════════════════

def format_status(st: RadarStatus, debug: bool = False) -> str:
    """Один рядок стану."""
    if st.state == "ALARM":
        header = "🆘 SOS ДРОН!!!"
    elif st.state == "ALARM_COASTING":
        header = "🆘 ДРОН — сигнал тимчасово зник"
    elif st.state == "TRACK":
        header = "👀 Схоже на дрона..."
    elif st.state == "LISTEN":
        header = "👂 Слухаю..."
    else:
        header = "💤 Тиша"

    parts = [header]

    if st.is_tracking:
        # Під час утримання ймовірність показується як ОСТАННЯ, а не як
        # поточна: стверджувати «сигнал зараз сильний», коли його немає,
        # означало б брехати оператору саме в той момент, коли він
        # вирішує, чи вірити напрямку на екрані.
        parts.append(f"p={st.p_smoothed:.0%}"
                     + (" (останнє)" if st.is_coasting else ""))
        parts.append(f"hits={st.confirmations}/{st.confirm_needed}")
        angle = ("н/д" if st.angle_deg is None
                 else f"{st.angle_deg:.0f}°"
                      + ("±" if st.angle_ambiguous else ""))
        parts.append(f"напрямок={angle}"
                     + (" (останній)" if st.is_coasting else ""))
        parts.append(f"відстань={st.range.format()}")
        if st.misses:
            parts.append(f"пропуск={st.misses}/{st.miss_tolerance}")
    else:
        parts.append(f"p={st.p_smoothed:.0%}")
        parts.append(f"рівень={st.level_dbfs:.0f}дБ")

    if debug:
        parts.append(f"[P_START={st.threshold:.2f} "
                     f"P_HOLD={st.hold_threshold:.2f} "
                     f"фон={st.range.noise_dbfs:.0f}дБ "
                     f"джерело={st.angle_source}]")

    return " | ".join(parts)


def print_status(text: str) -> None:
    """Динамічний однорядковий вивід (без спаму в термінал)."""
    cols = shutil.get_terminal_size().columns
    sys.stdout.write("\r" + text[:cols - 2].ljust(cols - 2))
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
#  UART
# ═══════════════════════════════════════════════════════════════

class UartSender:
    """Надсилає кут по UART. Формат сумісний зі старою версією."""

    MIN_INTERVAL = 1.0     # с — не частіше, щоб не забивати канал
    MIN_CHANGE_DEG = 5.0

    def __init__(self, port: str = UART_PORT, baud: int = UART_BAUD):
        self.serial = None
        self._last_time = 0.0
        self._last_angle: float | None = None
        try:
            import serial as pyserial
            self.serial = pyserial.Serial(port, baud, timeout=0.1)
            print(f"   ✅ UART: {port} @ {baud}")
        except Exception as exc:
            print(f"   ⚠️  UART недоступний ({exc}) — працюємо без нього")

    def send(self, st: RadarStatus) -> None:
        if self.serial is None or not st.is_alarm or st.angle_deg is None:
            return
        now = time.time()
        changed = (self._last_angle is None
                   or abs(st.angle_deg - self._last_angle) >= self.MIN_CHANGE_DEG)
        if now - self._last_time < self.MIN_INTERVAL and not changed:
            return
        try:
            self.serial.write(f"ANGLE:{int(round(st.angle_deg)) % 360:03d}\n"
                              .encode("ascii"))
            if UART_EXTENDED:
                dist = (f"{st.range.distance_m:.0f}" if st.range.ok else "NA")
                self.serial.write(
                    f"DRONE:{st.p_smoothed:.2f},"
                    f"{int(round(st.angle_deg)) % 360:03d},{dist}\n"
                    .encode("ascii"))
            self._last_time, self._last_angle = now, st.angle_deg
        except Exception:
            pass      # обрив UART не має валити радар

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()


# ═══════════════════════════════════════════════════════════════
#  Головний цикл
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Акустичний радар")
    parser.add_argument("--list-devices", action="store_true",
                        help="показати мікрофони і вийти")
    parser.add_argument("--debug", action="store_true",
                        help="додаткова діагностика у рядку стану")
    parser.add_argument("--no-uart", action="store_true")
    parser.add_argument("--model", default=ONNX_MODEL_PATH)
    args = parser.parse_args()

    if args.list_devices:
        from audio_io import print_input_devices
        print_input_devices()
        return

    cfg = calibration.load()

    print("=" * 66)
    print("🎯 Acoustic Radar v2.0")
    print("=" * 66)

    engine = RadarEngine(args.model, cfg)
    inp = resolve_input(cfg, features.SAMPLE_RATE)
    engine.attach_input(inp)

    tun = engine.tuning
    print(f"   Модель:       {Path(args.model).name}  "
          f"(P_START {engine.threshold:.3f}, "
          f"P_HOLD {engine.detector.hold_limit:.3f})")
    print(f"   Front-end:    {features.N_MELS} mel, вікно "
          f"{features.WINDOW_SEC} с, крок {BLOCK_SEC} с")
    print(f"   Час реакції:  блок {BLOCK_SEC * 1000:.0f} мс × "
          f"{tun.confirm_blocks} підтверджень = "
          f"{tun.confirm_seconds() * 1000:.0f} мс до тривоги; "
          f"утримання {tun.coast_seconds():.1f} с "
          f"({tun.miss_tolerance} блоків)")
    print(f"   Мікрофон:     {inp.describe()}")
    print(f"   Калібрування: {calibration.describe(cfg)}")

    if not engine.ranger.calibrated:
        print()
        print("   ℹ️  Відстань показуватиметься як «н/д», доки не виконано")
        print("      калібрування:  python calibrate.py range")
        print("      (Це навмисно: без вимірювання будь-які метри — вигадка.)")

    uart = None if args.no_uart else UartSender()

    print(f"\n🎤 Слухаю ефір... (Ctrl+C — зупинити)\n")

    stream = open_stream(inp, BLOCK_SAMPLES)
    overflow_count = 0
    try:
        with stream:
            while True:
                block, overflowed = stream.read(BLOCK_SAMPLES)
                if overflowed:
                    # ⚠️ Старий код робив `continue` і повністю пропускав
                    # блок разом з оновленням автомата. Дані вже прочитані —
                    # правильніше їх обробити і просто відзначити подію.
                    overflow_count += 1

                st = engine.process_block(np.asarray(block))
                if uart is not None:
                    uart.send(st)

                line = format_status(st, args.debug)
                if args.debug and overflow_count:
                    line += f" [overflow×{overflow_count}]"
                print_status(line)

    except KeyboardInterrupt:
        print("\n\n🛑 Радар зупинено.")
    finally:
        engine.close()
        if uart is not None:
            uart.close()
        if overflow_count:
            print(f"   ⚠️  Переповнень аудіобуфера: {overflow_count} "
                  f"(Pi не встигає — зменште навантаження)")
        print("   Bye! 👋")


if __name__ == "__main__":
    main()

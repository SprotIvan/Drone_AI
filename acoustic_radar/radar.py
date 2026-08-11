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
   → тривога через 10 секунд після появи дрона. Тепер ковзне вікно
   2 с з кроком 0.5 с → тривога за ~2.5 с.

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
DEFAULT_THRESHOLD = 0.60   # використовується, лише якщо немає model_config.json
HOLD_FACTOR       = 0.70   # гістерезис: утримання цілі на нижчому порозі
PROB_EMA          = 0.5    # згладжування ймовірності (1.0 = без згладжування)
CONFIRM_THRESHOLD = 5      # підтверджень до тривоги (≈2.5 с при кроці 0.5 с)
MISS_TOLERANCE    = 6      # пропусків до скидання (≈3 с)

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

@dataclass
class RadarStatus:
    """Знімок стану — використовується і консоллю, і GUI."""
    state: str = "SLEEP"
    p_drone: float = 0.0
    p_smoothed: float = 0.0
    confirmations: int = 0
    misses: int = 0
    threshold: float = DEFAULT_THRESHOLD
    level_dbfs: float = -120.0
    gated: bool = True
    angle_deg: float | None = None
    angle_confidence: float = 0.0
    angle_source: str = "none"
    angle_ambiguous: bool = False
    range: RangeEstimate = field(default_factory=RangeEstimate)
    overflow: bool = False

    @property
    def is_alarm(self) -> bool:
        return self.state == "ALARM"

    @property
    def is_tracking(self) -> bool:
        return self.state in ("TRACK", "ALARM")


class Detector:
    """
    Скінченний автомат виявлення.

    SLEEP  — цифрова тиша, ціль не шукається
    LISTEN — слухаємо, дрона немає
    TRACK  — є підозра, накопичуємо підтвердження
    ALARM  — дрон підтверджено

    Гістерезис: щоб УВІЙТИ у TRACK, потрібна ймовірність ≥ threshold;
    щоб ЛИШИТИСЬ — досить threshold·HOLD_FACTOR. Без цього ціль на
    межі порогу блимає між станами і тривога не встигає піднятись.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.state = "SLEEP"
        self.confirmations = 0
        self.misses = 0
        self.p_smoothed = 0.0

    def reset(self) -> None:
        self.state = "LISTEN"
        self.confirmations = 0
        self.misses = 0

    def update(self, p_drone: float, gated: bool) -> str:
        if gated:
            self.p_smoothed *= (1.0 - PROB_EMA)
            if self.state in ("TRACK", "ALARM"):
                self.misses += 1
                if self.misses > MISS_TOLERANCE:
                    self.reset()
                    self.state = "SLEEP"
            else:
                self.state = "SLEEP"
                self.confirmations = 0
            return self.state

        # Експоненційне згладжування — прибирає поодинокі спалахи
        self.p_smoothed = (PROB_EMA * p_drone
                           + (1.0 - PROB_EMA) * self.p_smoothed)

        hold = self.threshold * HOLD_FACTOR
        limit = hold if self.state in ("TRACK", "ALARM") else self.threshold

        if self.p_smoothed >= limit:
            self.confirmations += 1
            self.misses = 0
            self.state = ("ALARM" if self.confirmations >= CONFIRM_THRESHOLD
                          else "TRACK")
        else:
            if self.state in ("TRACK", "ALARM"):
                self.misses += 1
                if self.misses > MISS_TOLERANCE:
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
                 cfg: dict | None = None, verbose: bool = True):
        import onnxruntime as ort

        self.cfg = cfg if cfg is not None else calibration.load()
        self.verbose = verbose

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
        self.detector = Detector(self.threshold)
        self.ranger = RangeEstimator(self.cfg)

        self.buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._filled = 0
        self.doa: DOAProvider | None = None     # створюється у attach_input

        self.status = RadarStatus(threshold=self.threshold)

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
        self.doa = DOAProvider(self.cfg, inp.sample_rate, inp.channels)
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
        mono = to_mono(block)

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
            mel = self.frontend(self.buffer)
            logits = self.session.run(
                None, {self.input_name: MelFrontend.as_model_input(mel)})[0]
            exp = np.exp(logits[0] - logits[0].max())
            st.p_drone = float((exp / exp.sum())[1])

        st.state = self.detector.update(st.p_drone, st.gated)
        st.p_smoothed = self.detector.p_smoothed
        st.confirmations = self.detector.confirmations
        st.misses = self.detector.misses

        # ── Напрямок ──
        # Кут вимірюється лише поки ціль у супроводі: рахувати напрямок
        # на порожній фон немає сенсу, а SRP-PHAT на шумі дасть випадкові
        # значення. Коли ціль втрачена, трек сам звільняється по таймауту.
        if self.doa is not None:
            if st.is_tracking:
                self.doa.update(block if block.ndim == 2 else None)
            else:
                self.doa.tracker.update(DOAReading(None))
            st.angle_deg = self.doa.tracker.angle
            st.angle_confidence = self.doa.tracker.confidence
            st.angle_source = self.doa.tracker.source
            st.angle_ambiguous = self.doa.tracker.ambiguous

        # ── Відстань ──
        if st.is_tracking:
            st.range = self.ranger.estimate(self.buffer)
        else:
            # Фон і тиша — оновлюємо оцінку шумової підлоги
            self.ranger.observe_background(self.buffer)
            self.ranger.reset_track()
            st.range = RangeEstimate(level_dbfs=level,
                                     noise_dbfs=self.ranger.noise.level_dbfs,
                                     reason="цілі немає")
        return st

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
    elif st.state == "TRACK":
        header = "👀 Схоже на дрона..."
    elif st.state == "LISTEN":
        header = "👂 Слухаю..."
    else:
        header = "💤 Тиша"

    parts = [header]

    if st.is_tracking:
        parts.append(f"p={st.p_smoothed:.0%}")
        parts.append(f"hits={st.confirmations}/{CONFIRM_THRESHOLD}")
        angle = ("н/д" if st.angle_deg is None
                 else f"{st.angle_deg:.0f}°"
                      + ("±" if st.angle_ambiguous else ""))
        parts.append(f"напрямок={angle}")
        parts.append(f"відстань={st.range.format()}")
        if st.misses:
            parts.append(f"пропуск={st.misses}/{MISS_TOLERANCE}")
    else:
        parts.append(f"p={st.p_smoothed:.0%}")
        parts.append(f"рівень={st.level_dbfs:.0f}дБ")

    if debug:
        parts.append(f"[поріг={st.threshold:.2f} "
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

    print(f"   Модель:       {Path(args.model).name}  "
          f"(поріг {engine.threshold:.2f})")
    print(f"   Front-end:    {features.N_MELS} mel, вікно "
          f"{features.WINDOW_SEC} с, крок {BLOCK_SEC} с")
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

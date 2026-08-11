#!/usr/bin/env python3
"""
doa.py — Визначення напрямку на ціль (Direction of Arrival).

═══════════════════════════════════════════════════════════════════
⚠️ ЧОМУ РАДАР ПОКАЗУВАВ НЕПРАВИЛЬНИЙ КУТ
═══════════════════════════════════════════════════════════════════

БАГ №1 (головний) — будь-яка помилка мовчки перетворювалась на 90°.
    Старий get_usb_doa() мав `except Exception: return 90`. Якщо
    утиліта не знайдена, впала, або вивід не розпарсився — радар
    отримував рівно 90° і вважав це правдою. Саме тому кут «зазвичай
    неправильний»: він фактично сталий.
    → Тепер при збої повертається None і радар пише «н/д», а причина
      помилки друкується один раз.

БАГ №2 — парсинг ламався на реальному форматі виводу.
    `ast.literal_eval("0.273 3.688 2.993 2.993")` кидає SyntaxError:
    literal_eval розуміє лише "[0.273, 3.688]" з дужками і комами.
    xvf_host зазвичай друкує значення ЧЕРЕЗ ПРОБІЛ → виняток →
    див. баг №1 → завжди 90°.
    → Тепер значення витягуються регулярним виразом і формат не має
      значення (пробіли, коми, дужки — будь-що).

БАГ №3 — брався випадковий промінь: `vals[-1]`.
    AEC_AZIMUTH_VALUES повертає азимут КОЖНОГО з чотирьох променів AEC.
    Останній у списку не є «обраним» — це просто четвертий промінь.
    → Тепер вибирається найбільша узгоджена група променів (у прикладі
      2.993 повторюється двічі — це і є домінантне джерело), або
      конкретний індекс, якщо його задано у radar_calibration.json.

БАГ №4 — субпроцес запускався у головному аудіоциклі.
    Запуск python-інтерпретатора займає 200–500 мс. Це блокувало читання
    аудіо кожні 2 с → переповнення буфера → пропущені шматки звуку →
    гірша детекція.
    → Тепер опитування йде у власному потоці, аудіоцикл лише читає кеш.

БАГ №5 — не було жодної прив'язки до фізичної орієнтації масиву.
    Нуль градусів мікрофона майже ніколи не збігається з нулем на екрані.
    → Додано doa_offset_deg / doa_invert у radar_calibration.json
      (визначаються через `python calibrate.py doa`).

БАГ №6 — трекер намертво залипав на першому куті.
    SmartTracker запам'ятовував кут на початку захоплення і ігнорував
    усе, що відхилялось більше ніж на 50°. Якщо перший кут був хибним
    (див. баг №1 — а він був хибним завжди), ціль уже ніколи не могла
    «перетягнути» напрямок.
    → DOATracker тепер згладжує кут циркулярним середнім і перескакує
      на новий напрямок, якщо той підтверджується кілька разів поспіль.

Крім апаратного DOA реалізовано власний SRP-PHAT по сирих каналах —
він працює, якщо масив віддає 4 непроцесовані мікрофони.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPEED_OF_SOUND = 343.0   # м/с при +20 °C


# ═══════════════════════════════════════════════════════════════
#  Результат вимірювання
# ═══════════════════════════════════════════════════════════════

@dataclass
class DOAReading:
    """Одне вимірювання напрямку."""
    angle_deg: float | None      # None = напрямок невідомий
    confidence: float = 0.0      # 0..1
    source: str = "none"         # "usb" | "srp" | "none"
    ambiguous: bool = False      # True = можлива дзеркальна неоднозначність
    raw: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.angle_deg is not None


# ═══════════════════════════════════════════════════════════════
#  Розбір виводу xvf_host
# ═══════════════════════════════════════════════════════════════

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_azimuth_values(text: str, key: str = "AEC_AZIMUTH_VALUES"
                         ) -> list[float]:
    """
    Витягує числа з виводу xvf_host.

    Приймає будь-який із форматів:
        AEC_AZIMUTH_VALUES: [0.273, 3.688, 2.993, 2.993]
        AEC_AZIMUTH_VALUES 0.273 3.688 2.993 2.993
        aec_azimuth_values = 0.273,3.688,2.993,2.993

    Старий код використовував ast.literal_eval і падав на всьому,
    крім першого варіанта.
    """
    for line in text.splitlines():
        if key.lower() not in line.lower():
            continue
        tail = line[line.lower().index(key.lower()) + len(key):]
        values = [float(m) for m in _FLOAT_RE.findall(tail)]
        if values:
            return values
    return []


def azimuths_to_degrees(values: list[float]) -> list[float]:
    """
    Приводить азимути до градусів 0–359.

    Прошивки віддають або радіани (0..2π), або одразу градуси.
    Розрізняємо за діапазоном: якщо всі |значення| ≤ 2π — це радіани.
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if np.max(np.abs(arr)) <= 2.0 * math.pi + 0.2:
        arr = np.degrees(arr)
    return [float(a % 360.0) for a in arr]


def select_azimuth(degrees: list[float], beam_index: int = -1,
                   cluster_deg: float = 25.0) -> tuple[float | None, float]:
    """
    Обирає напрямок з чотирьох променів.

    beam_index >= 0 → беремо саме цей промінь.
    beam_index < 0  → шукаємо найбільшу узгоджену групу променів:
                      якщо кілька променів дивляться приблизно в один бік,
                      там і є домінантне джерело. Впевненість = частка
                      променів у цій групі.
    """
    if not degrees:
        return None, 0.0

    if 0 <= beam_index < len(degrees):
        return degrees[beam_index], 0.5

    best_members: list[float] = []
    for centre in degrees:
        members = [d for d in degrees
                   if abs((d - centre + 180.0) % 360.0 - 180.0) <= cluster_deg]
        if len(members) > len(best_members):
            best_members = members

    angle = circular_mean(best_members)
    confidence = len(best_members) / len(degrees)
    return angle, confidence


# ═══════════════════════════════════════════════════════════════
#  Циркулярна арифметика
# ═══════════════════════════════════════════════════════════════

def circular_mean(degrees: list[float],
                  weights: list[float] | None = None) -> float:
    """Середнє кутів. Звичайне арифметичне тут неправильне: (350°+10°)/2 = 180°."""
    if not degrees:
        return 0.0
    w = np.ones(len(degrees)) if weights is None else np.asarray(weights, float)
    rad = np.radians(degrees)
    deg = np.degrees(np.arctan2(np.sum(w * np.sin(rad)),
                                np.sum(w * np.cos(rad))))
    # Округлення прибирає похибку float: без нього середнє(350°, 10°)
    # дає 359.99999999999994 замість рівно 0°.
    return float(np.round(deg, 6) % 360.0)


def angular_diff(a: float, b: float) -> float:
    """Найкоротша різниця між кутами, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def apply_orientation(angle: float, offset_deg: float,
                      invert: bool) -> float:
    """Переводить кут масиву у кут установки (екран / компас)."""
    if invert:
        angle = -angle
    return (angle + offset_deg) % 360.0


# ═══════════════════════════════════════════════════════════════
#  Апаратний DOA через xvf_host (XVF3800)
# ═══════════════════════════════════════════════════════════════

_XVF_CANDIDATES = [
    "/home/pi/mk/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "~/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "~/mk/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "/opt/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
]


def find_xvf_host() -> Path | None:
    """
    Шукає xvf_host.py. Шлях можна задати змінною середовища XVF_HOST_PATH.

    Старий код мав ОДИН зашитий шлях; якщо він не збігався, радар мовчки
    працював з фіктивним кутом 90°.
    """
    env = os.environ.get("XVF_HOST_PATH")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()

    for candidate in _XVF_CANDIDATES:
        p = Path(candidate).expanduser()
        if p.exists():
            return p

    # Останній шанс — пошук у домашніх каталогах
    for home in (Path.home(), Path("/home")):
        try:
            for p in home.glob("**/python_control/xvf_host.py"):
                return p
        except (OSError, PermissionError):
            continue
    return None


class HardwareDOA:
    """
    Фонове опитування азимута у DSP мікрофонного масиву.

    Аудіоцикл ніколи не чекає на субпроцес — він читає останнє
    закешоване значення.
    """

    def __init__(self, script_path: str | Path | None = None,
                 interval: float = 0.35, beam_index: int = -1,
                 timeout: float = 3.0):
        self.script_path = (Path(script_path).expanduser() if script_path
                            else find_xvf_host())
        self.interval = interval
        self.beam_index = beam_index
        self.timeout = timeout

        self._lock = threading.Lock()
        self._reading = DOAReading(None, error="ще не опитано")
        self._stamp = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reported_error: str | None = None
        self.fail_count = 0

    # ── Життєвий цикл ──────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.script_path is not None and self.script_path.exists()

    def start(self) -> bool:
        if not self.available:
            with self._lock:
                self._reading = DOAReading(
                    None, error="xvf_host.py не знайдено "
                                "(задайте XVF_HOST_PATH)")
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading = self._query_once()
            with self._lock:
                self._reading = reading
                self._stamp = time.time()
            if reading.error and reading.error != self._reported_error:
                # Друкуємо кожну НОВУ помилку рівно один раз —
                # мовчазний провал і був причиною сталого кута 90°.
                print(f"\n⚠️  DOA (USB): {reading.error}")
                self._reported_error = reading.error
            self._stop.wait(self.interval)

    # ── Один запит ─────────────────────────────────────────────

    def _query_once(self) -> DOAReading:
        try:
            out = subprocess.run(
                [sys.executable, str(self.script_path), "AEC_AZIMUTH_VALUES"],
                capture_output=True, text=True, timeout=self.timeout,
                # ⚠️ xvf_host.py читає свої мапи команд відносно власної
                # директорії — без cwd він падає при запуску збоку.
                cwd=str(self.script_path.parent),
            )
        except subprocess.TimeoutExpired:
            self.fail_count += 1
            return DOAReading(None, error="таймаут запиту до масиву")
        except OSError as exc:
            self.fail_count += 1
            return DOAReading(None, error=f"не вдалось запустити: {exc}")

        text = (out.stdout or "") + "\n" + (out.stderr or "")
        values = parse_azimuth_values(text)

        if not values:
            self.fail_count += 1
            snippet = " ".join(text.split())[:120]
            return DOAReading(
                None, error=f"не вдалось розібрати вивід: «{snippet}»")

        degrees = azimuths_to_degrees(values)
        angle, conf = select_azimuth(degrees, self.beam_index)
        self.fail_count = 0
        return DOAReading(angle, confidence=conf, source="usb", raw=degrees)

    # ── Читання кешу ───────────────────────────────────────────

    def read(self, max_age: float = 2.0) -> DOAReading:
        with self._lock:
            reading, stamp = self._reading, self._stamp
        if reading.ok and time.time() - stamp > max_age:
            return DOAReading(None, error="дані застаріли")
        return reading


# ═══════════════════════════════════════════════════════════════
#  Власний SRP-PHAT по сирих каналах
# ═══════════════════════════════════════════════════════════════

class ArrayDOA:
    """
    SRP-PHAT: перебирає напрямки і шукає той, за якого затримки між
    мікрофонами найкраще узгоджуються з взаємною кореляцією.

    Працює, лише якщо звукова карта віддає СИРІ канали мікрофонів.
    Якщо масив уже все змікшував у оброблене стерео (типовий режим
    XVF3800 «2 канали»), канали будуть майже ідентичними — це виявляється
    і метод чесно повідомляє, що не може працювати.

    Точність з масивом 43 мм на 16 кГц — приблизно ±15°. Це грубо,
    але це РЕАЛЬНИЙ напрямок, а не константа.
    """

    def __init__(self, mic_positions: list[list[float]],
                 sample_rate: int,
                 band: tuple[float, float] = (200.0, 3500.0),
                 n_angles: int = 72, interp: int = 8):
        self.mics = np.asarray(mic_positions, dtype=np.float64)
        self.sample_rate = sample_rate
        self.band = band
        self.interp = interp
        self.n_mics = len(self.mics)

        self.angles = np.arange(0, 360, 360 // n_angles, dtype=np.float64)
        self.pairs = [(i, j) for i in range(self.n_mics)
                      for j in range(i + 1, self.n_mics)]

        # Максимальна можлива затримка (у семплах) для найдальшої пари
        max_dist = max(
            (np.linalg.norm(self.mics[i] - self.mics[j]) for i, j in self.pairs),
            default=0.0)
        self.max_lag = int(np.ceil(max_dist / SPEED_OF_SOUND * sample_rate)) + 2

        # Очікувані затримки для кожної пари та кожного напрямку.
        # tau_ij(θ) = -( (p_i - p_j) · u(θ) ) / c
        rad = np.radians(self.angles)
        unit = np.stack([np.cos(rad), np.sin(rad)], axis=1)     # [A, 2]
        self.expected_tau = np.stack(
            [-(self.mics[i] - self.mics[j]) @ unit.T / SPEED_OF_SOUND
             for i, j in self.pairs])                            # [P, A]

    # ── Допоміжне ──────────────────────────────────────────────

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
        spec[(freqs < self.band[0]) | (freqs > self.band[1])] = 0.0
        return np.fft.irfft(spec, n=len(x))

    def _gcc_phat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Взаємна кореляція з фазовим вирівнюванням (PHAT), інтерпольована
        у `interp` разів. Повертає вектор довжини 2*max_shift+1,
        центрований на нульовій затримці.
        """
        n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
        spec = np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n))
        # PHAT: лишаємо тільки фазу — робить пік вужчим і стійким до
        # спектральної форми джерела (реверберація, вітер).
        spec /= np.abs(spec) + 1e-12
        cc = np.fft.irfft(spec, n * self.interp)

        shift = self.max_lag * self.interp
        return np.concatenate((cc[-shift:], cc[:shift + 1]))

    @staticmethod
    def _sample_at(cc: np.ndarray, centre: int, idx: np.ndarray) -> np.ndarray:
        """Лінійна інтерполяція кореляції у дробових позиціях."""
        pos = np.clip(centre + idx, 0, len(cc) - 2)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        return cc[lo] * (1.0 - frac) + cc[lo + 1] * frac

    # ── Основний метод ─────────────────────────────────────────

    def channels_are_distinct(self, audio: np.ndarray,
                              threshold: float = 0.999) -> bool:
        """
        Перевіряє, що канали справді різні мікрофони, а не копії.

        Оброблене стерео від XVF3800 має майже ідентичні канали —
        рахувати по ньому DOA неможливо, і краще про це сказати,
        ніж видавати випадкові кути.
        """
        if audio.shape[1] < 2:
            return False
        a = audio[:, 0] - audio[:, 0].mean()
        b = audio[:, 1] - audio[:, 1].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            return False
        return abs(float(a @ b) / denom) < threshold

    def estimate(self, audio: np.ndarray) -> DOAReading:
        """
        Args:
            audio: ndarray [samples, channels] — сирі канали мікрофонів
        """
        n_ch = min(audio.shape[1], self.n_mics)
        if n_ch < 2:
            return DOAReading(None, error="потрібно щонайменше 2 канали")

        if not self.channels_are_distinct(audio):
            return DOAReading(
                None, error="канали ідентичні — масив віддає оброблене "
                            "стерео, SRP-PHAT неможливий")

        channels = [self._bandpass(audio[:, c].astype(np.float64))
                    for c in range(n_ch)]

        scores = np.zeros(len(self.angles))
        used = 0
        for k, (i, j) in enumerate(self.pairs):
            if i >= n_ch or j >= n_ch:
                continue
            cc = self._gcc_phat(channels[i], channels[j])
            centre = (len(cc) - 1) // 2
            idx = self.expected_tau[k] * self.sample_rate * self.interp
            scores += self._sample_at(cc, centre, idx)
            used += 1

        if used == 0:
            return DOAReading(None, error="немає придатних пар мікрофонів")

        best = int(np.argmax(scores))
        peak, floor_ = float(scores[best]), float(np.median(scores))
        spread = float(scores.max() - scores.min())
        confidence = 0.0 if spread < 1e-9 else min(
            1.0, max(0.0, (peak - floor_) / spread))

        # З двома мікрофонами напрямок визначається з точністю до
        # дзеркального відображення відносно осі масиву.
        ambiguous = n_ch < 3

        return DOAReading(float(self.angles[best]), confidence=confidence,
                          source="srp", ambiguous=ambiguous)


# ═══════════════════════════════════════════════════════════════
#  Згладжування та утримання напрямку
# ═══════════════════════════════════════════════════════════════

class DOATracker:
    """
    Згладжує потік вимірювань кута.

    Замість старої логіки «запам'ятати перший кут і відкидати все,
    що далі 50°» (яка намертво залипала на хибному початковому куті)
    тут:
      • поточний кут = зважене циркулярне середнє останніх вимірювань;
      • одиничний викид ігнорується;
      • але якщо новий напрямок підтверджується JUMP_CONFIRMATIONS разів
        поспіль — трек переходить на нього (ціль реально перелетіла
        або початкове захоплення було хибним);
      • без свіжих даних упевненість згасає і трек звільняється.
    """

    HISTORY = 8
    OUTLIER_DEG = 35.0
    JUMP_CONFIRMATIONS = 3
    RELEASE_SEC = 6.0

    def __init__(self):
        self._history: deque[tuple[float, float]] = deque(maxlen=self.HISTORY)
        self._pending: list[float] = []
        self._last_update = 0.0
        self.angle: float | None = None
        self.confidence = 0.0
        self.ambiguous = False
        self.source = "none"

    def reset(self) -> None:
        self._history.clear()
        self._pending.clear()
        self.angle = None
        self.confidence = 0.0
        self.source = "none"

    def update(self, reading: DOAReading, now: float | None = None) -> None:
        now = time.time() if now is None else now

        if not reading.ok or reading.confidence <= 0.0:
            # Немає даних — упевненість повільно згасає
            if self.angle is not None and now - self._last_update > self.RELEASE_SEC:
                self.reset()
            return

        angle = float(reading.angle_deg)
        self.ambiguous = reading.ambiguous
        self.source = reading.source

        if self.angle is None:
            self._accept(angle, reading.confidence, now)
            return

        if angular_diff(angle, self.angle) <= self.OUTLIER_DEG:
            self._pending.clear()
            self._accept(angle, reading.confidence, now)
            return

        # ── Викид: чекаємо підтвердження, перш ніж перескакувати ──
        self._pending.append(angle)
        if len(self._pending) >= self.JUMP_CONFIRMATIONS:
            recent = self._pending[-self.JUMP_CONFIRMATIONS:]
            centre = circular_mean(recent)
            if all(angular_diff(a, centre) <= self.OUTLIER_DEG for a in recent):
                # Новий напрямок стабільний — переходимо на нього
                self._history.clear()
                self._pending.clear()
                self._accept(centre, reading.confidence, now)
            else:
                self._pending.clear()

    def _accept(self, angle: float, weight: float, now: float) -> None:
        self._history.append((angle, max(weight, 0.05)))
        angles = [a for a, _ in self._history]
        weights = [w for _, w in self._history]
        self.angle = circular_mean(angles, weights)
        self.confidence = float(np.mean(weights))
        self._last_update = now

    def format(self) -> str:
        if self.angle is None:
            return "н/д"
        text = f"{self.angle:5.1f}°"
        if self.ambiguous:
            text += "±"          # дзеркальна неоднозначність (2 мікрофони)
        return text


# ═══════════════════════════════════════════════════════════════
#  Об'єднаний провайдер
# ═══════════════════════════════════════════════════════════════

class DOAProvider:
    """
    Обирає найкраще доступне джерело напрямку:
      1. апаратний азимут XVF3800 (найточніший — там 4 сирі мікрофони);
      2. власний SRP-PHAT по сирих каналах;
      3. якщо нічого не працює — чесне «н/д».
    """

    def __init__(self, cfg: dict, sample_rate: int, n_channels: int):
        self.offset = float(cfg.get("doa_offset_deg", 0.0))
        self.invert = bool(cfg.get("doa_invert", False))

        self.hardware = HardwareDOA(beam_index=int(cfg.get("doa_beam_index", -1)))
        self.hardware_ok = self.hardware.start()

        self.array = ArrayDOA(cfg.get("mic_positions_m"), sample_rate)
        self.array_ok = n_channels >= 2
        self._array_disabled_reason: str | None = None

        self.tracker = DOATracker()

    def describe(self) -> str:
        parts = []
        parts.append(f"USB DOA: {'✅ ' + str(self.hardware.script_path)}"
                     if self.hardware_ok else "USB DOA: ❌ недоступний")
        parts.append(f"SRP-PHAT: {'доступний' if self.array_ok else 'вимкнено'}")
        parts.append(f"зсув {self.offset:+.0f}°"
                     + (", дзеркально" if self.invert else ""))
        return " | ".join(parts)

    def update(self, audio: np.ndarray | None = None) -> DOAReading:
        """
        Args:
            audio: [samples, channels] — потрібно лише для SRP-PHAT
        """
        reading = self.hardware.read() if self.hardware_ok else DOAReading(None)

        if not reading.ok and self.array_ok and audio is not None \
                and self._array_disabled_reason is None:
            reading = self.array.estimate(audio)
            if not reading.ok and reading.error:
                # Немає сенсу рахувати SRP щоразу, якщо канали ідентичні
                if "ідентичні" in reading.error:
                    self._array_disabled_reason = reading.error
                    print(f"\n⚠️  DOA (SRP-PHAT): {reading.error}")

        if reading.ok:
            reading.angle_deg = apply_orientation(
                reading.angle_deg, self.offset, self.invert)

        self.tracker.update(reading)
        return reading

    def stop(self) -> None:
        self.hardware.stop()


# ═══════════════════════════════════════════════════════════════
#  Самотест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("🧭 doa.py — самотест")
    print("=" * 66)

    print("\n1. Розбір виводу xvf_host (те, на чому падав старий код):")
    samples = [
        "AEC_AZIMUTH_VALUES: [0.273, 3.688, 2.993, 2.993]",
        "AEC_AZIMUTH_VALUES 0.273 3.688 2.993 2.993",
        "aec_azimuth_values = 0.273,3.688,2.993,2.993",
        "AEC_AZIMUTH_VALUES:  0.273  3.688  2.993  2.993  ",
    ]
    for s in samples:
        vals = parse_azimuth_values(s)
        deg = azimuths_to_degrees(vals)
        angle, conf = select_azimuth(deg)
        print(f"   {s[:46]:<48s} → {angle:6.1f}°  (conf {conf:.2f})")

    import ast
    print("\n   Для порівняння — стара логіка ast.literal_eval:")
    for s in samples:
        try:
            vals = ast.literal_eval(s.split(":", 1)[1].strip())
            print(f"   {s[:46]:<48s} → {math.degrees(float(vals[-1])) % 360:.1f}°")
        except Exception as exc:
            print(f"   {s[:46]:<48s} → ❌ {type(exc).__name__} → "
                  f"старий код повертав 90°")

    print("\n2. Циркулярне середнє (звичайне середнє тут помиляється):")
    print(f"   середнє(350°, 10°) = {circular_mean([350.0, 10.0]):.1f}° "
          f"(арифметичне дало б 180°)")

    print("\n3. SRP-PHAT на синтетичній хвилі (квадрат 43 мм, 4 мікрофони):")
    sr = 16000
    mics = [[-0.0215, 0.0215], [0.0215, 0.0215],
            [0.0215, -0.0215], [-0.0215, -0.0215]]
    srp = ArrayDOA(mics, sr)
    rng = np.random.default_rng(0)

    for true_angle in (0.0, 45.0, 130.0, 250.0, 315.0):
        n = sr  # 1 секунда
        src = rng.standard_normal(n + 200)
        u = np.array([math.cos(math.radians(true_angle)),
                      math.sin(math.radians(true_angle))])
        channels = []
        for pos in mics:
            # Мікрофон, ближчий до джерела (p·u > 0), чує звук РАНІШЕ:
            # t_i = -p_i·u / c, тобто x_i[n] = s(n + p_i·u/c·sr).
            delay = np.dot(pos, u) / SPEED_OF_SOUND * sr
            idx = np.arange(n) + 100 + delay
            channels.append(np.interp(idx, np.arange(len(src)), src))
        audio = np.stack(channels, axis=1)
        audio += 0.01 * rng.standard_normal(audio.shape)

        r = srp.estimate(audio)
        err = angular_diff(r.angle_deg, true_angle) if r.ok else float("nan")
        print(f"   істина {true_angle:5.1f}° → оцінка {r.angle_deg:5.1f}°  "
              f"похибка {err:4.1f}°  conf={r.confidence:.2f}")

    print("\n4. Виявлення обробленого стерео (однакові канали):")
    mono = rng.standard_normal(16000)
    fake_stereo = np.stack([mono, mono], axis=1)
    r = ArrayDOA(mics[:2], sr).estimate(fake_stereo)
    print(f"   {r.error}")

    print("\n5. DOATracker — перескок на новий напрямок замість залипання:")
    tr = DOATracker()
    seq = [90.0] * 4 + [200.0] * 5
    for i, a in enumerate(seq):
        tr.update(DOAReading(a, confidence=0.8, source="test"), now=i * 0.5)
        print(f"   вимір {a:5.1f}° → трек {tr.format()}")
    print()

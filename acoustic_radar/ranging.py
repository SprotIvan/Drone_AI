#!/usr/bin/env python3
"""
ranging.py — Оцінка відстані до цілі за рівнем звуку.

═══════════════════════════════════════════════════════════════════
⚠️ ЧОМУ СТАРА ОЦІНКА ВІДСТАНІ БУЛА НЕПРАВИЛЬНОЮ
═══════════════════════════════════════════════════════════════════

Був один рядок:

    dist_m = int(min(200, 6.0 / max(peak_volume, 1e-4)))

Проблеми, кожна з яких сама по собі робить результат недійсним:

  1. Константа 6.0 взята зі стелі. Вона означає «на 1.0 повної шкали
     дрон за 6 метрів», але це залежить від чутливості мікрофона,
     підсилення драйвера і гучності конкретного дрона. Ніде не
     вимірювалось — тому метри були вигаданими.

  2. peak_volume — це ПІК широкосмугового сигналу за 2 секунди.
     Один порив вітру, ляскіт або клацання задає пік — і «відстань»
     стрибає, хоча дрон нікуди не полетів. Треба міряти стаціонарний
     RMS у смузі дрона.

  3. Не віднімався фон. У вітряну погоду фон сам по собі дає
     peak ≈ 0.2, тобто «36 метрів» навіть коли в небі порожньо.

  4. Закон 1/r рахувався від ПОВНОГО рівня, а не від рівня ЦІЛІ.

Що робить цей модуль:
  • міряє RMS у смузі 80–4000 Гц (де живе підпис мультикоптера);
  • віднімає поточний рівень фону (енергетично, а не в дБ);
  • переводить надлишок рівня у відстань за калібрувальною точкою
    «рівень L_ref на відстані r_ref», яку задає calibrate.py;
  • згладжує в логарифмічній шкалі та повертає ІНТЕРВАЛ, а не одне
    число, бо акустична далекометрія одним мікрофоном принципово
    неточна (реально ±40–60%);
  • якщо калібрування немає — повертає None і причину, а не вигадані
    метри.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from features import DRONE_BAND_HZ, SAMPLE_RATE, band_rms, to_db


# ═══════════════════════════════════════════════════════════════
#  Результат
# ═══════════════════════════════════════════════════════════════

def target_level_dbfs(level_dbfs: float, noise_dbfs: float
                      ) -> tuple[float, float]:
    """
    Виділяє рівень ЦІЛІ з повного рівня, віднімаючи фон.

    Віднімати треба ПОТУЖНОСТІ, а не децибели: якщо ціль і фон однакові
    за рівнем, повний рівень на 3 дБ вищий за кожен із них, а не вдвічі.

    ⚠️ Ця функція навмисно спільна для RangeEstimator і calibrate.py.
    Якщо калібрувальна точка вимірюється інакше, ніж робочий рівень,
    усі відстані зміщуються — саме такого роду розбіжність і зробила
    стару оцінку відстані непридатною.

    Returns:
        (рівень цілі у dBFS, надлишок над фоном у дБ)
    """
    p_total = 10.0 ** (level_dbfs / 10.0)
    p_noise = 10.0 ** (noise_dbfs / 10.0)
    p_target = p_total - p_noise

    if p_target <= 0.0 or p_noise <= 0.0:
        return -120.0, -99.0

    return (10.0 * float(np.log10(p_target)),
            10.0 * float(np.log10(p_target / p_noise)))


@dataclass
class RangeEstimate:
    """Оцінка відстані з інтервалом невизначеності."""
    distance_m: float | None = None
    lo_m: float | None = None
    hi_m: float | None = None
    level_dbfs: float = -120.0        # Повний рівень у смузі
    noise_dbfs: float = -120.0        # Оцінений фон
    excess_db: float = 0.0            # Рівень цілі над фоном
    reason: str = ""                  # Чому None, якщо None

    @property
    def ok(self) -> bool:
        return self.distance_m is not None

    def format(self) -> str:
        if not self.ok:
            return "н/д"
        if self.lo_m is not None and self.hi_m is not None:
            return f"~{self.distance_m:.0f} м ({self.lo_m:.0f}–{self.hi_m:.0f})"
        return f"~{self.distance_m:.0f} м"


# ═══════════════════════════════════════════════════════════════
#  Стеження за рівнем фону
# ═══════════════════════════════════════════════════════════════

class NoiseFloorTracker:
    """
    Оцінює рівень фону як низький перцентиль недавніх вимірювань.

    Перцентиль, а не мінімум: мінімум чіпляється за одну аномально тиху
    паузу і потім занижує фон назавжди. Перцентиль стійкий і повільно
    адаптується до зміни погоди.
    """

    def __init__(self, history: int = 240, percentile: float = 20.0):
        self._levels: deque[float] = deque(maxlen=history)
        self.percentile = percentile
        self._seed: float | None = None

    def seed(self, level_dbfs: float) -> None:
        """Початкове значення з калібрування (щоб не чекати накопичення)."""
        self._seed = level_dbfs

    def update(self, level_dbfs: float) -> None:
        self._levels.append(level_dbfs)

    @property
    def level_dbfs(self) -> float:
        if len(self._levels) >= 8:
            return float(np.percentile(self._levels, self.percentile))
        if self._seed is not None:
            return self._seed
        if self._levels:
            return float(np.min(self._levels))
        return -120.0

    @property
    def ready(self) -> bool:
        return len(self._levels) >= 8 or self._seed is not None


# ═══════════════════════════════════════════════════════════════
#  Оцінювач відстані
# ═══════════════════════════════════════════════════════════════

class RangeEstimator:
    """
    Використання:
        est = RangeEstimator(cfg)
        est.observe_background(audio)      # коли цілі немає
        r = est.estimate(audio)            # коли ціль підтверджено
    """

    # Мінімальний надлишок над фоном, за якого рівень взагалі щось означає
    MIN_EXCESS_DB = 1.5
    # Припущена невизначеність вимірювання рівня (орієнтація дрона,
    # вітер, тип апарата). Задає ширину інтервалу.
    LEVEL_UNCERTAINTY_DB = 6.0
    # Межі здорового глузду
    MIN_DISTANCE_M = 2.0
    MAX_DISTANCE_M = 500.0
    SMOOTH_HISTORY = 5

    def __init__(self, cfg: dict, sample_rate: int = SAMPLE_RATE,
                 band: tuple[float, float] = DRONE_BAND_HZ):
        self.sample_rate = sample_rate
        self.band = band

        self.ref_distance = cfg.get("range_ref_distance_m")
        self.ref_level = cfg.get("range_ref_level_dbfs")
        self.spreading_db = float(cfg.get("range_spreading_db", 22.0))

        self.noise = NoiseFloorTracker()
        if cfg.get("noise_floor_dbfs") is not None:
            self.noise.seed(float(cfg["noise_floor_dbfs"]))

        self._history: deque[float] = deque(maxlen=self.SMOOTH_HISTORY)

    # ── Стан ───────────────────────────────────────────────────

    @property
    def calibrated(self) -> bool:
        return self.ref_distance is not None and self.ref_level is not None

    def level_dbfs(self, audio: np.ndarray) -> float:
        """Рівень у смузі дрона (dBFS)."""
        return to_db(band_rms(audio, self.sample_rate, self.band))

    def observe_background(self, audio: np.ndarray) -> None:
        """Викликати, коли ціль НЕ виявлено — оновлює оцінку фону."""
        self.noise.update(self.level_dbfs(audio))

    def reset_track(self) -> None:
        """Скидає згладжування при втраті цілі."""
        self._history.clear()

    # ── Головний метод ─────────────────────────────────────────

    def estimate(self, audio: np.ndarray) -> RangeEstimate:
        level = self.level_dbfs(audio)
        noise = self.noise.level_dbfs

        # Рівень цілі = повна потужність мінус потужність фону
        target_level, excess = target_level_dbfs(level, noise)

        result = RangeEstimate(level_dbfs=level, noise_dbfs=noise,
                               excess_db=float(excess))

        if not self.calibrated:
            result.reason = ("не калібровано — запустіть "
                             "python calibrate.py range")
            return result

        if not self.noise.ready:
            result.reason = "накопичується оцінка фону"
            return result

        if excess < self.MIN_EXCESS_DB:
            result.reason = "ціль на рівні фону — дальність не визначити"
            return result

        # ── Закон згасання ──
        #   L = L_ref - spreading·log10(r / r_ref)
        # → r = r_ref · 10^((L_ref - L) / spreading)
        # spreading = 20 дБ/декаду у вільному полі; більше — з урахуванням
        # поглинання в повітрі та над ґрунтом.
        def to_distance(level_db: float) -> float:
            exponent = (self.ref_level - level_db) / self.spreading_db
            return float(self.ref_distance * (10.0 ** exponent))

        raw = to_distance(target_level)
        self._history.append(np.log10(max(raw, 1e-3)))

        # Медіана в лог-шкалі — стійка до поодиноких стрибків
        smoothed = float(10.0 ** np.median(self._history))
        smoothed = float(np.clip(smoothed, self.MIN_DISTANCE_M,
                                 self.MAX_DISTANCE_M))

        # Інтервал невизначеності з похибки вимірювання рівня
        lo = to_distance(target_level + self.LEVEL_UNCERTAINTY_DB)
        hi = to_distance(target_level - self.LEVEL_UNCERTAINTY_DB)
        scale = smoothed / max(raw, 1e-6)

        result.distance_m = smoothed
        result.lo_m = float(np.clip(lo * scale, self.MIN_DISTANCE_M,
                                    self.MAX_DISTANCE_M))
        result.hi_m = float(np.clip(hi * scale, self.MIN_DISTANCE_M,
                                    self.MAX_DISTANCE_M))
        return result


# ═══════════════════════════════════════════════════════════════
#  Самотест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("📏 ranging.py — самотест")
    print("=" * 66)

    rng = np.random.default_rng(0)
    sr = SAMPLE_RATE
    t = np.arange(2 * sr) / sr

    def drone_at(distance_m: float, ref_d=10.0, ref_amp=0.20,
                 spreading=22.0) -> np.ndarray:
        """Синтетичний дрон із фізично коректним згасанням + фон."""
        amp = ref_amp * 10.0 ** (-spreading * np.log10(distance_m / ref_d) / 20.0)
        sig = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 25))
        sig = sig / np.abs(sig).max() * amp
        background = 0.004 * rng.standard_normal(t.size)
        return (sig + background).astype(np.float32)

    # ── Без калібрування ──
    est = RangeEstimator({})
    for _ in range(10):
        est.observe_background(0.004 * rng.standard_normal(t.size).astype(np.float32))
    r = est.estimate(drone_at(50.0))
    print(f"\n1. Без калібрування → {r.format()}   ({r.reason})")
    print("   Старий код на цьому місці впевнено друкував вигадані метри.")

    # ── З калібруванням ──
    ref_d, ref_amp = 10.0, 0.20
    ref_level = to_db(band_rms(drone_at(ref_d, ref_d, ref_amp)))
    cfg = {"range_ref_distance_m": ref_d, "range_ref_level_dbfs": ref_level,
           "range_spreading_db": 22.0}
    print(f"\n2. Калібрувальна точка: {ref_level:.1f} dBFS @ {ref_d:.0f} м")

    print(f"\n   {'істина':>8} │ {'оцінка':>22} │ {'рівень':>9} │ "
          f"{'над фоном':>9} │ похибка")
    print(f"   {'─' * 8}─┼─{'─' * 22}─┼─{'─' * 9}─┼─{'─' * 9}─┼────────")
    for true_d in (5.0, 15.0, 40.0, 80.0, 150.0, 250.0):
        est = RangeEstimator(cfg)
        for _ in range(12):
            est.observe_background(
                0.004 * rng.standard_normal(t.size).astype(np.float32))
        for _ in range(5):
            r = est.estimate(drone_at(true_d, ref_d, ref_amp))
        if r.ok:
            err = abs(r.distance_m - true_d) / true_d
            print(f"   {true_d:7.0f}м │ {r.format():>22} │ "
                  f"{r.level_dbfs:8.1f}дБ │ {r.excess_db:8.1f}дБ │ {err:6.0%}")
        else:
            print(f"   {true_d:7.0f}м │ {'н/д':>22} │ "
                  f"{r.level_dbfs:8.1f}дБ │ {r.excess_db:8.1f}дБ │ "
                  f"{r.reason}")

    # ── Стійкість до поривів вітру ──
    print("\n3. Стійкість до пориву вітру (старий код рахував ПІК):")
    est = RangeEstimator(cfg)
    for _ in range(12):
        est.observe_background(0.004 * rng.standard_normal(t.size).astype(np.float32))
    clean = drone_at(60.0, ref_d, ref_amp)
    gust = clean.copy()
    gust[8000:8400] += 0.6      # різкий клац/порив
    r_clean = est.estimate(clean)
    est2 = RangeEstimator(cfg)
    est2.noise = est.noise
    r_gust = est2.estimate(gust)
    old_clean = min(200, 6.0 / max(float(np.abs(clean).max()), 1e-4))
    old_gust = min(200, 6.0 / max(float(np.abs(gust).max()), 1e-4))
    print(f"   без пориву:  новий {r_clean.format():>20} │ "
          f"старий ~{old_clean:.0f} м")
    print(f"   з поривом:   новий {r_gust.format():>20} │ "
          f"старий ~{old_gust:.0f} м  ← стрибок на порожньому місці")
    print()

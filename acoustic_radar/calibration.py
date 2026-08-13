#!/usr/bin/env python3
"""
calibration.py — Зберігання калібрувальних констант радара.

Розділення відповідальності:
  model_config.json        — параметри моделі (пише train.py)
  radar_calibration.json   — параметри конкретної УСТАНОВКИ (цей модуль)

Друге не можна вгадати з коду: воно залежить від того, як фізично
змонтовано мікрофонний масив, який у нього підсилювач і в яку сторону
дивиться «нуль градусів».

Саме відсутність цього файлу була причиною того, що радар показував
нісенітні відстані та кути: у старому коді константи були зашиті
«зі стелі» (`dist = 6.0 / peak_volume`, кут за замовчуванням 90°).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CALIBRATION_PATH = Path(__file__).with_name("radar_calibration.json")


# ═══════════════════════════════════════════════════════════════
#  Значення за замовчуванням
# ═══════════════════════════════════════════════════════════════

DEFAULTS: dict[str, Any] = {
    # ── Напрямок (DOA) ────────────────────────────────────────
    # Куди дивиться «0°» масиву відносно півночі/фронту установки.
    "doa_offset_deg": 0.0,
    # True, якщо кути масиву зростають проти годинникової стрілки,
    # а на екрані/компасі потрібен рух за годинниковою.
    "doa_invert": False,
    # Індекс променя AEC_AZIMUTH_VALUES, що відповідає обраному напрямку.
    # -1 = вибрати промінь автоматично (за узгодженістю показів).
    "doa_beam_index": -1,
    # Геометрія масиву: позиції мікрофонів у метрах (x, y).
    # За замовчуванням — квадрат зі стороною 43 мм (ReSpeaker 4-Mic).
    #
    # ⚠️ PHYSICAL HARDWARE VERIFICATION REQUIRED. Це ПРИПУЩЕННЯ про плату,
    # а не вимір. Від нього залежить уся геометрія SRP-PHAT: якщо реальний
    # масив має інший розмір або інший порядок мікрофонів, кути будуть
    # стабільно неправильними, і жоден зсув чи дзеркалення це не виправить.
    "mic_positions_m": [[-0.0215, 0.0215], [0.0215, 0.0215],
                        [0.0215, -0.0215], [-0.0215, -0.0215]],
    # Які канали звукового пристрою є ФІЗИЧНИМИ мікрофонами, у порядку
    # mic_positions_m. Порожньо = перші N каналів (теж припущення).
    # На 6-канальному XVF3800 два канали, за коментарем audio_io, не є
    # мікрофонами — якщо це так, вони не повинні потрапляти ні у SRP-PHAT,
    # ні у вхід класифікатора. Перевіряється постукуванням по кожному
    # мікрофону: `python calibrate.py check` друкує рівні по каналах.
    "mic_channels": [],

    # ── Конвенція напрямку (див. bearing_frame.py) ────────────
    # Джерело може розходитись із канонічним кадром ДВОМА незалежними
    # способами: поворотом (один параметр) і НАПРЯМКОМ ОБЕРТАННЯ (один
    # біт). Дзеркальність НЕ виправляється зсувом — див. доведення в
    # bearing_frame. Тому в кожного джерела свої параметри.
    #
    # doa_offset_deg / doa_invert описують USB DSP (XVF3800).
    # srp_zero_deg описує власний SRP-PHAT; його напрямок обертання
    # ВІДОМИЙ із коду (проти годинникової), тому окремого біта не треба.
    # None = не калібровано, і напрямок позначається як UNCAL.
    "srp_zero_deg": None,
    # Явний запис напрямку обертання USB DSP: "CW" | "CCW".
    # None = взяти зі старого ключа doa_invert.
    "doa_handedness": None,

    # ── Відстань (ranging) ────────────────────────────────────
    # Рівень дрона у смузі 80–4000 Гц, виміряний на відомій відстані.
    # None означає «не відкалібровано» — radar.py тоді чесно пише «н/д»,
    # замість того щоб вигадувати метри.
    "range_ref_distance_m": None,
    "range_ref_level_dbfs": None,
    # Показник згасання: 20 дБ/декаду = сферичне розходження у
    # вільному полі. Над землею з поглинанням реально 22–26.
    "range_spreading_db": 22.0,
    # Виміряний рівень фонової тиші (dBFS) — віднімається з рівня цілі.
    "noise_floor_dbfs": None,

    # ── Аудіо-вхід ────────────────────────────────────────────
    # Підрядок імені пристрою запису. Порожньо = системний за замовчуванням.
    "input_device": "ReSpeaker",
    "input_channels": None,   # None = визначити автоматично
}


# ═══════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════

def load(path: Path | str = CALIBRATION_PATH) -> dict[str, Any]:
    """Читає калібрування, доповнюючи відсутні ключі значеннями за замовчуванням."""
    cfg = dict(DEFAULTS)
    path = Path(path)
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"⚠️  Не вдалось прочитати {path}: {exc} — "
                  f"використано значення за замовчуванням")
    return cfg


def save(cfg: dict[str, Any], path: Path | str = CALIBRATION_PATH) -> None:
    """Записує калібрування (зливаючи з наявним файлом)."""
    path = Path(path)
    merged = load(path)
    merged.update(cfg)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"✅ Калібрування збережено: {path}")


def is_range_calibrated(cfg: dict[str, Any]) -> bool:
    """Чи можна взагалі рахувати відстань у метрах."""
    return (cfg.get("range_ref_distance_m") is not None
            and cfg.get("range_ref_level_dbfs") is not None)


def describe(cfg: dict[str, Any]) -> str:
    """Короткий людський опис стану калібрування."""
    lines = []
    if is_range_calibrated(cfg):
        lines.append(f"відстань: калібровано "
                     f"({cfg['range_ref_level_dbfs']:.1f} dBFS "
                     f"@ {cfg['range_ref_distance_m']:.0f} м)")
    else:
        lines.append("відстань: НЕ калібровано (буде показано «н/д»)")

    lines.append(f"кут: зсув {cfg['doa_offset_deg']:+.0f}°"
                 f"{', дзеркально' if cfg['doa_invert'] else ''}")
    return "; ".join(lines)


if __name__ == "__main__":
    cfg = load()
    print(f"Файл: {CALIBRATION_PATH}")
    print(f"Існує: {CALIBRATION_PATH.exists()}")
    print(f"Стан: {describe(cfg)}")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

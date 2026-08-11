#!/usr/bin/env python3
"""
calibrate.py — Прив'язка радара до конкретної установки.

Без цього кроку відстань і кут не можуть бути правильними в принципі:
  • метри залежать від чутливості мікрофона і підсилення драйвера —
    їх треба ОДИН РАЗ виміряти по дрону на відомій відстані;
  • нуль градусів масиву майже ніколи не збігається з нулем на екрані.

Стара версія радара цих вимірювань не мала і підставляла константи
(`6.0 / peak` для відстані, `90°` для кута) — саме тому показання були
неправильними «зазвичай».

Команди:
    python calibrate.py check      # діагностика: мікрофон, модель, DOA
    python calibrate.py devices    # список мікрофонів
    python calibrate.py noise      # виміряти рівень фону (без дрона)
    python calibrate.py range      # виміряти рівень дрона на відомій відстані
    python calibrate.py doa        # визначити зсув кута масиву
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

import calibration
import features
from audio_io import open_stream, print_input_devices, resolve_input, to_mono
from features import band_rms, to_db
from ranging import target_level_dbfs


BLOCK_SEC = 0.5
BLOCK_SAMPLES = int(features.SAMPLE_RATE * BLOCK_SEC)


# ═══════════════════════════════════════════════════════════════
#  Запис із індикатором
# ═══════════════════════════════════════════════════════════════

def record(seconds: float, cfg: dict, label: str) -> np.ndarray:
    """Записує аудіо, показуючи рівень у реальному часі."""
    inp = resolve_input(cfg, features.SAMPLE_RATE)
    print(f"   Мікрофон: {inp.describe()}")

    blocks: list[np.ndarray] = []
    n_blocks = int(math.ceil(seconds / BLOCK_SEC))

    stream = open_stream(inp, BLOCK_SAMPLES)
    with stream:
        for i in range(n_blocks):
            block, _ = stream.read(BLOCK_SAMPLES)
            block = np.asarray(block)
            blocks.append(block)

            level = to_db(band_rms(to_mono(block)))
            bars = int(np.clip((level + 80) / 80 * 34, 0, 34))
            sys.stdout.write(
                f"\r   {label} {i * BLOCK_SEC:5.1f}/{seconds:.0f}с  "
                f"[{'█' * bars}{'·' * (34 - bars)}] {level:6.1f} dBFS")
            sys.stdout.flush()
    sys.stdout.write("\n")
    return np.concatenate(blocks, axis=0)


def window_levels(audio: np.ndarray, window_sec: float = 2.0) -> np.ndarray:
    """Рівень у смузі дрона по 2-секундних вікнах."""
    mono = to_mono(audio)
    n = int(window_sec * features.SAMPLE_RATE)
    return np.array([to_db(band_rms(mono[i:i + n]))
                     for i in range(0, max(len(mono) - n + 1, 1), n // 2)])


# ═══════════════════════════════════════════════════════════════
#  check — загальна діагностика
# ═══════════════════════════════════════════════════════════════

def cmd_check(cfg: dict) -> None:
    print("=" * 66)
    print("🩺 Діагностика акустичного радара")
    print("=" * 66)

    ok = True

    # ── 1. Модель ──
    print("\n1. Модель")
    try:
        from radar import RadarEngine
        engine = RadarEngine(verbose=False)
        print(f"   ✅ ONNX завантажено, поріг {engine.threshold:.2f}")
    except SystemExit as exc:
        print(f"   ❌ {exc}")
        engine = None
        ok = False

    # ── 2. Реакція моделі на синтетику ──
    if engine is not None:
        print("\n2. Реакція моделі на тестові сигнали")
        rng = np.random.default_rng(0)
        t = np.arange(features.WINDOW_SAMPLES) / features.SAMPLE_RATE

        drone = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 30))
        drone = (drone + 0.3 * rng.standard_normal(t.size)).astype(np.float32)
        drone /= np.abs(drone).max()

        noise = rng.standard_normal(t.size).astype(np.float32)
        noise /= np.abs(noise).max()

        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        for name, sig in (("синтетичний дрон", drone),
                          ("білий шум", noise),
                          ("чистий тон 440 Гц", tone)):
            probs = []
            for gain in (1.0, 0.2, 0.03):
                mel = engine.frontend(sig * gain)
                logits = engine.session.run(
                    None, {engine.input_name:
                           engine.frontend.as_model_input(mel)})[0]
                e = np.exp(logits[0] - logits[0].max())
                probs.append(float((e / e.sum())[1]))
            spread = max(probs) - min(probs)
            print(f"   {name:<20s} P(дрон) = "
                  + ", ".join(f"{p:.2f}" for p in probs)
                  + f"   (розкид по гучності {spread:.3f})")

        print("   ℹ️  P(дрон) має бути СТАЛИМ при зміні гучності — "
              "інакше front-end не нормалізує рівень.")
        print("      Синтетика — лише груба перевірка «модель жива»; "
              "реальну якість показує train.py.")

    # ── 3. Мікрофон ──
    print("\n3. Мікрофон")
    try:
        audio = record(3.0, cfg, "запис")
        mono = to_mono(audio)
        level = to_db(band_rms(mono))
        peak = float(np.abs(mono).max())
        print(f"   Рівень у смузі 80–4000 Гц: {level:.1f} dBFS, "
              f"пік {peak:.3f}")
        if peak < 1e-5:
            print("   ❌ Тиша — мікрофон не пише. Перевірте пристрій "
                  "(python calibrate.py devices).")
            ok = False
        elif peak > 0.99:
            print("   ⚠️  Кліппінг — зменште підсилення (alsamixer).")

        if audio.ndim == 2 and audio.shape[1] >= 2:
            a = audio[:, 0] - audio[:, 0].mean()
            b = audio[:, 1] - audio[:, 1].mean()
            corr = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)
                                       + 1e-12))
            print(f"   Кореляція каналів 0↔1: {corr:.4f}")
            if corr > 0.999:
                print("      → канали ідентичні: масив віддає оброблене "
                      "стерео.")
                print("        Власний SRP-PHAT неможливий, потрібен "
                      "USB DOA від DSP.")
            else:
                print("      → канали різні: SRP-PHAT працюватиме.")
    except Exception as exc:
        print(f"   ❌ Помилка запису: {exc}")
        ok = False

    # ── 4. DOA ──
    print("\n4. Напрямок (DOA)")
    from doa import HardwareDOA, find_xvf_host
    path = find_xvf_host()
    if path is None:
        print("   ⚠️  xvf_host.py не знайдено.")
        print("      Задайте шлях:  export XVF_HOST_PATH=/шлях/xvf_host.py")
    else:
        print(f"   Знайдено: {path}")
        hw = HardwareDOA(path)
        reading = hw._query_once()
        if reading.ok:
            print(f"   ✅ Азимути променів: "
                  f"{[round(v, 1) for v in reading.raw]}")
            print(f"      Обрано: {reading.angle_deg:.1f}° "
                  f"(узгодженість {reading.confidence:.0%})")
        else:
            print(f"   ❌ {reading.error}")
            ok = False

    # ── 5. Калібрування ──
    print("\n5. Калібрування установки")
    print(f"   {calibration.describe(cfg)}")
    if not calibration.is_range_calibrated(cfg):
        print("   → Виконайте:  python calibrate.py noise, потім range")

    print("\n" + "=" * 66)
    print("✅ Основне працює" if ok else "❌ Є проблеми — див. вище")
    print("=" * 66)


# ═══════════════════════════════════════════════════════════════
#  noise — рівень фону
# ═══════════════════════════════════════════════════════════════

def cmd_noise(cfg: dict, seconds: float) -> None:
    print("=" * 66)
    print("🔇 Калібрування рівня фону")
    print("=" * 66)
    print(f"\n   У небі НЕ має бути дрона. Запис {seconds:.0f} с "
          f"типового фону майданчика.")
    input("   Натисніть Enter, коли готові... ")

    audio = record(seconds, cfg, "фон  ")
    levels = window_levels(audio)
    floor = float(np.percentile(levels, 20))

    print(f"\n   Рівень фону: медіана {np.median(levels):.1f} dBFS, "
          f"20-й перцентиль {floor:.1f} dBFS")
    print(f"   Розкид: {levels.min():.1f} … {levels.max():.1f} dBFS")

    calibration.save({"noise_floor_dbfs": floor})
    print("   Це значення віднімається від рівня цілі, тому вітер більше "
          "не «наближає» дрона.")


# ═══════════════════════════════════════════════════════════════
#  range — рівень дрона на відомій відстані
# ═══════════════════════════════════════════════════════════════

def cmd_range(cfg: dict, seconds: float) -> None:
    print("=" * 66)
    print("📏 Калібрування відстані")
    print("=" * 66)

    noise = cfg.get("noise_floor_dbfs")
    if noise is None:
        print("\n   ⚠️  Спочатку потрібен рівень фону.")
        print("      Запустіть: python calibrate.py noise")
        return

    print(f"\n   Рівень фону з калібрування: {noise:.1f} dBFS")
    print("   Потрібно зависнути дроном на ВІДОМІЙ відстані від мікрофона.")
    print("   Одна точка дасть робочу оцінку; дві і більше — ще й реальний")
    print("   показник згасання замість припущення 22 дБ/декаду.\n")

    points: list[tuple[float, float]] = []
    while True:
        raw = input(f"   Відстань до дрона у метрах "
                    f"(Enter — завершити, зібрано {len(points)}): ").strip()
        if not raw:
            break
        try:
            distance = float(raw.replace(",", "."))
        except ValueError:
            print("      Введіть число, наприклад 30")
            continue
        if distance <= 0:
            print("      Відстань має бути додатною")
            continue

        input(f"   Тримайте дрон на {distance:.0f} м і натисніть Enter... ")
        audio = record(seconds, cfg, "дрон ")
        levels = window_levels(audio)
        total = float(np.median(levels))
        target, excess = target_level_dbfs(total, float(noise))

        print(f"      Повний рівень {total:.1f} dBFS, "
              f"рівень дрона {target:.1f} dBFS, "
              f"над фоном {excess:.1f} дБ")
        if excess < 3.0:
            print("      ⚠️  Дрон майже не чути над фоном — точка "
                  "ненадійна, пропущено.")
            continue
        points.append((distance, target))

    if not points:
        print("\n   Жодної точки не зібрано — калібрування не змінено.")
        return

    if len(points) == 1:
        distance, level = points[0]
        spreading = float(cfg.get("range_spreading_db", 22.0))
        print(f"\n   Одна точка: {level:.1f} dBFS @ {distance:.0f} м")
        print(f"   Показник згасання лишається припущеним: "
              f"{spreading:.0f} дБ/декаду")
    else:
        # Лінійна регресія: L = a - b·log10(r)
        r = np.array([p[0] for p in points], dtype=float)
        L = np.array([p[1] for p in points], dtype=float)
        b, a = np.polyfit(np.log10(r), L, 1)
        spreading = float(-b)

        distance = float(10.0 ** np.mean(np.log10(r)))   # середнє геометричне
        level = float(a + b * np.log10(distance))

        print(f"\n   Підгонка по {len(points)} точках:")
        for d, lv in points:
            print(f"      {d:6.0f} м → {lv:7.1f} dBFS")
        print(f"   Згасання: {spreading:.1f} дБ/декаду "
              f"(у вільному полі 20; 20–28 — норма)")
        if not 12.0 <= spreading <= 40.0:
            print("   ⚠️  Значення поза розумними межами — імовірно,")
            print("      вимірювання зіпсоване вітром або AGC мікрофона.")
            print(f"      Використано 22 дБ/декаду.")
            spreading = 22.0

    calibration.save({
        "range_ref_distance_m": float(distance),
        "range_ref_level_dbfs": float(level),
        "range_spreading_db": float(spreading),
    })
    print(f"\n   Тепер radar.py показуватиме відстань у метрах.")
    print(f"   Реалістична точність — ±40–60%, тому виводиться інтервал.")


# ═══════════════════════════════════════════════════════════════
#  doa — зсув кута масиву
# ═══════════════════════════════════════════════════════════════

def measure_raw_angle(cfg: dict, seconds: float) -> tuple[float | None, str]:
    """Збирає СИРІ (без зсуву) виміри кута і повертає їх циркулярне середнє."""
    from doa import ArrayDOA, HardwareDOA, circular_mean

    hw = HardwareDOA(beam_index=int(cfg.get("doa_beam_index", -1)))
    samples: list[float] = []
    source = "none"

    if hw.available:
        deadline = time.time() + seconds
        while time.time() < deadline:
            reading = hw._query_once()
            if reading.ok:
                samples.append(reading.angle_deg)
                source = "usb"
                sys.stdout.write(f"\r   вимірів: {len(samples):3d}   "
                                 f"останній {reading.angle_deg:6.1f}°")
                sys.stdout.flush()
        sys.stdout.write("\n")

    if not samples:
        print("   USB DOA недоступний — пробуємо SRP-PHAT по сирих каналах")
        audio = record(seconds, cfg, "кут  ")
        if audio.ndim < 2 or audio.shape[1] < 2:
            return None, "none"
        srp = ArrayDOA(cfg["mic_positions_m"], features.SAMPLE_RATE)
        n = features.SAMPLE_RATE
        for i in range(0, len(audio) - n, n):
            r = srp.estimate(audio[i:i + n])
            if r.ok:
                samples.append(r.angle_deg)
                source = "srp"
        if not samples:
            return None, "none"

    return circular_mean(samples), source


def cmd_doa(cfg: dict, seconds: float) -> None:
    from doa import angular_diff, apply_orientation, circular_mean

    print("=" * 66)
    print("🧭 Калібрування напрямку")
    print("=" * 66)
    print("\n   Потрібне джерело звуку (дрон або гучний динамік) у ВІДОМОМУ")
    print("   напрямку. Напрямок задається так, як ви хочете бачити його на")
    print("   екрані: 0° = вперед/північ, 90° = праворуч, 180° = назад.")
    print("\n   Одна точка дає зсів кута. ДВІ точки додатково перевіряють,")
    print("   чи не дзеркальний напрямок обертання — тому краще зробити дві.\n")

    points: list[tuple[float, float]] = []
    while True:
        raw = input(f"   Істинний напрямок у градусах "
                    f"(Enter — завершити, зібрано {len(points)}): ").strip()
        if not raw:
            break
        try:
            true_angle = float(raw.replace(",", ".")) % 360.0
        except ValueError:
            print("      Введіть число, наприклад 90")
            continue

        input(f"   Увімкніть джерело на {true_angle:.0f}° і натисніть Enter... ")
        measured, source = measure_raw_angle(cfg, seconds)
        if measured is None:
            print("      ❌ Не вдалось виміряти кут — точка пропущена.")
            continue
        print(f"      Масив показує {measured:.1f}° (джерело: {source})")
        points.append((true_angle, measured))

    if not points:
        print("\n   Жодної точки — калібрування не змінено.")
        return

    # Перебираємо дві гіпотези напрямку обертання і беремо кращу
    best = None
    for invert in (False, True):
        offsets = [(true - (-meas if invert else meas)) % 360.0
                   for true, meas in points]
        offset = circular_mean(offsets)
        residual = sum(angular_diff(apply_orientation(meas, offset, invert),
                                    true) for true, meas in points)
        residual /= len(points)
        if best is None or residual < best[2]:
            best = (offset, invert, residual)

    offset, invert, residual = best
    print(f"\n   Зсув: {offset:+.1f}°"
          f"{'   (напрямок обертання дзеркальний)' if invert else ''}")
    print(f"   Середня похибка після корекції: {residual:.1f}°")

    if len(points) == 1:
        print("   ⚠️  Одна точка не може виявити дзеркальність — "
              "зробіть другий вимір під іншим кутом.")
    elif residual > 30.0:
        print("   ⚠️  Похибка велика. Ймовірні причини: масив рахує кут")
        print("      нестабільно, або джерело було не в тому напрямку.")

    calibration.save({"doa_offset_deg": float(offset),
                      "doa_invert": bool(invert)})


# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Калібрування акустичного радара")
    parser.add_argument("command",
                        choices=["check", "devices", "noise", "range", "doa"])
    parser.add_argument("--seconds", type=float, default=None,
                        help="тривалість вимірювання")
    args = parser.parse_args()

    cfg = calibration.load()

    if args.command == "devices":
        print_input_devices()
    elif args.command == "check":
        cmd_check(cfg)
    elif args.command == "noise":
        cmd_noise(cfg, args.seconds or 20.0)
    elif args.command == "range":
        cmd_range(cfg, args.seconds or 15.0)
    elif args.command == "doa":
        cmd_doa(cfg, args.seconds or 10.0)


if __name__ == "__main__":
    main()

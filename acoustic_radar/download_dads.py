#!/usr/bin/env python3
"""
download_dads.py — Завантаження сирих аудіо для датасету акустичного радара.

Джерела:
  1. DroneAudioDataset (Sara Al-Emadi) — записи дронів Bebop / Mambo
  2. ESC-50 (Karol Piczak)             — фонові звуки та «важкі негативи»

═══════════════════════════════════════════════════════════════════
⚠️ ЩО БУЛО ЗЛАМАНО У ПОПЕРЕДНІЙ ВЕРСІЇ ЦЬОГО СКРИПТА
═══════════════════════════════════════════════════════════════════

БАГ №1 (критичний) — у клас «дрон» потрапляли НЕ дрони.
    Репозиторій DroneAudioDataset має структуру:
        Binary_Drone_Audio/unknown/     ← шум (ESC-50 + Speech Commands)
        Binary_Drone_Audio/yes_drone/   ← власне дрони
    Старий код брав `[f for f in z.namelist() if f.endswith('.wav')]`
    і зупинявся після перших 30 файлів. У ZIP-архіві GitHub файли йдуть
    в алфавітному порядку, тому "unknown" стоїть ПЕРЕД "yes_drone" —
    усі 30 «дронів» насправді були фоновим шумом.
    Модель вчилася відрізняти шум від шуму, тому на Pi нічого не ловила.
    → Тепер беремо файли ВИКЛЮЧНО з `yes_drone/`.

БАГ №2 — неправильні номери класів ESC-50.
    Старий коментар стверджував «17=wind, 22=chirping_birds, 24=insects».
    Насправді 17=pouring_water, 22=clapping, 24=coughing
    (wind=16, chirping_birds=14, insects=7).
    Тобто у фон потрапляли переливання води, оплески і кашель,
    а вітер — найважливіший вуличний фон — не потрапляв узагалі.
    → Тепер фільтруємо за НАЗВОЮ категорії з meta/esc50.csv, а не за
      номером, який легко переплутати.

БАГ №3 — у датасеті не було «важких негативів».
    Без записів двигунів, бензопил, гелікоптерів і пилососів модель
    не має шансу навчитись відрізняти їх від дрона → хибні тривоги.
    → Тепер вони завантажуються в raw_audio/hard_negative/.

Використання:
    python download_dads.py
"""

from __future__ import annotations

import sys
import io
import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ═══════════════════════════════════════════════════════════════
#  Директорії
# ═══════════════════════════════════════════════════════════════

RAW_DIR      = Path("raw_audio")
DRONE_DIR    = RAW_DIR / "drone"          # Клас 1
AMBIENT_DIR  = RAW_DIR / "ambient"        # Клас 0 — нейтральний фон
HARDNEG_DIR  = RAW_DIR / "hard_negative"  # Клас 0 — схожі на дрон звуки
SPEECH_DIR   = RAW_DIR / "speech"         # Клас 0 — люди

for d in (DRONE_DIR, AMBIENT_DIR, HARDNEG_DIR, SPEECH_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Категорії ESC-50 (за НАЗВОЮ з meta/esc50.csv)
# ═══════════════════════════════════════════════════════════════

# Нейтральний вуличний / природний фон
AMBIENT_CATEGORIES = {
    "wind", "rain", "thunderstorm", "sea_waves", "crickets",
    "chirping_birds", "insects", "water_drops", "pouring_water",
    "crackling_fire", "footsteps", "frog", "crow", "dog", "church_bells",
}

# ⚠️ Найважливіша група: широкосмуговий/тональний гул моторів.
# Саме ці звуки модель плутає з дроном, якщо їх немає у класі 0.
HARD_NEGATIVE_CATEGORIES = {
    "engine", "chainsaw", "vacuum_cleaner", "helicopter", "airplane",
    "washing_machine", "hand_saw", "train", "siren", "car_horn",
    "clock_alarm", "fireworks", "glass_breaking", "can_opening",
}

# Людські звуки — щоб радар мовчав, коли поруч розмовляють
SPEECH_CATEGORIES = {
    "crying_baby", "laughing", "coughing", "sneezing", "breathing",
    "snoring", "clapping", "brushing_teeth", "drinking_sipping",
}


# ═══════════════════════════════════════════════════════════════
#  Завантаження з прогресом
# ═══════════════════════════════════════════════════════════════

def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done / total_size * 100
    bar = "█" * int(pct // 2.5) + "·" * (40 - int(pct // 2.5))
    sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  "
                     f"{done / 1e6:6.1f}/{total_size / 1e6:.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> Path:
    """Завантажує файл, якщо його ще немає (з можливістю продовжити)."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"    Архів уже завантажено: {dest} "
              f"({dest.stat().st_size / 1e6:.0f} MB) — пропускаємо")
        return dest
    print(f"    Завантаження: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    sys.stdout.write("\n")
    tmp.replace(dest)
    return dest


def count_wavs(d: Path) -> int:
    return len(list(d.glob("*.wav")))


# ═══════════════════════════════════════════════════════════════
#  1. Дрони — DroneAudioDataset / Binary_Drone_Audio / yes_drone
# ═══════════════════════════════════════════════════════════════

DRONE_URL = ("https://github.com/saraalemadi/DroneAudioDataset/"
             "archive/refs/heads/master.zip")
DRONE_ZIP = Path("drone_dataset.zip")

# Беремо ВСІ записи дронів. Їх ~1300 по ~1 с, це лише ~40 МБ.
# Раніше бралось 30 файлів — і навіть ті були не з тієї папки.
MAX_DRONE_FILES = 0   # 0 = без обмеження


def fetch_drones() -> None:
    print("=" * 62)
    print("1/2  Дрони — DroneAudioDataset (Binary_Drone_Audio/yes_drone)")
    print("=" * 62)

    # Прибираємо порожні файли, що могли лишитись від невдалих запусків
    for f in DRONE_DIR.glob("*.wav"):
        if f.stat().st_size == 0:
            f.unlink()

    print(f"    Уже є файлів дронів: {count_wavs(DRONE_DIR)}")
    download(DRONE_URL, DRONE_ZIP)

    print("    Розпаковка...")
    added = skipped_unknown = 0
    with zipfile.ZipFile(DRONE_ZIP, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".wav"):
                continue

            posix = name.replace("\\", "/")

            # ⚠️ ОСЬ ВИПРАВЛЕННЯ ГОЛОВНОГО БАГА:
            # беремо тільки yes_drone, все інше в архіві — це шум.
            if "/yes_drone/" not in posix:
                skipped_unknown += 1
                continue

            target = DRONE_DIR / f"dads_{Path(posix).name}"
            if target.exists():
                continue

            with z.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            added += 1

            if MAX_DRONE_FILES and added >= MAX_DRONE_FILES:
                break

    print(f"    ✅ Додано {added} записів дронів "
          f"(проігноровано {skipped_unknown} файлів не з yes_drone/)")
    print(f"    Разом у {DRONE_DIR}/: {count_wavs(DRONE_DIR)} файлів")

    if count_wavs(DRONE_DIR) == 0:
        print("    ❌ Жодного файлу дрона! Перевірте структуру архіву.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  2. Фон та важкі негативи — ESC-50
# ═══════════════════════════════════════════════════════════════

ESC_URL = ("https://github.com/karolpiczak/ESC-50/"
           "archive/refs/heads/master.zip")
ESC_ZIP = Path("esc50.zip")


def fetch_esc50() -> None:
    print(f"\n{'=' * 62}")
    print("2/2  Фон + важкі негативи — ESC-50")
    print("=" * 62)
    print(f"    Уже є: ambient={count_wavs(AMBIENT_DIR)}  "
          f"hard_negative={count_wavs(HARDNEG_DIR)}  "
          f"speech={count_wavs(SPEECH_DIR)}")

    download(ESC_URL, ESC_ZIP)  # ~600 МБ

    print("    Розпаковка за категоріями...")
    counts = {"ambient": 0, "hard_negative": 0, "speech": 0}

    with zipfile.ZipFile(ESC_ZIP, "r") as z:
        # ── Читаємо meta/esc50.csv: filename → category ──
        meta_name = next(n for n in z.namelist()
                         if n.replace("\\", "/").endswith("meta/esc50.csv"))
        with z.open(meta_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            category_of = {row["filename"]: row["category"] for row in reader}
        print(f"    Прочитано meta/esc50.csv: {len(category_of)} записів")

        # ── Розкладаємо wav-и по цільових папках ──
        for name in z.namelist():
            posix = name.replace("\\", "/")
            if not posix.lower().endswith(".wav") or "/audio/" not in posix:
                continue

            fname = Path(posix).name
            category = category_of.get(fname)
            if category is None:
                continue

            if category in AMBIENT_CATEGORIES:
                out_dir, key = AMBIENT_DIR, "ambient"
            elif category in HARD_NEGATIVE_CATEGORIES:
                out_dir, key = HARDNEG_DIR, "hard_negative"
            elif category in SPEECH_CATEGORIES:
                out_dir, key = SPEECH_DIR, "speech"
            else:
                continue

            # Категорія в імені файлу — щоб було видно, що саме завантажилось
            target = out_dir / f"esc_{category}_{fname}"
            if target.exists():
                continue

            with z.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[key] += 1

    for key, n in counts.items():
        print(f"    ✅ {key:<14s}: +{n} нових файлів")


# ═══════════════════════════════════════════════════════════════
#  Підсумок
# ═══════════════════════════════════════════════════════════════

def summary() -> None:
    print(f"\n{'=' * 62}")
    print("Підсумок raw_audio/")
    print("=" * 62)
    rows = [
        ("drone/",         DRONE_DIR,   "клас 1 — ціль"),
        ("ambient/",       AMBIENT_DIR, "клас 0 — нейтральний фон"),
        ("hard_negative/", HARDNEG_DIR, "клас 0 — мотори, схожі на дрон"),
        ("speech/",        SPEECH_DIR,  "клас 0 — люди"),
    ]
    for label, path, note in rows:
        print(f"   {label:<16s} {count_wavs(path):5d} файлів   — {note}")

    print(f"\n{'─' * 62}")
    print("⚠️  ВАЖЛИВО ПРО ЯКІСТЬ ДЕТЕКЦІЇ")
    print(f"{'─' * 62}")
    print("   Записи DroneAudioDataset зроблені В ПРИМІЩЕННІ, з близької")
    print("   відстані, лише для двох дронів (Bebop, Mambo). Це найкращий")
    print("   доступний старт, але акустика вулиці інша.")
    print()
    print("   Найбільший приріст якості дасть ваш ВЛАСНИЙ запис з того")
    print("   самого мікрофона ReSpeaker, у полі:")
    print("      raw_audio/drone/          — записи дрона (20-30 хв)")
    print("      raw_audio/ambient/        — той самий майданчик без дрона")
    print("      raw_audio/hard_negative/  — авто, мотокоса, генератор")
    print()
    print("   Формат будь-який (.wav/.mp3/.flac/.ogg), будь-яка частота —")
    print("   mixer.py сам приведе до 16 кГц моно.")
    print()
    print("   Далі:  python mixer.py  →  python train.py")
    print("=" * 62)


if __name__ == "__main__":
    fetch_drones()
    fetch_esc50()
    summary()

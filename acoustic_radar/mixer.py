#!/usr/bin/env python3
"""
mixer.py — Генератор синтетичного датасету для акустичного радара.

Створює двокласовий датасет:
    0 — Фон  (природа, люди, а ГОЛОВНЕ — мотори/бензопили/гелікоптери)
    1 — Дрон (мультикоптер на відстані 5–250 м)

═══════════════════════════════════════════════════════════════════
⚠️ ЩО БУЛО ЗЛАМАНО У ПОПЕРЕДНІЙ ВЕРСІЇ
═══════════════════════════════════════════════════════════════════

БАГ №1 (критичний) — «ярлик» у вигляді low-pass фільтра.
    Симуляція відстані (Butterworth low-pass 7.5 кГц → 2.5 кГц)
    застосовувалась ТІЛЬКИ до класу 1. Клас 0 залишався з повним
    спектром. Модель миттєво знайшла найпростіше правило:
        «є завал ВЧ вище 2.5-7.5 кГц → це дрон»
    Воно дає 96% на валідації і 0% користі в полі, бо реальний
    вуличний фон теж має завал ВЧ (вітер, дощ, далекі звуки).
    → Тепер ідентична симуляція відстані застосовується до ОБОХ класів.

БАГ №2 — мікшування «на око» замість заданого SNR.
    `mixed = background * bg_vol + target` з подальшою пік-нормалізацією
    робило реальний SNR некерованим: гучний фон просто «з'їдав» ціль.
    → Тепер ціль масштабується так, щоб отримати ЗАДАНИЙ SNR,
      який обчислюється з відстані.

БАГ №3 — модель прив'язувалась до абсолютної гучності.
    Кожен семпл нормалізувався рівно до піку 0.95, а живий мікрофон
    видає що завгодно. Модель бачила на Pi зовсім інший розподіл.
    → Тепер вихідний рівень рандомізується (0.03–0.95), а features.py
      додатково нормалізує спектрограму. Гучність більше не є ознакою.

БАГ №4 — валідація була нечесною.
    train.py робив випадковий split 80/20 ВЖЕ ЗМІКШОВАНИХ файлів, тому
    один і той самий вихідний запис дрона потрапляв і в train, і в val.
    Val accuracy показувала завчені записи, а не здатність узагальнювати.
    → Тепер вихідні файли розділяються на train/val ДО мікшування,
      і генеруються дві незалежні папки dataset/train та dataset/val.

Використання:
    python mixer.py
"""

from __future__ import annotations

import sys
import json
import random
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from features import SAMPLE_RATE, WINDOW_SEC

try:
    import librosa
except ImportError:  # pragma: no cover
    raise SystemExit("❌ Потрібен librosa:  pip install librosa")


# ═══════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════

TRAIN_SAMPLES_PER_CLASS = 4000
VAL_SAMPLES_PER_CLASS   = 800

DURATION_SEC = WINDOW_SEC       # ⚠️ має збігатися з вікном інференсу
TARGET_SR    = SAMPLE_RATE
VAL_SOURCE_FRACTION = 0.2       # Частка ВИХІДНИХ файлів, відкладених на val

# ── Симуляція відстані ──
DIST_MIN = 5.0                  # метри
DIST_MAX = 250.0

# SNR цілі відносно фону на межах діапазону відстаней.
# Дрон за 5 м добре чути, за 250 м він на межі фонового шуму.
SNR_AT_MIN_DIST_DB = 22.0
SNR_AT_MAX_DIST_DB = -10.0
SNR_JITTER_DB      = 4.0        # Розкид (різні дрони, вітер, вологість)

# Атмосферне поглинання ВЧ: частота зрізу залежно від відстані
CUTOFF_NEAR_HZ = 7500.0
CUTOFF_FAR_HZ  = 2200.0

# Вихідний рівень (пік). Рандомізований, щоб гучність не була ознакою.
OUT_PEAK_RANGE = (0.03, 0.95)

# Власний шум мікрофона — щоб «тиха» частина спектру не була ідеально нульовою
MIC_NOISE_DBFS_RANGE = (-75.0, -55.0)

# ── Директорії ──
RAW_DIR     = Path("raw_audio")
DRONE_DIR   = RAW_DIR / "drone"
AMBIENT_DIR = RAW_DIR / "ambient"
HARDNEG_DIR = RAW_DIR / "hard_negative"
SPEECH_DIR  = RAW_DIR / "speech"
MOTOR_DIR   = RAW_DIR / "motor"        # Сумісність зі старою структурою

OUTPUT_ROOT = Path("dataset")

# Ймовірність, з якою у клас 0 підмішується «важкий негатив» (мотор тощо).
# Це найцінніший вміст класу 0 — саме він прибирає хибні тривоги.
HARDNEG_PROB = 0.45
SPEECH_PROB  = 0.20


# ═══════════════════════════════════════════════════════════════
#  Симуляція відстані
# ═══════════════════════════════════════════════════════════════

def distance_to_snr_db(distance_m: float) -> float:
    """
    Відстань → SNR цілі відносно фону.

    Сферичне розходження дає -6 дБ на кожне подвоєння відстані, тому
    інтерполюємо лінійно за log2(відстані), а не за самою відстанню.
    """
    t = ((np.log2(distance_m) - np.log2(DIST_MIN)) /
         (np.log2(DIST_MAX) - np.log2(DIST_MIN)))
    t = float(np.clip(t, 0.0, 1.0))
    snr = SNR_AT_MIN_DIST_DB + t * (SNR_AT_MAX_DIST_DB - SNR_AT_MIN_DIST_DB)
    return snr + random.uniform(-SNR_JITTER_DB, SNR_JITTER_DB)


def atmospheric_lowpass(audio: np.ndarray, distance_m: float,
                        sr: int) -> np.ndarray:
    """
    Атмосферне поглинання високих частот (зростає з відстанню).

    ⚠️ Застосовується до ОБОХ класів — інакше стає «ярликом» для класу 1.
    """
    t = ((np.log2(distance_m) - np.log2(DIST_MIN)) /
         (np.log2(DIST_MAX) - np.log2(DIST_MIN)))
    t = float(np.clip(t, 0.0, 1.0))

    cutoff_hz = CUTOFF_NEAR_HZ - t * (CUTOFF_NEAR_HZ - CUTOFF_FAR_HZ)
    cutoff_norm = min(cutoff_hz / (sr / 2.0), 0.99)
    if cutoff_norm >= 0.98:
        return audio

    sos = butter(2, cutoff_norm, btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def random_distance() -> float:
    """Логарифмічно рівномірна відстань — більше семплів на далеких дистанціях."""
    return float(np.exp(random.uniform(np.log(DIST_MIN), np.log(DIST_MAX))))


# ═══════════════════════════════════════════════════════════════
#  Аугментації
# ═══════════════════════════════════════════════════════════════

def random_eq_tilt(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Випадковий нахил АЧХ ±6 дБ — імітує різні мікрофони, вітрозахист,
    орієнтацію дрона відносно масиву.
    """
    tilt_db = random.uniform(-6.0, 6.0)
    if abs(tilt_db) < 0.5:
        return audio
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    # Лінійний нахил у log-частоті від 100 Гц до Найквіста
    w = np.clip(np.log2(np.maximum(freqs, 1.0) / 100.0) / np.log2(sr / 2 / 100.0),
                0.0, 1.0)
    spec *= 10.0 ** (tilt_db * w / 20.0)
    return np.fft.irfft(spec, n=len(audio)).astype(np.float32)


def random_speed(audio: np.ndarray) -> np.ndarray:
    """
    Зміна швидкості ±6% — імітує різні оберти гвинтів і доплерівський зсув.
    Довжину відновлюємо обрізанням/зацикленням у викликаючому коді.
    """
    rate = random.uniform(0.94, 1.06)
    n_out = int(len(audio) / rate)
    x_old = np.linspace(0.0, 1.0, len(audio))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def add_mic_noise(audio: np.ndarray) -> np.ndarray:
    """Додає власний шум мікрофона (теплова підлога тракту)."""
    level = 10.0 ** (random.uniform(*MIC_NOISE_DBFS_RANGE) / 20.0)
    return audio + level * np.random.standard_normal(len(audio)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  Робота з аудіофайлами
# ═══════════════════════════════════════════════════════════════

_CACHE: dict[Path, np.ndarray] = {}
_CACHE_LIMIT = 1200


def load_file(path: Path) -> np.ndarray | None:
    """Завантажує файл у моно 16 кГц з кешуванням. None — якщо пошкоджений."""
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    try:
        audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if audio is None or len(audio) < TARGET_SR // 20:   # коротше 50 мс
        return None
    audio = audio.astype(np.float32)
    if len(_CACHE) < _CACHE_LIMIT:
        _CACHE[path] = audio
    return audio


def take_segment(files: list[Path], duration: float,
                 augment: bool = True) -> np.ndarray:
    """
    Формує фрагмент заданої тривалості з випадкових файлів списку.

    ⚠️ Записи дронів у DroneAudioDataset — це кліпи по ~1 с. Старий код
    просто зациклював ОДИН кліп (np.tile), створюючи неприродну періодичність
    з періодом рівно 1 с — ще один «ярлик», який модель могла вивчити.
    Тут короткі фрагменти склеюються з РІЗНИХ випадкових файлів через
    коротке перехресне згасання.
    """
    need = int(duration * TARGET_SR)
    xfade = int(0.02 * TARGET_SR)   # 20 мс
    out = np.zeros(0, dtype=np.float32)
    attempts = 0

    while len(out) < need and attempts < 60:
        attempts += 1
        path = random.choice(files)
        audio = load_file(path)
        if audio is None:
            if path in files:
                files.remove(path)
            if not files:
                raise RuntimeError("❌ Не лишилось придатних аудіофайлів!")
            continue

        if augment:
            audio = random_speed(audio)

        # Випадковий фрагмент цього файлу
        chunk_len = min(len(audio), need - len(out) + xfade)
        start = random.randint(0, max(0, len(audio) - chunk_len))
        chunk = audio[start:start + chunk_len].copy()

        if len(out) == 0:
            out = chunk
        else:
            # Перехресне згасання на стику
            n = min(xfade, len(out), len(chunk))
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out[-n:] = out[-n:] * (1.0 - ramp) + chunk[:n] * ramp
            out = np.concatenate([out, chunk[n:]])

    if len(out) < need:
        out = np.pad(out, (0, need - len(out)))
    return out[:need]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-20))


def mix_at_snr(target: np.ndarray, background: np.ndarray,
               snr_db: float) -> np.ndarray:
    """Масштабує ціль так, щоб отримати заданий SNR відносно фону."""
    t_rms, b_rms = rms(target), rms(background)
    if t_rms < 1e-9:
        return background
    scale = (b_rms * (10.0 ** (snr_db / 20.0))) / t_rms
    return background + target * scale


def finalize(audio: np.ndarray) -> np.ndarray:
    """Шум мікрофона + випадковий вихідний рівень + захист від кліппінгу."""
    audio = add_mic_noise(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak * random.uniform(*OUT_PEAK_RANGE)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  Сканування та розділення вихідних файлів
# ═══════════════════════════════════════════════════════════════

def scan(directory: Path, label: str, required: bool = True) -> list[Path]:
    """Знаходить усі аудіофайли у директорії."""
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.WAV", "*.MP3"):
        files.extend(sorted(directory.glob(ext)))
    files = sorted(set(files))

    mark = "📂" if files else ("❌" if required else "⚠️ ")
    print(f"   {mark} {label:<16s} {len(files):5d} файлів  ({directory})")

    if required and not files:
        raise FileNotFoundError(
            f"❌ Немає аудіо у '{directory}'. Запустіть спочатку: "
            f"python download_dads.py"
        )
    return files


def split_sources(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Ділить ВИХІДНІ файли на train/val так, щоб один і той самий запис
    не потрапив в обидві вибірки.

    Файли DroneAudioDataset названі як `dads_B_S2_D1_067-bebop_003_.wav` —
    кліпи одного сеансу мають спільний префікс. Групуємо за префіксом,
    щоб сусідні секунди одного польоту не розповзались між train і val.
    """
    groups: dict[str, list[Path]] = {}
    for f in files:
        key = f.stem.rsplit("_", 2)[0] if "_" in f.stem else f.stem
        groups.setdefault(key, []).append(f)

    keys = sorted(groups)
    random.Random(1337).shuffle(keys)
    n_val = max(1, int(len(keys) * VAL_SOURCE_FRACTION))

    val_files = [f for k in keys[:n_val] for f in groups[k]]
    train_files = [f for k in keys[n_val:] for f in groups[k]]

    # Захист від виродженого поділу (мало вихідних файлів)
    if not train_files or not val_files:
        cut = max(1, int(len(files) * VAL_SOURCE_FRACTION))
        val_files, train_files = files[:cut], files[cut:]
    return train_files, val_files


# ═══════════════════════════════════════════════════════════════
#  Генерація одного семплу
# ═══════════════════════════════════════════════════════════════

def make_sample(class_id: int, pools: dict[str, list[Path]]) -> tuple[np.ndarray, dict]:
    """
    Генерує один семпл заданого класу.

    ⚠️ Ключовий принцип: обидва класи проходять ІДЕНТИЧНИЙ конвеєр
    (фон → «джерело на відстані» → low-pass → SNR-мікс → рандомний рівень).
    Різниця лише в тому, ЩО саме є джерелом: дрон чи не-дрон.
    Через це модель змушена вчити тембр, а не побічні артефакти.
    """
    background = take_segment(pools["ambient"], DURATION_SEC)
    distance = random_distance()
    snr_db = distance_to_snr_db(distance)

    if class_id == 1:
        source_files = pools["drone"]
        source_kind = "drone"
    else:
        # Клас 0: фон сам по собі, або фон + «важкий негатив», або + мова
        r = random.random()
        if r < HARDNEG_PROB and pools["hard_negative"]:
            source_files, source_kind = pools["hard_negative"], "hard_negative"
        elif r < HARDNEG_PROB + SPEECH_PROB and pools["speech"]:
            source_files, source_kind = pools["speech"], "speech"
        else:
            # Чистий фон: другий незалежний фрагмент фону як «джерело»
            source_files, source_kind = pools["ambient"], "ambient"

    source = take_segment(source_files, DURATION_SEC)
    source = random_eq_tilt(source, TARGET_SR)
    source = atmospheric_lowpass(source, distance, TARGET_SR)

    audio = finalize(mix_at_snr(source, background, snr_db))
    meta = {
        "class": class_id,
        "source": source_kind,
        "distance_m": round(distance, 1),
        "snr_db": round(snr_db, 1),
    }
    return audio, meta


# ═══════════════════════════════════════════════════════════════
#  Генерація сплітів
# ═══════════════════════════════════════════════════════════════

def generate_split(name: str, per_class: int,
                   pools: dict[str, list[Path]]) -> None:
    out_root = OUTPUT_ROOT / name
    for cls in (0, 1):
        (out_root / str(cls)).mkdir(parents=True, exist_ok=True)
        # Чистимо попередню генерацію, щоб не змішувати версії датасету
        for old in (out_root / str(cls)).glob("*.wav"):
            old.unlink()

    manifest = []
    print(f"\n── Спліт '{name}': {per_class} семплів на клас ──")
    for class_id in (0, 1):
        label = {0: "Фон", 1: "Дрон"}[class_id]
        for i in tqdm(range(per_class), desc=f"  {class_id}:{label:<5s}",
                      ncols=68):
            audio, meta = make_sample(class_id, pools)
            fname = f"{class_id}_{i:05d}.wav"
            sf.write(str(out_root / str(class_id) / fname), audio,
                     TARGET_SR, subtype="PCM_16")
            meta["file"] = f"{class_id}/{fname}"
            manifest.append(meta)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # Статистика по відстанях (тільки клас 1)
    dists = [m["distance_m"] for m in manifest if m["class"] == 1]
    bins = [(5, 25), (25, 60), (60, 120), (120, 250)]
    print(f"   Розподіл відстаней (клас 1, всього {len(dists)}):")
    for lo, hi in bins:
        n = sum(1 for d in dists if lo <= d < hi)
        print(f"      {lo:3d}–{hi:3d} м: {n:5d} ({n / max(len(dists), 1):5.1%})")

    kinds: dict[str, int] = {}
    for m in manifest:
        if m["class"] == 0:
            kinds[m["source"]] = kinds.get(m["source"], 0) + 1
    print(f"   Склад класу 0: " +
          ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


# ═══════════════════════════════════════════════════════════════
#  Точка входу
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("Acoustic Radar — Dataset Generator v3.0")
    print("=" * 68)

    print("\nСканування raw_audio/ ...")
    drone_all   = scan(DRONE_DIR, "drone/", required=True)
    ambient_all = scan(AMBIENT_DIR, "ambient/", required=True)
    hardneg_all = scan(HARDNEG_DIR, "hard_negative/", required=False)
    hardneg_all += scan(MOTOR_DIR, "motor/ (legacy)", required=False)
    speech_all  = scan(SPEECH_DIR, "speech/", required=False)

    if not hardneg_all:
        print("\n   ⚠️  Немає 'важких негативів' (мотори, бензопили, "
              "гелікоптери).")
        print("       Радар працюватиме, але буде давати хибні тривоги на "
              "будь-який гул.")
        print("       Запустіть python download_dads.py щоб їх завантажити.")

    # ── Розділяємо ВИХІДНІ файли на train/val ──
    pools_train, pools_val = {}, {}
    for key, files in (("drone", drone_all), ("ambient", ambient_all),
                       ("hard_negative", hardneg_all), ("speech", speech_all)):
        if files:
            tr, va = split_sources(files)
        else:
            tr, va = [], []
        pools_train[key], pools_val[key] = tr, va

    print("\nПоділ ВИХІДНИХ файлів (train / val не перетинаються):")
    for key in ("drone", "ambient", "hard_negative", "speech"):
        print(f"   {key:<14s} train={len(pools_train[key]):5d}  "
              f"val={len(pools_val[key]):5d}")

    print(f"\nПараметри:")
    print(f"   Тривалість:   {DURATION_SEC} с @ {TARGET_SR} Гц")
    print(f"   Відстань:     {DIST_MIN:.0f}–{DIST_MAX:.0f} м "
          f"(лог-рівномірно)")
    print(f"   SNR:          {SNR_AT_MIN_DIST_DB:+.0f} дБ (близько) → "
          f"{SNR_AT_MAX_DIST_DB:+.0f} дБ (далеко)")
    print(f"   Low-pass:     {CUTOFF_NEAR_HZ:.0f} → {CUTOFF_FAR_HZ:.0f} Гц "
          f"— для ОБОХ класів")
    print(f"   Вихідний пік: {OUT_PEAK_RANGE[0]:.2f}–{OUT_PEAK_RANGE[1]:.2f} "
          f"(рандомізовано)")

    generate_split("train", TRAIN_SAMPLES_PER_CLASS, pools_train)
    generate_split("val", VAL_SAMPLES_PER_CLASS, pools_val)

    print(f"\n{'=' * 68}")
    print(f"✅ Готово. Датасет у {OUTPUT_ROOT.resolve()}")
    print(f"   Наступний крок:  python train.py")
    print("=" * 68)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

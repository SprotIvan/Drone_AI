#!/usr/bin/env python3
"""
features.py — ЄДИНИЙ front-end обчислення Mel-спектрограм.

⚠️ КРИТИЧНО ВАЖЛИВО ⚠️
Цей модуль використовується І під час навчання (dataset.py), І під час
інференсу на Raspberry Pi (radar.py). Це гарантує, що модель бачить
*точно ті самі* числа в обох випадках.

Раніше було дві незалежні реалізації (torchaudio у навчанні + ручний NumPy
в radar.py), і вони давали різні спектрограми:
  - різні трикутні Mel-фільтри (округлення до FFT-бінів vs точна інтерполяція)
  - різне вікно Ганна (симетричне vs періодичне)
  - відсутність нормалізації рівня
Через це модель, натренована на PC, на Pi отримувала сміття на вході.

Ключові властивості:
  1. Реалізація на чистому NumPy — на Pi не потрібен PyTorch/torchaudio.
  2. Mel-фільтри будуються точною лінійною інтерполяцією по частоті
     (ідентично torchaudio.functional.melscale_fbanks, mel_scale="htk").
  3. Per-sample нормалізація (mean/std) робить ознаки НЕЗАЛЕЖНИМИ від
     гучності запису. Це найважливіше виправлення: датасет нормалізований
     по піку, а живий мікрофон — ні, тому без цього кроку модель на Pi
     отримувала вхід, зміщений на десятки дБ від тренувального розподілу.
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Параметри front-end (єдине джерело правди)
# ═══════════════════════════════════════════════════════════════

SAMPLE_RATE = 16_000     # Частота дискретизації (Гц)
N_FFT       = 1024       # Вікно FFT — 64 мс. Довге вікно дає роздільну
                         # здатність 15.6 Гц, достатню щоб розділити
                         # гармоніки лопатей дрона (blade-pass ~100-250 Гц).
HOP_LENGTH  = 256        # Крок — 16 мс
N_MELS      = 64         # Кількість Mel-фільтрів
F_MIN       = 50.0       # Нижня межа — відрізає інфразвуковий гул вітру
F_MAX       = 8000.0     # Верхня межа (Найквіст)
TOP_DB      = 80.0       # Обмеження динамічного діапазону

# Тривалість вікна аналізу. Навчання і інференс використовують ОДНАКОВУ
# тривалість — модель має AdaptiveAvgPool, але статистики нормалізації
# стабільніші при однаковій довжині.
WINDOW_SEC    = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
# n_frames = 1 + (32000 + 1024 - 1024) // 256 = 126
TARGET_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH

# Смуга, у якій живе акустичний підпис мультикоптера. Використовується
# для оцінки рівня сигналу (ranging.py) та для шумового гейта.
DRONE_BAND_HZ = (80.0, 4000.0)


# ═══════════════════════════════════════════════════════════════
#  Mel-шкала (HTK)
# ═══════════════════════════════════════════════════════════════

def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Гц → Mel за формулою HTK (та сама, що в torchaudio mel_scale='htk')."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Mel → Гц (обернена до hz_to_mel)."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = SAMPLE_RATE,
                   n_fft: int = N_FFT,
                   n_mels: int = N_MELS,
                   f_min: float = F_MIN,
                   f_max: float = F_MAX) -> np.ndarray:
    """
    Будує банк трикутних Mel-фільтрів.

    ⚠️ Стара реалізація в radar.py округлювала межі фільтрів до цілих
    FFT-бінів (`np.floor((n_fft+1)*hz/sr)`). При n_mels=32 і n_fft=400
    нижні фільтри вироджувались у одиничні біни, і форма фільтрів
    відрізнялась від torchaudio (cosine similarity падала до 0.73 саме
    в найважливішій для дрона низькочастотній області).

    Тут використана точна лінійна інтерполяція по частоті — так само,
    як у torchaudio._create_triangular_filterbank.

    Returns:
        fb: ndarray [n_mels, n_fft//2 + 1], dtype float32
    """
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    # Рівновіддалені точки у Mel-шкалі → назад у Гц
    m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    # Трикутники: підйом від f_pts[i] до f_pts[i+1], спад до f_pts[i+2]
    f_diff = np.diff(f_pts)                       # [n_mels + 1]
    slopes = f_pts[None, :] - all_freqs[:, None]  # [n_freqs, n_mels + 2]

    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]

    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return np.ascontiguousarray(fb.T, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  STFT
# ═══════════════════════════════════════════════════════════════

def hann_window(n: int) -> np.ndarray:
    """
    Періодичне вікно Ганна — ідентичне torch.hann_window(n) за замовчуванням.

    ⚠️ np.hanning(n) повертає СИМЕТРИЧНЕ вікно (інша формула!), тому
    використовувати його для сумісності з torchaudio не можна.
    """
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def _frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Нарізає сигнал на кадри без копіювання (stride tricks)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_frames = 1 + (len(x) - frame_len) // hop
    if n_frames <= 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame_len), strides=(hop * stride, stride),
        writeable=False,
    )


def power_spectrogram(audio: np.ndarray,
                      n_fft: int = N_FFT,
                      hop_length: int = HOP_LENGTH,
                      window: np.ndarray | None = None) -> np.ndarray:
    """
    Спектрограма потужності (|STFT|²), center=True, pad_mode='reflect'.

    Returns:
        ndarray [n_fft//2 + 1, n_frames]
    """
    if window is None:
        window = hann_window(n_fft)

    pad = n_fft // 2
    # reflect потребує сигнал довший за pad; коротший — доповнюємо нулями
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if len(audio) <= pad:
        audio = np.pad(audio, (0, pad + 1 - len(audio)))
    padded = np.pad(audio, (pad, pad), mode="reflect")

    frames = _frame_signal(padded, n_fft, hop_length) * window   # [T, n_fft]
    spec = np.fft.rfft(frames, axis=1)                           # [T, n_bins]
    return np.ascontiguousarray((spec.real ** 2 + spec.imag ** 2).T,
                                dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  Головна функція: аудіо → нормалізована Mel-спектрограма
# ═══════════════════════════════════════════════════════════════

class MelFrontend:
    """
    Обгортка з передобчисленими фільтрами та вікном.

    Використання:
        fe = MelFrontend()
        mel = fe(audio_1d_float32)          # [n_mels, n_frames], нормалізовано
        x   = fe.as_model_input(mel)        # [1, 1, n_mels, n_frames] float32
    """

    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS,
                 f_min: float = F_MIN,
                 f_max: float = F_MAX,
                 top_db: float = TOP_DB,
                 normalize: bool = True):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        self.normalize = normalize
        self.filterbank = mel_filterbank(sample_rate, n_fft, n_mels,
                                         f_min, f_max)
        self.window = hann_window(n_fft)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.mel_db(audio)

    def mel_db(self, audio: np.ndarray) -> np.ndarray:
        """
        Аудіо (1D float32) → Mel-спектрограма.

        Кроки:
          1. STFT потужності
          2. Mel-фільтрація
          3. → дБ з обмеженням динамічного діапазону (top_db)
          4. Per-sample нормалізація (mean/std) — робить ознаки
             незалежними від абсолютної гучності

        Returns:
            ndarray [n_mels, n_frames], float32
        """
        power = power_spectrogram(audio, self.n_fft, self.hop_length,
                                  self.window)
        mel = self.filterbank @ power                 # [n_mels, T]

        mel = np.maximum(mel, 1e-10)
        mel_db = 10.0 * np.log10(mel, dtype=np.float32)

        # Обмеження динамічного діапазону (аналог AmplitudeToDB(top_db=80))
        mel_db = np.maximum(mel_db, mel_db.max() - self.top_db)

        if self.normalize:
            # ── Найважливіший крок ──
            # Прибирає залежність від гучності мікрофона / підсилення.
            # Форма спектру (гармоніки лопатей) зберігається, абсолютний
            # рівень — ні. Рівень для оцінки відстані міряється окремо
            # у ranging.py, безпосередньо з хвилі.
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)

        return np.ascontiguousarray(mel_db, dtype=np.float32)

    @staticmethod
    def as_model_input(mel: np.ndarray) -> np.ndarray:
        """[n_mels, T] → [1, 1, n_mels, T] для ONNX/PyTorch."""
        return mel[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

    def fix_length(self, mel: np.ndarray,
                   target_frames: int = TARGET_FRAMES) -> np.ndarray:
        """
        Приводить спектрограму до фіксованої довжини.

        ⚠️ Доповнення робиться МІНІМУМОМ спектрограми, а не нулями.
        Нуль у нормалізованій шкалі — це «середня гучність», тобто
        zero-padding вставляв би у тишу штучний сигнал.
        """
        n_frames = mel.shape[-1]
        if n_frames < target_frames:
            pad = target_frames - n_frames
            fill = float(mel.min())
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant",
                         constant_values=fill)
        elif n_frames > target_frames:
            mel = mel[:, :target_frames]
        return mel


# ═══════════════════════════════════════════════════════════════
#  Утиліти рівня сигналу (для гейта та оцінки відстані)
# ═══════════════════════════════════════════════════════════════

def band_rms(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
             band: tuple[float, float] = DRONE_BAND_HZ) -> float:
    """
    RMS сигналу у заданій смузі частот (через FFT, теорема Парсеваля).

    Чому не просто np.max(np.abs(audio)):
      - пік ловить поодинокі клацання/пориви вітру, а не стаціонарний гул;
      - широкосмуговий пік домінує низькочастотним гулом вітру, який
        не має стосунку до дрона.
    RMS у смузі 80–4000 Гц набагато краще корелює з наявністю дрона.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return 0.0
    n = audio.size
    spec = np.fft.rfft(audio * hann_window(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    # Компенсація енергії вікна Ганна (coherent power gain = 3/8)
    power = np.sum(np.abs(spec[mask]) ** 2) / (n * n * 0.375) * 2.0
    return float(np.sqrt(max(power, 0.0)))


def to_db(x: float, floor_db: float = -120.0) -> float:
    """Амплітуда → дБ відносно повної шкали (dBFS)."""
    return float(max(20.0 * np.log10(max(x, 1e-12)), floor_db))


# ═══════════════════════════════════════════════════════════════
#  Самотест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("🎛  MelFrontend — самотест")
    print("=" * 62)
    print(f"   sr={SAMPLE_RATE} n_fft={N_FFT} hop={HOP_LENGTH} "
          f"n_mels={N_MELS} f=[{F_MIN:.0f}, {F_MAX:.0f}] Гц")
    print(f"   вікно {WINDOW_SEC} с → {TARGET_FRAMES} фреймів")

    fe = MelFrontend()
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE

    # Синтетичний «дрон»: гармоніки частоти проходження лопатей + шум
    drone = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 30))
    drone = (drone + 0.3 * rng.standard_normal(t.size)).astype(np.float32)
    drone /= np.abs(drone).max()

    mel = fe(drone)
    print(f"\n   mel.shape = {mel.shape}  (очікується "
          f"({N_MELS}, {TARGET_FRAMES}))")
    print(f"   mel: mean={mel.mean():+.3f} std={mel.std():.3f} "
          f"(має бути ~0 / ~1)")

    # ── Головна перевірка: інваріантність до гучності ──
    print("\n   Інваріантність до рівня (той самий сигнал, різне підсилення):")
    ref = fe(drone)
    for gain in (1.0, 0.3, 0.05, 0.01):
        d = np.abs(fe(drone * gain) - ref).max()
        rms = band_rms(drone * gain)
        print(f"      gain={gain:<5} band_rms={rms:.5f} "
              f"({to_db(rms):+6.1f} dBFS)  max|Δmel|={d:.4f}")
    print("      → Δmel ≈ 0 означає, що класифікація не залежить від "
          "гучності ✅")

    print(f"\n   Fbank: {fe.filterbank.shape}, "
          f"порожніх фільтрів: {int((fe.filterbank.sum(1) == 0).sum())}")
    print()

#!/usr/bin/env python3
"""
radar.py — Real-time акустичний радар для Raspberry Pi.

Smart Tracking State Machine для виявлення дронів та автомобілів
через мікрофонний масив ReSpeaker з передачею DOA кута по UART.

Дальність виявлення: до ~200 метрів.

Стани State Machine:
  SLEEP     — тиша, інференс вимкнено (гучність < 0.03)
  LISTENING — активне прослуховування, модель працює
  ATTENTION — накопичення підтверджень дрона (1–4 з 5)
  ALARM 🚁  — тривога, дрон підтверджено (≥5 підтверджень)

Використання:
  python radar.py
"""

import numpy as np
import sounddevice as sd
import onnxruntime as ort
import serial
import time
import shutil
import sys
import math
import subprocess
import ast


# ═══════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════

# ── Аудіо ──
SAMPLE_RATE    = 16000       # Частота дискретизації (Гц)
CHANNELS       = 2           # Кількість каналів (ReSpeaker 2-mic)
CHUNK_DURATION = 2.0         # Тривалість чанка (секунди)
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)  # 32000 семплів

# ── Mel-спектрограма (МАЮТЬ збігатися з dataset.py!) ──
N_MELS     = 32
N_FFT      = 400
HOP_LENGTH = 160

# ── Модель ──
ONNX_MODEL_PATH = "acoustic_radar.onnx"
CLASS_NAMES = {0: "Фон", 1: "ДРОН"}

# ── Smart Tracking ──
NOISE_GATE          = 0.03   # Мін. гучність для активації (поріг ~200м)
CONFIRM_THRESHOLD   = 5      # Підтверджень поспіль для тривоги
CONFIDENCE_MIN      = 0.85   # Мін. впевненість (85%)
MISS_TOLERANCE      = 5      # Пропуски перед скиданням тривоги

# ── DOA (Direction of Arrival) ──
MIC_DISTANCE        = 0.058  # Відстань між мікрофонами ReSpeaker (метри)
SPEED_OF_SOUND      = 343.0  # Швидкість звуку (м/с)
DOA_UPDATE_INTERVAL = 1.5    # Мін. інтервал оновлення кута (секунди)

# ── UART ──
UART_PORT = "/dev/serial0"
UART_BAUD = 115200


# ═══════════════════════════════════════════════════════════════
#  Mel-спектрограма (чистий NumPy, без PyTorch)
#
#  На Raspberry Pi не потрібен PyTorch для інференсу —
#  обчислюємо Mel-спектрограму вручну через NumPy/FFT.
# ═══════════════════════════════════════════════════════════════

def create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Створює банк трикутних Mel-фільтрів.

    Аналог librosa.filters.mel / torchaudio MelSpectrogram.

    Args:
        sr:     Частота дискретизації
        n_fft:  Розмір FFT вікна
        n_mels: Кількість Mel-фільтрів

    Returns:
        filterbank: ndarray [n_mels, n_fft//2+1]
    """
    f_min = 0.0
    f_max = sr / 2.0

    # Конвертація Гц → Mel
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    # Рівномірно розподілені точки в Mel-шкалі
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Конвертація Mel → Гц
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # Конвертація Гц → біни FFT
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Побудова трикутних фільтрів
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        # Висхідний нахил
        for j in range(bin_points[i], bin_points[i + 1]):
            denom = bin_points[i + 1] - bin_points[i]
            if denom > 0:
                filterbank[i, j] = (j - bin_points[i]) / denom

        # Низхідний нахил
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            denom = bin_points[i + 2] - bin_points[i + 1]
            if denom > 0:
                filterbank[i, j] = (bin_points[i + 2] - j) / denom

    return filterbank


def compute_mel_spectrogram(audio: np.ndarray, sr: int, n_fft: int,
                            hop_length: int, n_mels: int,
                            mel_fb: np.ndarray) -> np.ndarray:
    """
    Обчислює Mel-спектрограму в дБ (аналог torchaudio з center=True).

    Відтворює поведінку:
      torchaudio.transforms.MelSpectrogram(center=True, pad_mode='reflect')
      + AmplitudeToDB(stype='power', top_db=80)

    Args:
        audio:      1D ndarray — моно аудіо
        sr:         Частота дискретизації
        n_fft:      Розмір FFT вікна
        hop_length: Крок FFT вікна
        n_mels:     Кількість Mel-фільтрів
        mel_fb:     Mel-фільтр банк [n_mels, n_fft//2+1]

    Returns:
        mel_db: ndarray [n_mels, n_frames] — Mel-спектрограма в дБ
    """
    # ── 1. Center padding (як у torchaudio з center=True) ──
    pad_amount = n_fft // 2
    audio_padded = np.pad(audio, (pad_amount, pad_amount), mode="reflect")

    # ── 2. STFT (Short-Time Fourier Transform) ──
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float32)

    # Векторизована STFT через stride tricks
    n_bins = n_fft // 2 + 1
    stft_matrix = np.zeros((n_bins, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio_padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft_matrix[:, i] = np.abs(spectrum) ** 2  # Power spectrum

    # ── 3. Mel-фільтрація ──
    mel_spec = mel_fb @ stft_matrix  # [n_mels, n_frames]

    # ── 4. Амплітуда → дБ ──
    mel_spec = np.maximum(mel_spec, 1e-10)       # Захист від log(0)
    mel_db = 10.0 * np.log10(mel_spec)

    # Top-dB clipping (аналог torchaudio AmplitudeToDB top_db=80)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)

    return mel_db


# ═══════════════════════════════════════════════════════════════
#  DOA: USB Hardware Azimuth (XVF3800)
#
#  Оскільки мікрофон сам обробляє 4 канали і віддає чисте стерео,
#  ми не можемо рахувати DOA з аудіо. Замість цього ми запитуємо
#  кут напряму у DSP-процесора через офіційну утиліту xvf_host.py.
# ═══════════════════════════════════════════════════════════════

def get_usb_doa() -> int:
    """
    Запитує Direction of Arrival через USB-утиліту Seeed Studio.
    Повертає кут від 0 до 359 градусів.
    """
    try:
        # Шлях до офіційної утиліти, яку ми завантажили
        script_path = "/home/pi/mk/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py"
        
        # Запускаємо її через той самий Python (з venv)
        cmd = [sys.executable, script_path, "AEC_AZIMUTH_VALUES"]
        
        # Запуск у фоні, чекаємо результат (зазвичай ~50мс)
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        # Парсимо вивід (наприклад: "AEC_AZIMUTH_VALUES: [0.273, 3.688, 2.993, 2.993]")
        for line in out.splitlines():
            if line.startswith("AEC_AZIMUTH_VALUES:"):
                # Відрізаємо назву, залишаємо масив
                arr_str = line.split(":", 1)[1].strip()
                # Перетворюємо рядок у масив Python
                vals = ast.literal_eval(arr_str)
                
                # Останнє значення в масиві — це зазвичай Auto selected beam
                angle_rad = float(vals[-1])
                angle_deg = math.degrees(angle_rad)
                
                # Повертаємо кут 0-359
                return int(round(angle_deg)) % 360
                
        return 90  # Якщо не знайшли рядок, повертаємо центр
    except Exception as e:
        # У разі помилки USB (щоб не крешити весь радар)
        return 90


# ═══════════════════════════════════════════════════════════════
#  Smart Tracking State Machine
# ═══════════════════════════════════════════════════════════════

class SmartTracker:
    """
    State Machine для інтелектуального відстеження цілей (Дрон та Мотор).

    Стани:
      SLEEP     — тиша, інференс вимкнено
      LISTENING — активне прослуховування
      ATTENTION — накопичення підтверджень (1-4/5)
      ALARM     — тривога, ціль підтверджено (≥5)
    """

    STATE_SLEEP     = "SLEEP"
    STATE_LISTENING = "LISTEN"
    STATE_ATTENTION = "TRACK"
    STATE_ALARM     = "ALARM"

    def __init__(self):
        self.state          = self.STATE_SLEEP
        self.target_class   = 0          # 0=None, 1=Дрон, 2=Мотор
        self.target_conf    = 0.0        # Впевненість захопленої цілі
        self.confirmations  = 0
        self.miss_count     = 0
        self.current_angle  = 0
        self.last_doa_time  = 0.0

    def update(self, predicted_class: int, confidence: float,
               angle: int, current_time: float) -> bool:
        """
        Оновлює стан на основі результатів інференсу.
        """
        send_uart = False
        
        # Відстежуємо дрон (1)
        is_target = (predicted_class == 1) and (confidence > CONFIDENCE_MIN)

        if is_target:
            if self.state in (self.STATE_SLEEP, self.STATE_LISTENING):
                # Починаємо нове захоплення
                self.target_class = predicted_class
                self.target_conf = confidence
                self.confirmations = 1
                self.miss_count = 0
                self.state = self.STATE_ATTENTION
                self.current_angle = angle
                self.last_doa_time = current_time
            elif self.state in (self.STATE_ATTENTION, self.STATE_ALARM):
                if predicted_class == self.target_class:
                    # Продовжуємо супроводження тієї ж цілі
                    self.confirmations += 1
                    self.miss_count = 0
                    self.target_conf = confidence
                    if self.confirmations >= CONFIRM_THRESHOLD:
                        self.state = self.STATE_ALARM
                        
                    # Оновлення кута DOA (ТІЛЬКИ якщо новий кут відхиляється не більше ніж на 50°)
                    diff = abs((angle - self.current_angle + 180) % 360 - 180)
                    if diff <= 50:
                        if current_time - self.last_doa_time >= DOA_UPDATE_INTERVAL:
                            self.current_angle = angle
                            self.last_doa_time = current_time
                            send_uart = True
                else:
                    # З'явилася інша ціль. Рахуємо як пропуск, щоб не скидати відразу
                    self.miss_count += 1
        else:
            # Фон або низька впевненість
            if self.state in (self.STATE_ATTENTION, self.STATE_ALARM):
                self.miss_count += 1
            else:
                self.state = self.STATE_LISTENING
                self.target_class = 0
                self.confirmations = 0

        # Перевірка на скидання через перевищення пропусків
        if self.state in (self.STATE_ATTENTION, self.STATE_ALARM) and self.miss_count > MISS_TOLERANCE:
            self.state = self.STATE_LISTENING
            self.target_class = 0
            self.confirmations = 0
            self.miss_count = 0

        return send_uart

    def set_sleep(self):
        """Перехід у режим сну (тиша)."""
        if self.state not in (self.STATE_ALARM, self.STATE_ATTENTION):
            self.state = self.STATE_SLEEP
            self.target_class = 0
        else:
            self.miss_count += 1
            if self.miss_count > MISS_TOLERANCE:
                self.state = self.STATE_SLEEP
                self.target_class = 0
                self.confirmations = 0
                self.miss_count = 0

    def format_status(self, peak_volume: float) -> str:
        """Форматує стан. Прогресивні повідомлення від підозри до SOS."""
        hits = f"{self.confirmations}/{CONFIRM_THRESHOLD}"
        
        if self.state in (self.STATE_ALARM, self.STATE_ATTENTION):
            # Рахуємо відстань ТІЛЬКИ для цілі (дрона/машини)
            dist_m = int(min(200, 6.0 / max(peak_volume, 1e-4)))
            dist_str = f"~{dist_m}m"
        else:
            # Немає сенсу міряти відстань до "тиші"
            dist_str = "---"

        if self.state == self.STATE_ALARM:
            # ═══ ПІДТВЕРДЖЕНО ═══
            target = CLASS_NAMES.get(self.target_class, "?")
            header = f"🆘 SOS {target}!!!"
            conf_pct = f"{self.target_conf:.0%}"
        elif self.state == self.STATE_ATTENTION:
            # ═══ ПІДОЗРА ═══
            target = CLASS_NAMES.get(self.target_class, "?")
            header = f"👀 Схоже на {target}..."
            conf_pct = f"{self.target_conf:.0%}"
        elif self.state == self.STATE_LISTENING:
            header = "👂 Слухаю..."
            conf_pct = "---"
        else:
            header = "💤 Тиша"
            conf_pct = "---"

        parts = [
            header,
            f"conf={conf_pct}",
            f"hits={hits}",
            f"DOA={self.current_angle:3d}°",
            f"dist={dist_str}",
        ]

        if self.miss_count > 0:
            parts.append(f"miss={self.miss_count}/{MISS_TOLERANCE}")

        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Консольний UI (Anti-spam: один динамічний рядок)
# ═══════════════════════════════════════════════════════════════

def print_status(text: str):
    """
    Виводить статус в один рядок через \\r (без перенесення).

    - Отримує ширину термінала через shutil.get_terminal_size()
    - Жорстко обрізає рядок до (ширина - 2) символів
    - Використовує .ljust() для затирання старих залишків
    """
    cols = shutil.get_terminal_size().columns
    max_len = cols - 2
    # Обрізаємо та доповнюємо пробілами
    line = text[:max_len].ljust(max_len)
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
#  Головний цикл (Real-time інференс)
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🎯 Acoustic Radar — Smart Tracking v1.0")
    print("=" * 60)
    print(f"   Модель:       {ONNX_MODEL_PATH}")
    print(f"   Аудіо:        {SAMPLE_RATE} Hz, {CHANNELS} ch, "
          f"chunk={CHUNK_DURATION}s")
    print(f"   UART:         {UART_PORT} @ {UART_BAUD}")
    print(f"   Noise Gate:   {NOISE_GATE} (чутливість до ~200м)")
    print(f"   Тривога:      {CONFIRM_THRESHOLD} підтверджень, "
          f"conf > {CONFIDENCE_MIN:.0%}")
    print()

    # ── 1. ONNX Runtime ─────────────────────────────────────
    print("🔧 Ініціалізація...")
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    print(f"   ✅ ONNX модель: {ONNX_MODEL_PATH}")

    # ── 2. Mel-фільтри (обчислюємо один раз) ────────────────
    mel_fb = create_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    mel_fb = mel_fb.astype(np.float32)
    print(f"   ✅ Mel filterbank: {N_MELS} filters, "
          f"FFT={N_FFT}, hop={HOP_LENGTH}")

    # ── 3. UART ──────────────────────────────────────────────
    uart = None
    try:
        uart = serial.Serial(UART_PORT, UART_BAUD, timeout=0.1)
        print(f"   ✅ UART: {UART_PORT} @ {UART_BAUD}")
    except Exception as e:
        print(f"   ⚠️  UART недоступний: {e}")
        print(f"       Працюємо без UART (кут лише в терміналі)")

    # ── 4. State Machine ─────────────────────────────────────
    tracker = SmartTracker()

    # ── 5. Запуск аудіо потоку ───────────────────────────────
    print(f"\n🎤 Прослуховування ефіру... (Ctrl+C для зупинки)\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:

            while True:
                # ── Читання чанка аудіо ──
                audio_chunk, overflowed = stream.read(CHUNK_SAMPLES)
                current_time = time.time()

                # Перевірка overflow (пропущені семпли)
                if overflowed:
                    print_status("⚠️  OVERFLOW — аудіо буфер переповнений")
                    continue

                # audio_chunk shape: [CHUNK_SAMPLES, CHANNELS]
                # Канали: ch1 = audio_chunk[:, 0], ch2 = audio_chunk[:, 1]

                # ── Моно для інференсу (середнє каналів) ──
                mono = audio_chunk.mean(axis=1)

                # ── Noise Gate: перевірка гучності ──
                peak_volume = float(np.max(np.abs(mono)))

                if peak_volume < NOISE_GATE:
                    # Тиша — пропускаємо інференс
                    tracker.set_sleep()
                    print_status(
                        f"💤 [SLEEP] vol={peak_volume:.4f} "
                        f"< {NOISE_GATE} (тиша)"
                    )
                    continue

                # ══════════════════════════════════════════════
                #  Інференс (гучність > порогу)
                # ══════════════════════════════════════════════

                # ── Mel-спектрограма ──
                mel = compute_mel_spectrogram(
                    mono, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, mel_fb
                )

                # Формат для ONNX: [batch=1, channels=1, n_mels, n_frames]
                mel_input = mel[np.newaxis, np.newaxis, :, :].astype(
                    np.float32
                )

                # ── ONNX інференс ──
                logits = session.run(None, {input_name: mel_input})[0]

                # Softmax (NumPy)
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / exp_logits.sum()

                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[0, predicted_class])

                # ── DOA (Прямо з процесора мікрофона через USB) ──
                angle = get_usb_doa()

                # ── Оновлення State Machine ──
                send_uart = tracker.update(
                    predicted_class, confidence, angle, current_time
                )

                # ── Відправка кута по UART ──
                if send_uart and uart is not None:
                    try:
                        uart_msg = f"ANGLE:{tracker.current_angle:03d}\n"
                        uart.write(uart_msg.encode("ascii"))
                    except Exception:
                        pass  # Не блокуємо основний цикл

                # ── Вивід у термінал (один рядок) ──
                status = tracker.format_status(peak_volume)
                print_status(status)

    except KeyboardInterrupt:
        # Ctrl+C — коректне завершення
        print("\n\n🛑 Радар зупинено.")

    finally:
        if uart is not None:
            uart.close()
            print("   UART закрито.")
        print("   Bye! 👋")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

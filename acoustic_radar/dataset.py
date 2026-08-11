#!/usr/bin/env python3
"""
dataset.py — AudioDataset для навчання моделі акустичного радара.

Завантажує .wav файли з класифікованих директорій, перетворює їх
у Mel-спектрограми (дБ) та повертає пари (спектрограма, мітка класу).

Параметри Mel-спектрограми ідентичні в dataset.py та radar.py,
щоб забезпечити однаковий feature extraction під час навчання та інференсу.
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  Параметри Mel-спектрограми
#  ⚠️ Ці значення мають ТОЧНО збігатися з radar.py!
# ═══════════════════════════════════════════════════════════════

SAMPLE_RATE   = 16000    # Частота дискретизації (Гц)
N_MELS        = 32       # Кількість Mel-фільтрів
N_FFT         = 400      # Розмір FFT вікна (25 мс при 16 кГц)
HOP_LENGTH    = 160      # Крок FFT вікна (10 мс при 16 кГц)

# Фіксована кількість фреймів для 3-секундного аудіо при center=True:
# n_frames = (48000 + 400 - 400) / 160 + 1 = 301
TARGET_FRAMES = 301


# ═══════════════════════════════════════════════════════════════
#  AudioDataset
# ═══════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    """
    PyTorch Dataset для аудіо-класифікації.

    Очікує структуру директорій:
      root_dir/
        0/         ← .wav файли класу 0 (Фон)
        1/         ← .wav файли класу 1 (Дрон)
        2/         ← .wav файли класу 2 (Мотор)

    Кожен __getitem__ повертає:
      mel:   Tensor [1, N_MELS, TARGET_FRAMES]  — Mel-спектрограма в дБ
      label: int                                 — Мітка класу (0, 1, або 2)
    """

    def __init__(self, root_dir: str, sample_rate: int = SAMPLE_RATE):
        """
        Args:
            root_dir:    Шлях до кореневої директорії датасету.
            sample_rate: Цільова частота дискретизації.
        """
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.samples = []  # Список кортежів (шлях_до_файлу, мітка_класу)

        # ── Ініціалізація torchaudio трансформацій ──
        # Mel-спектрограма: конвертує waveform → mel power spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=True,           # Padding для збереження довжини
            pad_mode="reflect",    # Режим padding (як у librosa)
        )

        # Конвертація амплітуди в децибели (логарифмічна шкала)
        # top_db=80 обмежує динамічний діапазон
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

        # ── Сканування директорій класів ──
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                label = int(class_dir.name)
                wav_files = sorted(class_dir.glob("*.wav"))
                for audio_file in wav_files:
                    self.samples.append((audio_file, label))

        if not self.samples:
            raise RuntimeError(
                f"❌ Датасет порожній! Перевірте шлях: '{root_dir}'\n"
                f"   Очікувана структура: {root_dir}/0/*.wav, "
                f"{root_dir}/1/*.wav, {root_dir}/2/*.wav"
            )

        # Статистика
        class_counts = self._count_classes()
        print(f"📊 AudioDataset завантажено:")
        print(f"   Шлях:    {self.root_dir}")
        print(f"   Файлів:  {len(self.samples)}")
        print(f"   Класи:   {class_counts}")
        print(f"   Mel:     n_mels={N_MELS}, n_fft={N_FFT}, "
              f"hop={HOP_LENGTH}, frames={TARGET_FRAMES}")

    def _count_classes(self) -> dict:
        """Рахує кількість семплів у кожному класі."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        """Повертає загальну кількість семплів у датасеті."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Завантажує та обробляє один семпл.

        Повертає:
            mel:   Tensor [1, N_MELS, TARGET_FRAMES] — Mel-спектрограма (дБ)
            label: int — Мітка класу
        """
        filepath, label = self.samples[idx]

        # ── 1. Завантаження аудіо (soundfile замість torchaudio.load) ──
        audio_np, sr = sf.read(str(filepath), dtype="float32")
        # audio_np shape: [samples] або [samples, channels]

        # Конвертуємо в torch tensor [channels, samples]
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.T)

        # ── 2. Ресемплінг (якщо SR не збігається) ──
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # ── 3. Конвертація в моно ──
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ── 4. Mel-спектрограма → дБ ──
        mel = self.mel_transform(waveform)       # [1, N_MELS, T]
        mel = self.amplitude_to_db(mel)           # Логарифмічна шкала

        # ── 5. Фіксація довжини (padding / truncation) ──
        mel = self._fix_length(mel, TARGET_FRAMES)

        return mel, label

    @staticmethod
    def _fix_length(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Приводить спектрограму до фіксованої кількості фреймів.

        Якщо спектрограма коротша — доповнює нулями справа (zero-padding).
        Якщо довша — обрізає справа (truncation).
        """
        _, _, current_frames = mel.shape

        if current_frames < target_frames:
            # Доповнення нулями: pad(left, right) по останньому виміру
            pad_size = target_frames - current_frames
            mel = torch.nn.functional.pad(mel, (0, pad_size), value=0.0)
        elif current_frames > target_frames:
            # Обрізання
            mel = mel[:, :, :target_frames]

        return mel


# ═══════════════════════════════════════════════════════════════
#  Тест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Швидкий тест завантаження
    dataset = AudioDataset("dataset/train")
    mel, label = dataset[0]
    print(f"\n🔍 Тест: mel.shape={list(mel.shape)}, label={label}")
    print(f"   Mel range: [{mel.min():.1f}, {mel.max():.1f}] dB")

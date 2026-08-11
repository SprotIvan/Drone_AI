#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset для навчання акустичного радара.

⚠️ ГОЛОВНА ЗМІНА ПРОТИ ПОПЕРЕДНЬОЇ ВЕРСІЇ
Раніше тут використовувався torchaudio.transforms.MelSpectrogram, а на
Raspberry Pi (radar.py) — окрема ручна NumPy-реалізація. Вони давали
РІЗНІ спектрограми:
    • Mel-фільтри radar.py округлювались до цілих FFT-бінів → форма
      нижніх фільтрів (саме там гармоніки лопатей!) відрізнялась,
      cosine similarity падала до 0.73;
    • np.hanning() — симетричне вікно, torch.hann_window() — періодичне;
    • у radar.py нижні Mel-канали давали -25 дБ там, де torchaudio давав
      +4.6 дБ на тому самому сигналі.
Модель навчалась на одних числах, а на Pi отримувала інші.

Тепер ОБИДВА шляхи використовують features.MelFrontend — одну й ту саму
функцію. Це усуває розбіжність за побудовою, а не «на віру».
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from features import (
    MelFrontend, N_MELS, SAMPLE_RATE, TARGET_FRAMES, WINDOW_SAMPLES,
)

__all__ = ["AudioDataset", "N_MELS", "TARGET_FRAMES", "SAMPLE_RATE"]


# ═══════════════════════════════════════════════════════════════
#  SpecAugment (тільки для train)
# ═══════════════════════════════════════════════════════════════

def spec_augment(mel: np.ndarray,
                 freq_masks: int = 2, freq_width: int = 8,
                 time_masks: int = 2, time_width: int = 16) -> np.ndarray:
    """
    Маскує випадкові смуги частот і відрізки часу.

    Змушує модель спиратися на всю гармонічну структуру, а не на один
    зручний Mel-канал. Особливо важливо тут, бо вихідних записів дронів
    мало (два апарати), і без цього модель легко перенавчається на
    конкретний тембр.
    """
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    fill = float(mel.min())

    for _ in range(random.randint(0, freq_masks)):
        w = random.randint(1, freq_width)
        f0 = random.randint(0, max(0, n_mels - w))
        mel[f0:f0 + w, :] = fill

    for _ in range(random.randint(0, time_masks)):
        w = random.randint(1, time_width)
        t0 = random.randint(0, max(0, n_frames - w))
        mel[:, t0:t0 + w] = fill

    return mel


# ═══════════════════════════════════════════════════════════════
#  AudioDataset
# ═══════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    """
    Очікує структуру:
        root_dir/
          0/*.wav   ← клас 0 (Фон)
          1/*.wav   ← клас 1 (Дрон)

    __getitem__ повертає:
        mel:   Tensor [1, N_MELS, TARGET_FRAMES] — нормалізована спектрограма
        label: int
    """

    def __init__(self, root_dir: str | Path, augment: bool = False,
                 verbose: bool = True):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.frontend = MelFrontend()
        self.samples: list[tuple[Path, int]] = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                label = int(class_dir.name)
                for wav in sorted(class_dir.glob("*.wav")):
                    self.samples.append((wav, label))

        if not self.samples:
            raise RuntimeError(
                f"❌ Датасет порожній: '{self.root_dir}'\n"
                f"   Очікується {self.root_dir}/0/*.wav та "
                f"{self.root_dir}/1/*.wav\n"
                f"   Спочатку запустіть: python mixer.py"
            )

        if verbose:
            counts: dict[int, int] = {}
            for _, lbl in self.samples:
                counts[lbl] = counts.get(lbl, 0) + 1
            print(f"📊 AudioDataset  {self.root_dir}")
            print(f"   Файлів: {len(self.samples)}   Класи: "
                  f"{dict(sorted(counts.items()))}   "
                  f"Augment: {'так' if augment else 'ні'}")

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, lbl in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]

        audio, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)                       # → моно

        if sr != SAMPLE_RATE:
            # mixer.py пише вже 16 кГц; це лише запобіжник для власних записів
            n_out = int(round(len(audio) * SAMPLE_RATE / sr))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out),
                np.arange(len(audio)), audio,
            ).astype(np.float32)

        # ── Приводимо ХВИЛЮ до довжини вікна (не спектрограму) ──
        if len(audio) < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
        elif len(audio) > WINDOW_SAMPLES:
            start = (random.randint(0, len(audio) - WINDOW_SAMPLES)
                     if self.augment else 0)
            audio = audio[start:start + WINDOW_SAMPLES]

        mel = self.frontend(audio)                       # [N_MELS, T]
        mel = self.frontend.fix_length(mel, TARGET_FRAMES)

        if self.augment:
            mel = spec_augment(mel)

        return torch.from_numpy(mel).unsqueeze(0), label


# ═══════════════════════════════════════════════════════════════
#  Тест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ds = AudioDataset("dataset/train", augment=True)
    mel, label = ds[0]
    print(f"\n🔍 mel.shape={list(mel.shape)}  label={label}")
    print(f"   mean={mel.mean():+.3f}  std={mel.std():.3f}  "
          f"range=[{mel.min():.2f}, {mel.max():.2f}]")
    print(f"   (після нормалізації mean має бути ≈0, std ≈1)")

#!/usr/bin/env python3
"""
model.py — Легка згорткова нейромережа (AudioCNN) для акустичного радара.

Архітектура оптимізована для edge-інференсу на Raspberry Pi через ONNX Runtime.
Підтримує 3 класи: фон (0), дрон (1), мотор (2).

Ключова особливість: AdaptiveAvgPool2d дозволяє моделі приймати
спектрограми будь-якої довжини (3 сек навчання → 2 сек інференс).
"""

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """
    Компактна CNN для класифікації аудіо за Mel-спектрограмами.

    Архітектура:
    ┌──────────────────────────────────────────────────────┐
    │  Input:  [batch, 1, 32, T]  (Mel-спектрограма)      │
    │                                                      │
    │  Block 1:  Conv2d(1→16, 3×3) → BN → ReLU → Pool     │
    │  Block 2:  Conv2d(16→32, 3×3) → BN → ReLU → Pool    │
    │  Block 3:  Conv2d(32→64, 3×3) → BN → ReLU           │
    │            → AdaptiveAvgPool2d(1, 1)                 │
    │                                                      │
    │  Classifier:  Flatten → Dropout(0.3) → Linear(64→3) │
    │                                                      │
    │  Output: [batch, 3]  (logits для 3 класів)           │
    └──────────────────────────────────────────────────────┘

    Параметри: ~30K (залежить від num_classes)
    Час інференсу: <10 мс на Raspberry Pi 4 (ONNX Runtime)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        """
        Args:
            num_classes: Кількість класів (2: фон, дрон).
            dropout:     Ймовірність dropout перед фінальним шаром.
        """
        super().__init__()

        # ── Згорткові блоки (feature extraction) ──────────
        self.features = nn.Sequential(
            # Блок 1: вхід 1 канал → 16 фільтрів
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Після: [batch, 16, 16, T//2]

            # Блок 2: 16 → 32 фільтри
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Після: [batch, 32, 8, T//4]

            # Блок 3: 32 → 64 фільтри
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Після: [batch, 64, 8, T//4]

            # Адаптивне глобальне усереднення — вихід ЗАВЖДИ [64, 1, 1]
            # Це дозволяє приймати спектрограми будь-якої довжини T
            nn.AdaptiveAvgPool2d((1, 1)),
            # Після: [batch, 64, 1, 1]
        )

        # ── Класифікатор ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),              # [batch, 64]
            nn.Dropout(p=dropout),     # Регуляризація
            nn.Linear(64, num_classes),  # Фінальний шар
            # Без Softmax — CrossEntropyLoss включає LogSoftmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor [batch, 1, n_mels, n_frames]
               — Mel-спектрограма (1 канал, 32 mels, T фреймів)

        Returns:
            logits: Tensor [batch, num_classes] — Сирі логіти (до softmax)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Підраховує загальну кількість навчальних параметрів моделі."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
#  Самотест: перевірка розмірностей
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = AudioCNN(num_classes=3)
    total_params = model.count_parameters()

    print("=" * 50)
    print("🧠 AudioCNN — Self-Test")
    print("=" * 50)
    print(f"   Параметрів: {total_params:,}")
    print()

    # Тест з різними довжинами спектрограм
    # 301 фрейм = 3 сек (навчання), 201 фрейм = 2 сек (інференс)
    for label, frames in [("3 сек (train)", 301), ("2 сек (infer)", 201)]:
        dummy = torch.randn(1, 1, 32, frames)
        output = model(dummy)
        print(f"   {label}: input={list(dummy.shape)} → "
              f"output={list(output.shape)}")

    print(f"\n   ✅ Модель працює коректно")
    print()

    # Детальна структура
    print("📋 Архітектура:")
    print(model)

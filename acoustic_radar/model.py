#!/usr/bin/env python3
"""
model.py — Легка згорткова нейромережа (AudioCNN) для акустичного радара.

2 класи:  0 = Фон,  1 = Дрон.

⚠️ ЗМІНА ПРОТИ ПОПЕРЕДНЬОЇ ВЕРСІЇ — пулінг.
Раніше після згорток стояв `AdaptiveAvgPool2d((1, 1))`, який усереднював
одразу і по ЧАСУ, і по ЧАСТОТІ. Тобто модель бачила лише «скільки всього
енергії у кожному з 64 фільтрів», але не бачила, НА ЯКИХ ЧАСТОТАХ ця
енергія. Для дрона це критично: його підпис — це саме гребінка гармонік
частоти проходження лопатей у певній смузі.

Тепер усереднення відбувається лише по часу (сигнал стаціонарний, тому
це коректно і зберігає незалежність від довжини запису), а частотна вісь
стискається до 4 смуг і подається у класифікатор. Додано stat-pooling
(середнє + максимум), щоб короткі прольоти не «розмивались» усередненням.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """
    Компактна CNN для класифікації аудіо за Mel-спектрограмами.

    ┌────────────────────────────────────────────────────────────┐
    │  Вхід:  [B, 1, 64, T]   (нормалізована Mel-спектрограма)   │
    │                                                            │
    │  Блок 1:  Conv(1→16)  → BN → ReLU → MaxPool 2×2            │
    │  Блок 2:  Conv(16→32) → BN → ReLU → MaxPool 2×2            │
    │  Блок 3:  Conv(32→64) → BN → ReLU → MaxPool 2×1  (лише F)  │
    │                                → [B, 64, 8, T/4]           │
    │                                                            │
    │  Пулінг:  mean по часу ⊕ max по часу  → [B, 128, 8]        │
    │           avg_pool по частоті 8→4     → [B, 128, 4]        │
    │           flatten                     → [B, 512]           │
    │                                                            │
    │  Класифікатор: Dropout → Linear(512→64) → ReLU             │
    │                → Dropout → Linear(64→2)                    │
    └────────────────────────────────────────────────────────────┘

    Час інференсу: ~5 мс на Raspberry Pi 5 (ONNX Runtime, 1 потік).

    ⚠️ Пулінг по часу зроблено через mean()/amax(), а не через
    AdaptiveAvgPool2d. Причина суто практична: torch.onnx.export не вміє
    експортувати adaptive pooling, коли часова вісь оголошена динамічною
    («Unsupported: ONNX export of operator adaptive_avg_pool2d, input
    size not accessible»). Reduce-операції експортуються без проблем
    і дають той самий результат.
    """

    FREQ_BANDS = 4   # На скільки частотних смуг стискається вісь частот

    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 n_mels: int = 64):
        super().__init__()
        self.n_mels = n_mels

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [B,16,32,T/2]

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [B,32,16,T/4]

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Стискаємо лише частоту — часову роздільність зберігаємо,
            # щоб max-pooling нижче міг «зловити» короткий проліт.
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B,64,8,T/4]
        )

        # Після трьох стиснень частоти вдвічі лишається n_mels // 8 рядків.
        freq_rows = max(n_mels // 8, 1)
        self.freq_kernel = max(freq_rows // self.FREQ_BANDS, 1)
        out_bands = freq_rows // self.freq_kernel

        feat_dim = 64 * 2 * out_bands         # (mean ⊕ max) × смуги

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
            # Без Softmax — CrossEntropyLoss містить LogSoftmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, n_mels, n_frames] — нормалізована Mel-спектрограма
        Returns:
            logits: [B, num_classes]
        """
        x = self.features(x)                       # [B, 64, F, T']

        # Стискаємо ЧАС: сигнал дрона стаціонарний, тому середнє несе
        # основну інформацію, а максимум ловить короткий проліт.
        stats = torch.cat([x.mean(dim=3), x.amax(dim=3)], dim=1)  # [B,128,F]

        # Стискаємо ЧАСТОТУ до кількох смуг — але не до однієї, щоб
        # збереглося, НА ЯКИХ частотах зосереджена енергія.
        if self.freq_kernel > 1:
            stats = nn.functional.avg_pool1d(stats, self.freq_kernel)

        return self.classifier(stats)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
#  Самотест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from features import N_MELS, TARGET_FRAMES

    model = AudioCNN(num_classes=2, n_mels=N_MELS)
    print("=" * 58)
    print("🧠 AudioCNN — самотест")
    print("=" * 58)
    print(f"   Параметрів: {model.count_parameters():,}")
    print()

    model.eval()
    for label, frames in [(f"{TARGET_FRAMES} фреймів (робочий)", TARGET_FRAMES),
                          ("удвічі довше", TARGET_FRAMES * 2),
                          ("вдвічі коротше", TARGET_FRAMES // 2)]:
        dummy = torch.randn(2, 1, N_MELS, frames)
        with torch.no_grad():
            out = model(dummy)
        print(f"   {label:<28s} {list(dummy.shape)} → {list(out.shape)}")

    print("\n   ✅ Модель приймає спектрограми будь-якої довжини")

    # Експорт в ONNX з динамічною віссю часу — саме тут падала
    # попередня версія з AdaptiveAvgPool2d.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.onnx")
        torch.onnx.export(
            model, torch.randn(1, 1, N_MELS, TARGET_FRAMES), path,
            input_names=["mel_input"], output_names=["class_logits"],
            dynamic_axes={"mel_input": {0: "batch", 3: "time_frames"},
                          "class_logits": {0: "batch"}},
            opset_version=13, do_constant_folding=True)
        print(f"   ✅ Експорт в ONNX з динамічною віссю часу працює")
    print()

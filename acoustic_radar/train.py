#!/usr/bin/env python3
"""
train.py — Тренувальний pipeline для моделі акустичного радара.

Повний цикл навчання:
  1. Завантаження датасету (AudioDataset)
  2. Train/Val split (80/20)
  3. Навчання з Early Stopping (patience=5)
  4. Збереження найкращої моделі (.pth)
  5. Автоматичний експорт в ONNX для Raspberry Pi

Використання:
  python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time

from dataset import AudioDataset, N_MELS, TARGET_FRAMES
from model import AudioCNN


# ═══════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════

DATASET_DIR         = "dataset/train"       # Шлях до датасету
BATCH_SIZE          = 32                    # Розмір батчу
LEARNING_RATE       = 0.001                 # Швидкість навчання (Adam)
MAX_EPOCHS          = 50                    # Макс. кількість епох
EARLY_STOP_PATIENCE = 5                     # Терпіння Early Stopping
VAL_SPLIT           = 0.2                   # Частка валідаційної вибірки (20%)
NUM_CLASSES         = 2                     # Кількість класів (0: Фон, 1: Дрон)
NUM_WORKERS         = 0                     # 0 для Windows (уникає проблем з multiprocessing)

# Вихідні файли
BEST_MODEL_PATH = "best_model.pth"          # Найкраща модель (PyTorch)
ONNX_MODEL_PATH = "acoustic_radar.onnx"     # Експортована модель (ONNX)

# Пристрій: CUDA якщо доступний, інакше CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
#  Функції навчання та валідації
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Виконує одну епоху навчання.

    Returns:
        avg_loss:  float — Середній loss за епоху
        accuracy:  float — Точність за епоху (0.0 – 1.0)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in loader:
        mel = mel.to(device)           # [batch, 1, 32, 301]
        labels = labels.to(device)     # [batch]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(mel)           # [batch, 3]
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item() * mel.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Валідація моделі (без градієнтів).

    Returns:
        avg_loss:  float — Середній loss
        accuracy:  float — Точність (0.0 – 1.0)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in loader:
        mel = mel.to(device)
        labels = labels.to(device)

        outputs = model(mel)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * mel.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════
#  Експорт в ONNX
# ═══════════════════════════════════════════════════════════════

def export_to_onnx(model, onnx_path: str, device):
    """
    Експортує PyTorch модель у формат ONNX для Raspberry Pi.

    Використовує динамічну вісь часу, щоб модель приймала
    спектрограми будь-якої довжини (2 сек, 3 сек, тощо).
    """
    model.eval()

    # Dummy input: [batch=1, channels=1, n_mels=32, frames=301]
    dummy_input = torch.randn(1, 1, N_MELS, TARGET_FRAMES).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["mel_input"],
        output_names=["class_logits"],
        dynamic_axes={
            "mel_input": {0: "batch", 3: "time_frames"},
            "class_logits": {0: "batch"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    # Перевірка розміру файлу
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ✅ ONNX збережено: {onnx_path} ({size_kb:.1f} KB)")


# ═══════════════════════════════════════════════════════════════
#  Верифікація ONNX
# ═══════════════════════════════════════════════════════════════

def verify_onnx(onnx_path: str):
    """Перевіряє, що ONNX модель працює коректно."""
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        # Тест з різними довжинами
        for frames, label in [(301, "3 сек"), (201, "2 сек")]:
            dummy = np.random.randn(1, 1, N_MELS, frames).astype(np.float32)
            result = session.run(None, {input_name: dummy})
            output_shape = result[0].shape
            print(f"   ONNX тест ({label}): input=[1,1,32,{frames}] → "
                  f"output={list(output_shape)} ✅")

    except ImportError:
        print("   ⚠️  onnxruntime не встановлено, пропускаємо верифікацію")


# ═══════════════════════════════════════════════════════════════
#  Головний цикл навчання
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("🧠 Acoustic Radar — Model Training Pipeline")
    print("=" * 65)
    print(f"   Device:          {DEVICE}")
    print(f"   Max Epochs:      {MAX_EPOCHS}")
    print(f"   Early Stopping:  patience={EARLY_STOP_PATIENCE}")
    print(f"   Batch Size:      {BATCH_SIZE}")
    print(f"   Learning Rate:   {LEARNING_RATE}")
    print(f"   Val Split:       {VAL_SPLIT:.0%}")
    print()

    # ── 1. Датасет ──────────────────────────────────────────
    print("📂 Завантаження датасету...")
    dataset = AudioDataset(DATASET_DIR)

    # Train/Val split (детерміністично з seed=42)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"   Train: {train_size} семплів")
    print(f"   Val:   {val_size} семплів")

    # ── 2. Модель ───────────────────────────────────────────
    model = AudioCNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\n🔧 Модель: AudioCNN ({model.count_parameters():,} параметрів)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── 3. Тренування ───────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
          f"{'Val Loss':>10} │ {'Val Acc':>9} │ Status")
    print("──────┼────────────┼───────────┼────────────┼───────────┼"
          "──────────────────")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # Навчання
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Валідація
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        elapsed = time.time() - t0

        # ── Early Stopping logic ──
        if val_loss < best_val_loss:
            # Покращення: зберігаємо модель, скидаємо лічильник
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            status = f"✅ saved ({elapsed:.1f}s)"
        else:
            # Без покращення: збільшуємо лічильник
            patience_counter += 1
            status = (f"   [{patience_counter}/{EARLY_STOP_PATIENCE}] "
                      f"({elapsed:.1f}s)")

        # Вивід епохи
        print(f"{epoch:5d} │ {train_loss:10.4f} │ {train_acc:8.1%} │ "
              f"{val_loss:10.4f} │ {val_acc:8.1%} │ {status}")

        # Перевірка Early Stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early Stopping на епосі {epoch} "
                  f"(val_loss не покращувався {EARLY_STOP_PATIENCE} епох)")
            break

    # ── 4. Підсумки ─────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"🏆 Результати навчання:")
    print(f"   Найкраща Val Loss: {best_val_loss:.4f}")
    print(f"   Модель збережена:  {BEST_MODEL_PATH}")

    # ── 5. Експорт в ONNX ──────────────────────────────────
    print(f"\n📦 Експорт найкращої моделі в ONNX...")

    # Завантажуємо найкращі ваги
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    export_to_onnx(model, ONNX_MODEL_PATH, DEVICE)

    # Верифікація ONNX
    print(f"\n🔍 Верифікація ONNX моделі...")
    verify_onnx(ONNX_MODEL_PATH)

    # ── Фінал ──
    print()
    print("=" * 65)
    print(f"🎯 Готово! Наступні кроки:")
    print(f"   1. Скопіюй '{ONNX_MODEL_PATH}' на Raspberry Pi")
    print(f"   2. Запусти: python radar.py")
    print("=" * 65)


if __name__ == "__main__":
    main()

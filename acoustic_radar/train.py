#!/usr/bin/env python3
"""
train.py — Навчання моделі акустичного радара.

Що змінилось проти попередньої версії:

  1. ЧЕСНА ВАЛІДАЦІЯ.
     Раніше був `random_split(dataset, [0.8, 0.2])` вже ЗМІКШОВАНИХ файлів.
     Оскільки 2500 семплів класу 1 генерувались з 30 вихідних записів,
     той самий запис дрона потрапляв і в train, і в val. Val accuracy 96%
     означала «модель запам'ятала ці 30 записів», а не «модель впізнає дрон».
     Тепер mixer.py створює dataset/train і dataset/val із НЕПЕРЕТИЧНИХ
     вихідних файлів, і ми просто беремо їх як є.

  2. ПРАВИЛЬНІ МЕТРИКИ.
     Accuracy на збалансованому датасеті приховує, який саме клас
     провалюється. Тепер друкуються precision / recall / F1 по класу
     «дрон» і матриця плутанини.

  3. ПОРІГ ВИБИРАЄТЬСЯ, А НЕ ВГАДУЄТЬСЯ.
     `CONFIDENCE_MIN = 0.85` у radar.py було взято зі стелі. Тепер після
     навчання перебираються пороги, і обирається найменший, що дає
     хибнопозитивних не більше TARGET_FALSE_ALARM_RATE. Він зберігається
     у model_config.json, звідки його читає radar.py.

  4. model_config.json фіксує параметри front-end.
     radar.py звіряє їх при старті — якщо модель навчена з іншими
     n_mels/n_fft, ви побачите помилку, а не тихо погану детекцію.

Використання:
    python train.py
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import features
from dataset import AudioDataset
from model import AudioCNN


# ═══════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════

TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"

BATCH_SIZE          = 64
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
MAX_EPOCHS          = 60
EARLY_STOP_PATIENCE = 8
NUM_CLASSES         = 2
NUM_WORKERS         = 0          # 0 для Windows

# Максимально прийнятна частка хибних тривог на валідації.
# 2% означає: у середньому одна хибна тривога на 50 фонових вікон.
# SmartTracker у radar.py вимагає ще й кілька підтверджень поспіль,
# тому реальна частота хибних тривог буде значно нижчою.
TARGET_FALSE_ALARM_RATE = 0.02

# Нижня межа порогу. Валідаційна вибірка скінченна, тому «0% хибних
# тривог при порозі 0.15» може виявитись просто везінням на кількох
# сотнях семплів, а в полі перетворитись на постійне виття сирени.
# Поріг нижче 0.35 не приймаємо навіть якщо метрики його дозволяють.
MIN_DECISION_THRESHOLD = 0.35

BEST_MODEL_PATH = "best_model.pth"
ONNX_MODEL_PATH = "acoustic_radar.onnx"
CONFIG_PATH     = "model_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
#  Цикли навчання / валідації
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Returns:
        loss, accuracy, p_drone (ndarray), labels (ndarray)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_p, all_y = [], []

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        outputs = model(mel)
        total_loss += criterion(outputs, labels).item() * mel.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

        all_p.append(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        all_y.append(labels.cpu().numpy())

    return (total_loss / total, correct / total,
            np.concatenate(all_p), np.concatenate(all_y))


# ═══════════════════════════════════════════════════════════════
#  Метрики
# ═══════════════════════════════════════════════════════════════

def metrics_at(p_drone: np.ndarray, y: np.ndarray,
               threshold: float) -> dict:
    """Precision / recall / F1 для класу «дрон» при заданому порозі."""
    pred = p_drone >= threshold
    truth = y == 1

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    far = fp / max(fp + tn, 1)          # частка хибних тривог на фоні

    return {"threshold": float(threshold), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "false_alarm_rate": far}


def choose_threshold(p_drone: np.ndarray, y: np.ndarray,
                     max_far: float) -> dict:
    """
    Обирає найнижчий поріг, що тримає хибні тривоги в межах max_far.
    Нижчий поріг = вища чутливість до далеких дронів.
    """
    best = None
    for thr in np.arange(MIN_DECISION_THRESHOLD, 0.996, 0.005):
        m = metrics_at(p_drone, y, float(thr))
        if m["false_alarm_rate"] <= max_far:
            best = m
            break
    if best is None:
        # Навіть 0.995 не дає потрібної чистоти — беремо кращий за F1
        best = max((metrics_at(p_drone, y, float(t))
                    for t in np.arange(MIN_DECISION_THRESHOLD, 0.996, 0.005)),
                   key=lambda m: m["f1"])
        print("   ⚠️  Не вдалось досягти цільової частки хибних тривог — "
              "взято поріг за максимумом F1.")
    return best


def print_report(m: dict) -> None:
    print(f"\n   Поріг ухвалення рішення: {m['threshold']:.3f}")
    print(f"   ┌─────────────────┬──────────────┬──────────────┐")
    print(f"   │                 │ модель: фон  │ модель: дрон │")
    print(f"   ├─────────────────┼──────────────┼──────────────┤")
    print(f"   │ насправді фон   │ {m['tn']:12d} │ {m['fp']:12d} │")
    print(f"   │ насправді дрон  │ {m['fn']:12d} │ {m['tp']:12d} │")
    print(f"   └─────────────────┴──────────────┴──────────────┘")
    print(f"   Recall (скільки дронів спіймано):  {m['recall']:.1%}")
    print(f"   Precision (з тривог — справжні):   {m['precision']:.1%}")
    print(f"   F1:                                {m['f1']:.3f}")
    print(f"   Хибні тривоги на фоні:             "
          f"{m['false_alarm_rate']:.2%}")


# ═══════════════════════════════════════════════════════════════
#  Експорт
# ═══════════════════════════════════════════════════════════════

def export_to_onnx(model, onnx_path: str, device) -> None:
    """Експорт з динамічною віссю часу (модель приймає будь-яку довжину)."""
    model.eval()
    dummy = torch.randn(1, 1, features.N_MELS, features.TARGET_FRAMES).to(device)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["mel_input"],
        output_names=["class_logits"],
        dynamic_axes={"mel_input": {0: "batch", 3: "time_frames"},
                      "class_logits": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ✅ ONNX збережено: {onnx_path} ({size_kb:.1f} KB)")


def save_config(threshold: float, val_metrics: dict) -> None:
    """
    Записує параметри front-end і робочий поріг.
    radar.py читає цей файл і звіряє свої константи.
    """
    cfg = {
        "sample_rate":   features.SAMPLE_RATE,
        "n_fft":         features.N_FFT,
        "hop_length":    features.HOP_LENGTH,
        "n_mels":        features.N_MELS,
        "f_min":         features.F_MIN,
        "f_max":         features.F_MAX,
        "top_db":        features.TOP_DB,
        "window_sec":    features.WINDOW_SEC,
        "target_frames": features.TARGET_FRAMES,
        "normalize":     True,
        "num_classes":   NUM_CLASSES,
        "class_names":   {"0": "Фон", "1": "ДРОН"},
        "decision_threshold": float(threshold),
        "val_metrics":   {k: (float(v) if isinstance(v, float) else v)
                          for k, v in val_metrics.items()},
    }
    Path(CONFIG_PATH).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   ✅ Конфіг збережено: {CONFIG_PATH}")


def verify_onnx(onnx_path: str, model, val_loader, device) -> None:
    """Перевіряє, що ONNX дає ті самі числа, що й PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("   ⚠️  onnxruntime не встановлено — перевірку пропущено")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    mel, _ = next(iter(val_loader))
    mel = mel[:8]
    model.eval()
    with torch.no_grad():
        torch_out = model(mel.to(device)).cpu().numpy()
    onnx_out = sess.run(None, {inp: mel.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    status = "✅" if max_diff < 1e-3 else "❌"
    print(f"   {status} PyTorch vs ONNX: max |Δlogits| = {max_diff:.2e}")

    for frames in (features.TARGET_FRAMES // 2, features.TARGET_FRAMES * 2):
        dummy = np.random.randn(1, 1, features.N_MELS,
                                frames).astype(np.float32)
        out = sess.run(None, {inp: dummy})[0]
        print(f"   ✅ Динамічна довжина: [1,1,{features.N_MELS},{frames}] "
              f"→ {list(out.shape)}")


# ═══════════════════════════════════════════════════════════════
#  Головна функція
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("🧠 Acoustic Radar — навчання")
    print("=" * 68)
    print(f"   Пристрій:       {DEVICE}")
    print(f"   Front-end:      n_mels={features.N_MELS} "
          f"n_fft={features.N_FFT} hop={features.HOP_LENGTH} "
          f"вікно={features.WINDOW_SEC}с → {features.TARGET_FRAMES} фреймів")
    print(f"   Batch / LR:     {BATCH_SIZE} / {LEARNING_RATE}")
    print()

    if not Path(VAL_DIR).exists():
        raise SystemExit(
            f"❌ Немає '{VAL_DIR}'.\n"
            f"   Валідація має бути на НЕЗАЛЕЖНИХ вихідних записах.\n"
            f"   Перегенеруйте датасет:  python mixer.py"
        )

    print("📂 Завантаження датасету...")
    train_ds = AudioDataset(TRAIN_DIR, augment=True)
    val_ds = AudioDataset(VAL_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=(DEVICE.type == "cuda"))

    model = AudioCNN(num_classes=NUM_CLASSES,
                     n_mels=features.N_MELS).to(DEVICE)
    print(f"\n🔧 AudioCNN: {model.count_parameters():,} параметрів")

    # Ваги класів — на випадок незбалансованого датасету
    counts = train_ds.class_counts()
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(c, 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE)
    print(f"   Ваги класів: {weights.cpu().numpy().round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_f1, patience = -1.0, 0

    print(f"\n{'Епоха':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
          f"{'Val Loss':>9} │ {'Val Acc':>8} │ {'Val F1':>7} │ Статус")
    print("──────┼────────────┼───────────┼───────────┼──────────┼"
          "─────────┼──────────────")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, DEVICE)
        va_loss, va_acc, p_drone, y = evaluate(model, val_loader, criterion,
                                               DEVICE)
        # Стежимо за F1 при нейтральному порозі 0.5 — він відображає
        # реальну корисність краще, ніж val_loss.
        f1 = metrics_at(p_drone, y, 0.5)["f1"]
        scheduler.step(f1)
        dt = time.time() - t0

        if f1 > best_f1:
            best_f1, patience = f1, 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            status = f"✅ збережено ({dt:.0f}с)"
        else:
            patience += 1
            status = f"   [{patience}/{EARLY_STOP_PATIENCE}] ({dt:.0f}с)"

        print(f"{epoch:5d} │ {tr_loss:10.4f} │ {tr_acc:8.1%} │ "
              f"{va_loss:9.4f} │ {va_acc:7.1%} │ {f1:7.3f} │ {status}")

        if patience >= EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early Stopping на епосі {epoch}")
            break

    # ── Фінальна оцінка найкращої моделі ──
    print("\n" + "=" * 68)
    print("🏆 Оцінка найкращої моделі на валідації")
    print("=" * 68)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    _, _, p_drone, y = evaluate(model, val_loader, criterion, DEVICE)

    print("\n   Компроміс чутливість / хибні тривоги:")
    print("   поріг │ recall │ хибні тривоги │   F1")
    print("   ──────┼────────┼───────────────┼───────")
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        m = metrics_at(p_drone, y, thr)
        print(f"    {thr:.2f} │ {m['recall']:6.1%} │ "
              f"{m['false_alarm_rate']:13.2%} │ {m['f1']:.3f}")

    best = choose_threshold(p_drone, y, TARGET_FALSE_ALARM_RATE)
    print_report(best)

    if best["recall"] < 0.6:
        print("\n   ⚠️  Recall нижче 60%. Найімовірніші причини:")
        print("       • замало РІЗНИХ вихідних записів дронів")
        print("       • у raw_audio/drone/ потрапили не-дрони")
        print("       Додайте власні польові записи у raw_audio/drone/.")

    # ── Експорт ──
    print(f"\n📦 Експорт...")
    export_to_onnx(model, ONNX_MODEL_PATH, DEVICE)
    save_config(best["threshold"], best)
    print(f"\n🔍 Перевірка ONNX...")
    verify_onnx(ONNX_MODEL_PATH, model, val_loader, DEVICE)

    print("\n" + "=" * 68)
    print("🎯 Готово. Наступні кроки:")
    print(f"   1. Скопіювати на Raspberry Pi: {ONNX_MODEL_PATH}, "
          f"{CONFIG_PATH}, features.py, radar.py, doa.py, ranging.py")
    print(f"   2. Відкалібрувати відстань і кут:  python calibrate.py")
    print(f"   3. Запустити:  python radar.py    (або python radar_gui.py)")
    print("=" * 68)


if __name__ == "__main__":
    main()

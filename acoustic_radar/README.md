# Acoustic Radar (Акустичний Радар) 🚁🎯

Автономна система акустичного виявлення та відстеження дронів у реальному часі на базі **Raspberry Pi**, **ReSpeaker 4-Mic Array (XVF3800)** та легких згорткових нейромереж (**AudioCNN / ONNX**).

---

## 📌 Особливості системи

- **Класифікація у реальному часі:** Розпізнавання двох класів за Mel-спектрограмами:
  - `0`: **Фон (Ambient / Speech)**
  - `1`: **Дрон (Drone)**
- **Симуляція фізики відстані:** Генератор датасету симулює згасання звуку (законом $1/r$) та атмосферне поглинання високих частот (Low-pass) для дистанцій від 5м до 200м.
- **Апаратний 360° DOA (Direction of Arrival):** Зчитування кута напрямку на ціль безпосередньо з DSP-процесора ReSpeaker XVF3800 через USB.
- **Фільтрація хибних спалахів (50°):** Логіка `SmartTracker` фіксує напрямок на ціль та ігнорує розсіяні звукові відлуння чи стрибки кута понад 50°.
- **Графічний UI (Pygame):** Круговий радар на 360 градусів із візуалізацією дистанції, анімацією сканування, сектором захоплення (±25°) та світлозвуковими індикаторами тривоги.

---

## 🛠 Структура проєкту

```text
acoustic_radar/
├── download_dads.py    # Завантаження сирих звуків (DroneAudioDataset & ESC-50)
├── mixer.py            # Генератор синтетичного датасету з симуляцією відстаней
├── dataset.py          # PyTorch Dataset трансформації (Mel-Spectrogram)
├── model.py            # Легка CNN архітектура (AudioCNN)
├── train.py            # Тренувальний пайплайн (Early Stopping, експорт в ONNX)
├── radar.py            # Основний модуль інференсу та контролер SmartTracker
├── radar_gui.py        # Графічний інтерфейс радару (Pygame)
├── requirements.txt    # Залежності проекту
├── .gitignore          # Правила виключення файлів для Git
└── README.md           # Документація
```

---

## 🚀 Швидкий старт

### 1. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 2. Завантаження даних та підготовка датасету
```bash
# 1. Завантажити сирі фонові звуки та записи дронів
python download_dads.py

# 2. Згенерувати 5000 синтетичних семплів (2500 Ambient / 2500 Drone)
python mixer.py
```

### 3. Навчання та експорт моделі у формат ONNX
```bash
python train.py
```
Після успішного навчання у корені з'явиться файл `acoustic_radar.onnx`.

---

## 🍏 Запуск на Raspberry Pi

### 1. Встановлення системних залежностей для USB
```bash
sudo apt-get update
sudo apt-get install libusb-1.0-0-dev -y
```

### 2. Налаштування правил доступу udev (без sudo)
```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="001a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-respeaker.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3. Запуск інтерфейсу радара
```bash
python radar_gui.py
```

---

## 📜 Ліцензія
MIT License

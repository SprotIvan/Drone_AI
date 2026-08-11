"""
Завантажує звуки дронів із DroneAudioDataset (GitHub, прямі файли).
Також завантажує ambient з ESC-50.
"""
import urllib.request
import zipfile
import shutil
from pathlib import Path

DRONE_DIR   = Path("raw_audio/drone")
AMBIENT_DIR = Path("raw_audio/ambient")
MOTOR_DIR   = Path("raw_audio/motor")

DRONE_DIR.mkdir(parents=True, exist_ok=True)
AMBIENT_DIR.mkdir(parents=True, exist_ok=True)
MOTOR_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════
#  1. Дрони — DroneAudioDataset (GitHub ZIP ~50 МБ)
# ═══════════════════════════════════════════════════
DRONE_URL = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"
DRONE_ZIP = "drone_dataset.zip"

print("=" * 55)
print("1/2: Downloading drone sounds...")
print("     Source: DroneAudioDataset (GitHub)")
print(f"     URL: {DRONE_URL}")
print("=" * 55)

# Clean up any 0-byte empty files in DRONE_DIR
for f in DRONE_DIR.glob("*.wav"):
    if f.stat().st_size == 0:
        f.unlink()

existing = len(list(DRONE_DIR.glob("*.wav")))
print(f"  Info: Found {existing} valid drone files, downloading dataset...")
print("  Downloading ZIP...")
urllib.request.urlretrieve(DRONE_URL, DRONE_ZIP)
print("  Extracting...")

with zipfile.ZipFile(DRONE_ZIP, 'r') as z:
    wav_files = [f for f in z.namelist() if f.lower().endswith('.wav')]
    print(f"  Found {len(wav_files)} WAV files in archive")
    
    count = 0
    for wf in wav_files:
        fname = Path(wf).name
        if not fname:
            continue
        target = DRONE_DIR / f"drone_gh_{count:03d}.wav"
        with z.open(wf) as src, open(target, 'wb') as dst:
            dst.write(src.read())
        count += 1
        print(f"    Loaded {target.name}")
        if count >= 30:
            break

Path(DRONE_ZIP).unlink(missing_ok=True)
total = len(list(DRONE_DIR.glob("*.wav")))
print(f"\n  Drone files: +{count} new, total {total} in {DRONE_DIR}/")

# ═══════════════════════════════════════════════════
#  2. Ambient — ESC-50 (GitHub ZIP ~600 МБ)
# ═══════════════════════════════════════════════════
ESC_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ESC_ZIP = "esc50.zip"

print(f"\n{'=' * 55}")
print("2/2: Downloading ambient sounds (ESC-50)...")
print(f"     Source: ESC-50 (GitHub)")
print("=" * 55)

# Категорії ESC-50 для ambient (за номером класу у назві файлу)
# Формат назви: {fold}-{clip_id}-{take}-{target}.wav
# target: 10=rain, 11=sea_waves, 17=wind, 19=thunderstorm, 
#         22=chirping_birds, 24=insects
AMBIENT_TARGETS = {"10", "11", "17", "19", "22", "24"}

existing = len(list(AMBIENT_DIR.glob("*.wav")))
print(f"  Info: Found {existing} ambient files...")

if existing < 30:
    print("  Downloading ZIP (~600 MB)...")
    urllib.request.urlretrieve(ESC_URL, ESC_ZIP)
    print("  Extracting ambient categories...")

    with zipfile.ZipFile(ESC_ZIP, 'r') as z:
        wav_files = [f for f in z.namelist() 
                     if f.lower().endswith('.wav') and '/audio/' in f]
        
        count = 0
        for wf in wav_files:
            fname = Path(wf).name
            parts = fname.replace('.wav', '').split('-')
            if len(parts) >= 4 and parts[3] in AMBIENT_TARGETS:
                target = AMBIENT_DIR / f"esc_{fname}"
                if not target.exists():
                    with z.open(wf) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
                    count += 1
        
        print(f"    Added {count} new ambient files")

    Path(ESC_ZIP).unlink(missing_ok=True)

total = len(list(AMBIENT_DIR.glob("*.wav")))
print(f"\n  Ambient total: {total} files in {AMBIENT_DIR}/")

# ═══════════════════════════════════════════════════
print(f"\n{'=' * 55}")
print("Summary:")
print(f"   drone/:   {len(list(DRONE_DIR.glob('*.wav')))} files")
print(f"   ambient/: {len(list(AMBIENT_DIR.glob('*.wav')))} files")
print(f"{'=' * 55}")

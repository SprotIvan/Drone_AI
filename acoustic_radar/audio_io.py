#!/usr/bin/env python3
"""
audio_io.py — Вибір і відкриття мікрофонного входу.

⚠️ ЩО БУЛО НЕ ТАК
Старий radar.py відкривав потік так:

    sd.InputStream(samplerate=16000, channels=2, dtype="float32",
                   blocksize=CHUNK_SAMPLES)

Тут не вказано `device`, тому бралось СИСТЕМНЕ УСТРОЙСТВО ЗА
ЗАМОВЧУВАННЯМ. На Raspberry Pi це часто HDMI-вхід, вбудована звукова
або перший-ліпший USB-пристрій — тобто радар міг слухати взагалі не
ReSpeaker і не розуміти, чому «нічого не детектиться».
Кількість каналів теж була зашита (2), хоча масив може віддавати 6.

Тепер:
  • пристрій шукається за підрядком імені (за замовчуванням "ReSpeaker");
  • кількість каналів визначається автоматично (щоб дістати СИРІ
    мікрофони для SRP-PHAT, якщо масив їх віддає);
  • при старті друкується, що саме відкрито — без здогадок.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class InputConfig:
    """Опис фактично відкритого входу."""
    device_index: int | None
    device_name: str
    channels: int
    sample_rate: int
    raw_array: bool          # True = ≥4 канали, доступний власний SRP-PHAT

    def describe(self) -> str:
        kind = ("сирі канали масиву" if self.raw_array
                else "оброблене стерео" if self.channels == 2
                else f"{self.channels} кан.")
        name = self.device_name or "за замовчуванням"
        return f"{name} — {self.channels} кан. @ {self.sample_rate} Гц ({kind})"


def list_input_devices() -> list[tuple[int, dict]]:
    """Усі пристрої, здатні записувати звук."""
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def print_input_devices() -> None:
    print("Доступні пристрої запису:")
    default_in = sd.default.device[0] if sd.default.device else None
    for idx, dev in list_input_devices():
        mark = " ← за замовчуванням" if idx == default_in else ""
        print(f"   [{idx:2d}] {dev['name']}  "
              f"({dev['max_input_channels']} кан., "
              f"{dev['default_samplerate']:.0f} Гц){mark}")


def find_device(name_hint: str | None) -> tuple[int | None, dict]:
    """
    Шукає пристрій за підрядком імені (без урахування регістру).

    Повертає (індекс, інформація). Індекс None = системний за замовчуванням.
    """
    if name_hint:
        for idx, dev in list_input_devices():
            if name_hint.lower() in dev["name"].lower():
                return idx, dev
        print(f"⚠️  Пристрій з «{name_hint}» у назві не знайдено — "
              f"беремо системний за замовчуванням.")
        print_input_devices()

    try:
        info = sd.query_devices(kind="input")
    except Exception:
        info = {"name": "?", "max_input_channels": 1}
    return None, info


def resolve_input(cfg: dict, sample_rate: int,
                  prefer_raw: bool = True) -> InputConfig:
    """
    Визначає, який пристрій і скільки каналів відкривати.

    prefer_raw=True — намагаємось відкрити всі доступні канали (до 6),
    щоб дістати сирі мікрофони. Якщо масив віддає лише 2 — так і буде.
    """
    idx, info = find_device(cfg.get("input_device"))
    max_ch = int(info.get("max_input_channels", 2)) or 1

    forced = cfg.get("input_channels")
    if forced:
        channels = min(int(forced), max_ch)
    elif prefer_raw:
        # 6 каналів XVF3800 = 4 мікрофони + 2 опорні
        channels = min(max_ch, 6) if max_ch >= 4 else min(max_ch, 2)
    else:
        channels = min(max_ch, 2)

    return InputConfig(device_index=idx,
                       device_name=str(info.get("name", "?")),
                       channels=max(channels, 1),
                       sample_rate=sample_rate,
                       raw_array=channels >= 4)


def open_stream(inp: InputConfig, blocksize: int) -> sd.InputStream:
    """Відкриває потік із падінням до 2 і далі 1 каналу, якщо драйвер відмовляє."""
    last_error: Exception | None = None
    for channels in (inp.channels, 2, 1):
        if channels > inp.channels:
            continue
        try:
            stream = sd.InputStream(
                samplerate=inp.sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                device=inp.device_index,
            )
            if channels != inp.channels:
                print(f"⚠️  {inp.channels} каналів недоступно "
                      f"({last_error}) — відкрито {channels}.")
                inp.channels = channels
                inp.raw_array = channels >= 4
            return stream
        except Exception as exc:            # sd.PortAudioError та інші
            last_error = exc
    raise RuntimeError(f"Не вдалось відкрити мікрофон: {last_error}")


def to_mono(block: np.ndarray) -> np.ndarray:
    """[samples, channels] → моно float32."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    return block.mean(axis=1).astype(np.float32, copy=False)


if __name__ == "__main__":
    import calibration

    print_input_devices()
    cfg = calibration.load()
    inp = resolve_input(cfg, 16000)
    print(f"\nБуде використано: {inp.describe()}")
    if not inp.raw_array:
        print("   ℹ️  Масив віддає менше 4 каналів — власний SRP-PHAT "
              "неможливий, напрямок береться з DSP по USB.")

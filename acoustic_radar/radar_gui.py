#!/usr/bin/env python3
"""
radar_gui.py — Графічний інтерфейс акустичного радара (Pygame).

⚠️ ЩО ЗМІНИЛОСЬ
Раніше цей файл містив ВЛАСНУ копію циклу інференсу: свій розрахунок
спектрограми, свій виклик моделі, свою формулу відстані
(`6.0 / peak_volume`). Копія розійшлась з radar.py — наприклад, GUI
взагалі не вмів надсилати кут по UART і рахував відстань інакше, ніж
консоль. Тепер уся обробка живе у radar.RadarEngine, а цей файл лише
малює.

Що показує інтерфейс:
  • кругова шкала 360° з кільцями дальності;
  • позначка цілі за напрямком і відстанню;
  • ДУГА невизначеності кута і КІЛЬЦЕ невизначеності відстані —
    щоб не створювати ілюзію точності, якої немає;
  • якщо напрямок невідомий — ціль показується як пульсуюче коло
    без напрямку, а не в довільній точці;
  • якщо відстань не відкалібрована — промінь напрямку без дистанції.

Запуск:
    python radar_gui.py
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
import pygame  # noqa: E402

import calibration  # noqa: E402
import features  # noqa: E402
from audio_io import open_stream, resolve_input  # noqa: E402
from radar import (  # noqa: E402
    BLOCK_SAMPLES, CONFIRM_THRESHOLD, RadarEngine, RadarStatus, UartSender,
)
from telegram_alert import TelegramAlerter  # noqa: E402
from audio_logger import AudioLogger  # noqa: E402

TELEGRAM_TOKEN = "8875007303:AAGb8p-3-_h8qJaUFVAICO4I2sXSCpbTwOE"

# Максимальна дальність шкали (метри)
MAX_RANGE_M = 250.0
RINGS_M = (50, 100, 150, 200, 250)


# ═══════════════════════════════════════════════════════════════
#  Спільний стан між потоками
# ═══════════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = RadarStatus()
        self.error: str | None = None
        self.running = True

    def set(self, status: RadarStatus) -> None:
        with self.lock:
            # Копіюємо поля, а не посилання — рушій змінює свій об'єкт на місці
            self.status = RadarStatus(
                state=status.state, p_drone=status.p_drone,
                p_smoothed=status.p_smoothed,
                confirmations=status.confirmations, misses=status.misses,
                threshold=status.threshold, level_dbfs=status.level_dbfs,
                gated=status.gated, angle_deg=status.angle_deg,
                angle_confidence=status.angle_confidence,
                angle_source=status.angle_source,
                angle_ambiguous=status.angle_ambiguous, range=status.range,
            )

    def get(self) -> RadarStatus:
        with self.lock:
            return self.status


shared = SharedState()


# ═══════════════════════════════════════════════════════════════
#  Фоновий потік обробки
# ═══════════════════════════════════════════════════════════════

def audio_thread(engine: RadarEngine, inp, use_uart: bool,
                 alerter: TelegramAlerter | None = None,
                 logger: AudioLogger | None = None) -> None:
    uart = UartSender() if use_uart else None
    try:
        stream = open_stream(inp, BLOCK_SAMPLES)
        with stream:
            while shared.running:
                block, _ = stream.read(BLOCK_SAMPLES)
                block_np = np.asarray(block)
                status = engine.process_block(block_np)
                if uart is not None:
                    uart.send(status)
                shared.set(status)

                # ── Автозапис аудіо ──
                if logger is not None:
                    logger.feed(block_np)
                    logger.on_status(
                        status.state, status.angle_deg,
                        status.range.format(),
                    )

                # ── Telegram тривога ──
                if alerter is not None and alerter.ready:
                    alerter.on_status(
                        status.state, status.angle_deg,
                        status.range.format(), status.p_smoothed,
                    )
    except Exception as exc:
        with shared.lock:
            shared.error = str(exc)
        print(f"Помилка аудіопотоку: {exc}")
    finally:
        if uart is not None:
            uart.close()


# ═══════════════════════════════════════════════════════════════
#  Малювання
# ═══════════════════════════════════════════════════════════════

BLACK        = (10, 10, 15)
GREEN_DARK   = (0, 70, 0)
GREEN_MID    = (0, 140, 0)
GREEN_BRIGHT = (0, 255, 0)
RED          = (255, 60, 60)
RED_DARK     = (110, 30, 30)
AMBER        = (255, 200, 50)
WHITE        = (235, 235, 235)
GRAY         = (110, 110, 110)
PANEL_BG     = (20, 20, 26)


def to_screen(cx: int, cy: int, angle_deg: float, radius_px: float
              ) -> tuple[int, int]:
    """
    Кут установки → екранні координати.

    0° = вгору (північ), далі за годинниковою стрілкою:
    90° = праворуч, 180° = вниз, 270° = ліворуч.
    """
    rad = math.radians(angle_deg - 90.0)
    return (int(cx + radius_px * math.cos(rad)),
            int(cy + radius_px * math.sin(rad)))


def draw_grid(screen, font, cx, cy, max_radius, px_per_m) -> None:
    for d in RINGS_M:
        r = int(d * px_per_m)
        if r > max_radius:
            continue
        pygame.draw.circle(screen, GREEN_DARK, (cx, cy), r, 1)
        label = font.render(f"{d} м", True, GRAY)
        screen.blit(label, (cx + 6, cy - r - 16))

    for ang in range(0, 360, 45):
        x, y = to_screen(cx, cy, ang, max_radius)
        pygame.draw.line(screen, GREEN_DARK, (cx, cy), (x, y), 1)
        lx, ly = to_screen(cx, cy, ang, max_radius + 18)
        label = font.render(f"{ang}°", True, GRAY)
        screen.blit(label, (lx - label.get_width() // 2,
                            ly - label.get_height() // 2))


def draw_sweep(screen, cx, cy, max_radius, sweep_angle) -> None:
    for i in range(0, 26, 2):
        x, y = to_screen(cx, cy, sweep_angle + i, max_radius)
        shade = max(0, 150 - i * 6)
        pygame.draw.line(screen, (0, shade, 0), (cx, cy), (x, y), 2)
    x, y = to_screen(cx, cy, sweep_angle, max_radius)
    pygame.draw.line(screen, GREEN_MID, (cx, cy), (x, y), 2)


def draw_target(screen, font, cx, cy, max_radius, px_per_m,
                st: RadarStatus) -> None:
    """Малює ціль з урахуванням того, що саме нам ВІДОМО."""
    alarm = st.state == "ALARM"
    colour = RED if alarm else AMBER
    pulse = 10 + math.sin(time.time() * 12.0) * 4 if alarm else 8

    # ── Напрямок невідомий: коло навколо центру ──
    if st.angle_deg is None:
        radius = int(max_radius * 0.55 + math.sin(time.time() * 6) * 6)
        pygame.draw.circle(screen, colour, (cx, cy), radius, 2)
        text = font.render("напрямок н/д", True, colour)
        screen.blit(text, (cx - text.get_width() // 2, cy - radius - 22))
        return

    angle = st.angle_deg
    # Ширина сектора невизначеності: чим нижча впевненість, тим ширше
    half_width = 12.0 + (1.0 - min(st.angle_confidence, 1.0)) * 20.0

    # ── Сектор напрямку ──
    for offset in (-half_width, half_width):
        x, y = to_screen(cx, cy, angle + offset, max_radius)
        pygame.draw.line(screen, RED_DARK, (cx, cy), (x, y), 1)

    if st.range.ok:
        r_px = min(st.range.distance_m * px_per_m, max_radius - 4)
        tx, ty = to_screen(cx, cy, angle, r_px)

        # Кільце невизначеності відстані вздовж сектора
        if st.range.lo_m is not None and st.range.hi_m is not None:
            lo_px = min(st.range.lo_m * px_per_m, max_radius)
            hi_px = min(st.range.hi_m * px_per_m, max_radius)
            for offset in np.linspace(-half_width, half_width, 9):
                p1 = to_screen(cx, cy, angle + offset, lo_px)
                p2 = to_screen(cx, cy, angle + offset, hi_px)
                pygame.draw.line(screen, RED_DARK, p1, p2, 1)

        if alarm:
            pygame.draw.circle(screen, colour, (tx, ty), int(pulse))
        pygame.draw.circle(screen, WHITE, (tx, ty), 5)
        caption = f"ДРОН {angle:.0f}° · {st.range.format()}"
    else:
        # Відстань невідома — малюємо промінь, а не точку
        x, y = to_screen(cx, cy, angle, max_radius - 4)
        pygame.draw.line(screen, colour, (cx, cy), (x, y), 3)
        tx, ty = to_screen(cx, cy, angle, max_radius * 0.62)
        if alarm:
            pygame.draw.circle(screen, colour, (tx, ty), int(pulse), 2)
        caption = f"ДРОН {angle:.0f}° · відстань н/д"

    if st.angle_ambiguous:
        # Дзеркальний напрямок за двома мікрофонами — показуємо обидва
        mx, my = to_screen(cx, cy, (180.0 - angle) % 360.0,
                           max_radius * 0.62)
        pygame.draw.circle(screen, RED_DARK, (mx, my), 5, 1)
        caption += " (±дзеркально)"

    label = font.render(caption, True, WHITE)
    screen.blit(label, (min(tx + 14, screen.get_width() - label.get_width() - 8),
                        ty - 8))


def draw_panel(screen, fonts, st: RadarStatus, calib_note: str | None) -> None:
    font_large, font_med, font_small = fonts
    rect = pygame.Rect(10, 10, 360, 214)
    pygame.draw.rect(screen, PANEL_BG, rect, border_radius=10)
    pygame.draw.rect(screen, GREEN_DARK, rect, 2, border_radius=10)

    if st.state == "ALARM":
        title, colour = "SOS ДРОН!", RED
    elif st.state == "TRACK":
        title, colour = "Підозра...", AMBER
    elif st.state == "LISTEN":
        title, colour = "Слухаю...", GREEN_BRIGHT
    else:
        title, colour = "Тиша", GRAY
    screen.blit(font_large.render(title, True, colour), (25, 22))

    rows = [
        (f"Підтверджень: {st.confirmations}/{CONFIRM_THRESHOLD}", WHITE),
        (f"Ймовірність: {st.p_smoothed:.0%}  (поріг {st.threshold:.0%})",
         WHITE),
    ]
    y = 86
    for text, col in rows:
        screen.blit(font_med.render(text, True, col), (25, y))
        y += 32

    small = [
        f"Рівень: {st.level_dbfs:.0f} dBFS   Фон: {st.range.noise_dbfs:.0f} dBFS",
        f"Напрямок: " + ("н/д" if st.angle_deg is None
                         else f"{st.angle_deg:.0f}° ({st.angle_source})"),
        f"Відстань: {st.range.format()}",
    ]
    for text in small:
        screen.blit(font_small.render(text, True, GRAY), (25, y))
        y += 22

    if calib_note:
        note = font_small.render(calib_note, True, AMBER)
        screen.blit(note, (25, y))


# ═══════════════════════════════════════════════════════════════
#  Головна функція
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    cfg = calibration.load()

    print("=" * 66)
    print("Acoustic Radar + Telegram + Audio Logger")
    print("=" * 66)

    engine = RadarEngine(cfg=cfg)
    inp = resolve_input(cfg, features.SAMPLE_RATE)
    engine.attach_input(inp)
    print(f"   Mikrofon: {inp.describe()}")
    print(f"   {calibration.describe(cfg)}")

    calib_note = None
    if not engine.ranger.calibrated:
        calib_note = "distance not calibrated: calibrate.py range"
        print(f"\n   {calib_note}")

    # ── Telegram ──
    alerter = TelegramAlerter(TELEGRAM_TOKEN)
    alerter.discover_chats()
    if alerter.ready:
        print(f"   Telegram: {len(alerter.chat_ids)} chat(s) connected")
        alerter.send_startup()
    else:
        print("   Telegram: no chats yet (send /start to bot)")

    # ── Audio Logger ──
    logger = AudioLogger()
    print(f"   Audio Logger: detections/ ({logger.detection_count} saved)")

    pygame.init()
    width, height = 1024, 600
    fullscreen = "--fullscreen" in sys.argv
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Acoustic Radar")
    clock = pygame.time.Clock()

    fonts = (pygame.font.Font(None, 58),
             pygame.font.Font(None, 30),
             pygame.font.Font(None, 22))
    font_small = fonts[2]

    cx, cy = int(width * 0.66), height // 2
    max_radius = height // 2 - 34
    px_per_m = max_radius / MAX_RANGE_M

    thread = threading.Thread(target=audio_thread,
                              args=(engine, inp, True, alerter, logger),
                              daemon=True)
    thread.start()

    sweep_angle = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE, pygame.K_q):
                running = False

        sweep_angle = (sweep_angle + 2.0) % 360.0
        st = shared.get()

        screen.fill(BLACK)
        draw_grid(screen, font_small, cx, cy, max_radius, px_per_m)
        draw_sweep(screen, cx, cy, max_radius, sweep_angle)
        if st.state in ("TRACK", "ALARM"):
            draw_target(screen, font_small, cx, cy, max_radius, px_per_m, st)
        draw_panel(screen, fonts, st, calib_note)

        if shared.error:
            err = font_small.render(f"Аудіо: {shared.error}", True, RED)
            screen.blit(err, (12, height - 26))

        pygame.display.flip()
        clock.tick(30)

    shared.running = False
    engine.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

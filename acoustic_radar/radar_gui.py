#!/usr/bin/env python3
"""
radar_gui.py — Графічний інтерфейс для акустичного радара.
Використовує Pygame для відмальовки радару та імпортує логіку з radar.py.
"""

import os
import sys
import time
import math
import threading
import numpy as np
import sounddevice as sd
import onnxruntime as ort

# Вимикаємо вітальне повідомлення pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Імпортуємо константи та логіку з нашого консольного радару
from radar import (
    SAMPLE_RATE, CHANNELS, CHUNK_SAMPLES, N_FFT, HOP_LENGTH, N_MELS,
    ONNX_MODEL_PATH, NOISE_GATE, MIC_DISTANCE, CONFIRM_THRESHOLD, CLASS_NAMES,
    create_mel_filterbank, compute_mel_spectrogram, SmartTracker, get_usb_doa
)

# ═══════════════════════════════════════════════════════════════
# Спільний стан (Shared State) між потоками
# ═══════════════════════════════════════════════════════════════
class RadarState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state_str = "SLEEP"
        self.target_class = 0
        self.conf = 0.0
        self.doa = 90
        self.dist_m = 200
        self.hits = 0
        self.vol = 0.0
        self.miss = 0

shared_state = RadarState()

# ═══════════════════════════════════════════════════════════════
# Аудіо потік (Background)
# ═══════════════════════════════════════════════════════════════
def audio_inference_thread():
    print("🔧 Ініціалізація ONNX (Background Thread)...")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    mel_fb = create_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS).astype(np.float32)
    tracker = SmartTracker()

    print("🎤 Аудіо потік запущено. Інференс працює у фоні.")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SAMPLES)
                current_time = time.time()
                
                mono = audio_chunk.mean(axis=1)
                peak_volume = float(np.max(np.abs(mono)))
                
                if peak_volume < NOISE_GATE:
                    tracker.set_sleep()
                    with shared_state.lock:
                        shared_state.state_str = tracker.state
                        shared_state.vol = peak_volume
                        shared_state.hits = tracker.confirmations
                        shared_state.miss = tracker.miss_count
                    continue
                    
                mel = compute_mel_spectrogram(mono, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, mel_fb)
                mel_input = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
                
                logits = session.run(None, {input_name: mel_input})[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                
                pred_class = int(np.argmax(probs))
                conf = float(probs[0, pred_class])
                
                # ── DOA (Прямо з процесора мікрофона через USB) ──
                angle = get_usb_doa()
                tracker.update(pred_class, conf, angle, current_time)
                
                dist_m = int(min(200, 6.0 / max(peak_volume, 1e-4)))
                
                with shared_state.lock:
                    shared_state.state_str = tracker.state
                    shared_state.target_class = tracker.target_class
                    shared_state.conf = tracker.target_conf
                    shared_state.doa = tracker.current_angle
                    shared_state.hits = tracker.confirmations
                    shared_state.vol = peak_volume
                    shared_state.dist_m = dist_m
                    shared_state.miss = tracker.miss_count

    except Exception as e:
        print(f"Помилка в аудіо потоці: {e}")


# ═══════════════════════════════════════════════════════════════
# Pygame UI (Main Thread)
# ═══════════════════════════════════════════════════════════════
def main_gui():
    pygame.init()
    
    # Можна змінити на повноекранний режим для RPi (pygame.FULLSCREEN)
    WIDTH, HEIGHT = 1024, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Acoustic Radar UI")
    clock = pygame.time.Clock()
    
    # ── Кольори ──
    BLACK = (10, 10, 15)
    GREEN_DARK = (0, 70, 0)
    GREEN_BRIGHT = (0, 255, 0)
    RED = (255, 50, 50)
    BLUE = (50, 150, 255)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    
    # ── Шрифти ──
    # Використовуємо дефолтний системний шрифт, щоб працювало всюди
    font_large = pygame.font.Font(None, 64)
    font_med = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    # ── Геометрія радару ──
    cx, cy = WIDTH // 2, HEIGHT // 2
    max_radius = min(WIDTH, HEIGHT) // 2 - 20
    px_per_m = max_radius / 200.0
    
    sweep_angle = 360.0
    
    # Запуск фонового потоку інференсу
    audio_t = threading.Thread(target=audio_inference_thread, daemon=True)
    audio_t.start()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Вихід по Esc
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        # Анімація сканування
        sweep_angle -= 1.5
        if sweep_angle < 0:
            sweep_angle = 360.0
            
        screen.fill(BLACK)
        
        # ── 1. Малюємо сітку радару (повне коло) ──
        for d in [50, 100, 150, 200]:
            r = int(d * px_per_m)
            pygame.draw.circle(screen, GREEN_DARK, (cx, cy), r, 2)
            
            # Підписи дистанції (тільки на верхній осі, щоб не забивати екран)
            lbl = font_small.render(f"{d}m", True, GRAY)
            screen.blit(lbl, (cx + 5, cy - r - 20))
            
        # Радіальні лінії (кожні 45 градусів)
        for ang in range(0, 360, 45):
            rad = math.radians(ang - 90)
            x = cx + max_radius * math.cos(rad)
            y = cy + max_radius * math.sin(rad)
            pygame.draw.line(screen, GREEN_DARK, (cx, cy), (x, y), 2)
            
            # Підписи кутів
            ang_lbl = font_small.render(f"{ang}°", True, GRAY)
            # Невеликий відступ для тексту
            tx = x + 10 * math.cos(rad)
            ty = y + 10 * math.sin(rad)
            screen.blit(ang_lbl, (tx - 10, ty - 10))
            
        # ── 2. Малюємо лінію сканування ──
        rad_sweep = math.radians(sweep_angle - 90)
        sx = cx + max_radius * math.cos(rad_sweep)
        sy = cy + max_radius * math.sin(rad_sweep)
        pygame.draw.line(screen, GREEN_BRIGHT, (cx, cy), (sx, sy), 3)
        
        # Заливка сектора (ефект післясвітіння)
        for i in range(1, 15):
            lag_rad = math.radians(sweep_angle + i - 90)
            lx = cx + max_radius * math.cos(lag_rad)
            ly = cy + max_radius * math.sin(lag_rad)
            color = (0, max(0, 200 - i*15), 0)
            pygame.draw.line(screen, color, (cx, cy), (lx, ly), 2)
        
        # ── 3. Читаємо стан і малюємо ціль (Blip) ──
        with shared_state.lock:
            state = shared_state.state_str
            t_class = shared_state.target_class
            conf = shared_state.conf
            doa = shared_state.doa
            dist = shared_state.dist_m
            hits = shared_state.hits
            vol = shared_state.vol
            miss = shared_state.miss
            
        if state in ("TRACK", "ALARM"):
            blip_color = RED
            
            # Конвертація DOA (0-359) у координати для 360 екрану
            # 0° (Північ) = Верх, 90° (Схід) = Право, 180° (Південь) = Низ, 270° (Захід) = Ліво
            target_rad = math.radians(doa - 90)
            
            # Відстань
            r_px = min(dist * px_per_m, max_radius)
            
            tx = cx + r_px * math.cos(target_rad)
            ty = cy + r_px * math.sin(target_rad)
            
            # Малюємо 50-градусний сектор захоплення цілі (±25° від основного кута)
            sec_left_rad = math.radians(doa - 25 - 90)
            sec_right_rad = math.radians(doa + 25 - 90)
            x_l = cx + max_radius * math.cos(sec_left_rad)
            y_l = cy + max_radius * math.sin(sec_left_rad)
            x_r = cx + max_radius * math.cos(sec_right_rad)
            y_r = cy + max_radius * math.sin(sec_right_rad)
            
            sector_color = (120, 40, 40)
            pygame.draw.line(screen, sector_color, (cx, cy), (int(x_l), int(y_l)), 1)
            pygame.draw.line(screen, sector_color, (cx, cy), (int(x_r), int(y_r)), 1)
            
            # Пульсація при тривозі
            if state == "ALARM":
                pulse = 12 + math.sin(time.time() * 15) * 6
                pygame.draw.circle(screen, blip_color, (int(tx), int(ty)), int(pulse))
            
            pygame.draw.circle(screen, WHITE, (int(tx), int(ty)), 6)
            
            # Текст біля цілі
            lbl = font_small.render(f"DRONE {doa}° | ~{dist}m (±25°)", True, WHITE)
            screen.blit(lbl, (int(tx) + 15, int(ty) - 10))

        # ── 4. Малюємо інформаційну панель (Лівий верхній кут) ──
        panel_rect = (10, 10, 340, 200)
        pygame.draw.rect(screen, (20, 20, 25), panel_rect, border_radius=10)
        pygame.draw.rect(screen, GREEN_DARK, panel_rect, 2, border_radius=10)
        
        if state == "ALARM":
            status_txt = font_large.render("SOS ДРОН!", True, RED)
        elif state == "TRACK":
            status_txt = font_large.render("Підозра...", True, (255, 200, 50))
        elif state == "LISTEN":
            status_txt = font_large.render("Слухаю...", True, GREEN_BRIGHT)
        else:
            status_txt = font_large.render("Тиша", True, GRAY)
            
        screen.blit(status_txt, (25, 20))
        
        # Статистика
        hits_txt = font_med.render(f"Хіти: {hits}/{CONFIRM_THRESHOLD}", True, WHITE)
        screen.blit(hits_txt, (25, 90))
        
        conf_str = f"Впевненість: {conf:.0%}" if state in ("TRACK", "ALARM") else "Впевненість: ---"
        conf_txt = font_med.render(conf_str, True, WHITE)
        screen.blit(conf_txt, (25, 125))
        
        vol_txt = font_small.render(f"Гучність: {vol:.4f} (Поріг: {NOISE_GATE})", True, GRAY)
        screen.blit(vol_txt, (25, 160))
        
        miss_txt = font_small.render(f"Пропуски: {miss}/5", True, GRAY)
        screen.blit(miss_txt, (200, 160))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS, плавна анімація
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_gui()

#!/usr/bin/env python3
"""
audio_logger.py — Автозапис аудіо при детекції дрона.

Зберігає WAV-файл із 10-секундним буфером (5 сек ДО тривоги + 5 сек ПІСЛЯ)
у папку detections/ з часовою міткою.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DETECTIONS_DIR = Path(__file__).with_name("detections")
BUFFER_SEC = 5          # секунд аудіо ДО тривоги
RECORD_AFTER_SEC = 5    # секунд аудіо ПІСЛЯ початку тривоги
SAMPLE_RATE = 16000


class AudioLogger:
    """
    Кільцевий буфер аудіо з автозбереженням при ALARM.

    ═══════════════════════════════════════════════════════════════
    ⚠️ ЗАПИС НА ДИСК НЕ ВІДБУВАЄТЬСЯ В АУДІО-ПОТОЦІ
    ═══════════════════════════════════════════════════════════════

    `feed()` викликається з real-time аудіо-потоку (acoustic_worker._loop),
    і цей потік має жорсткий бюджет: `stream.read()` віддає блок кожні
    block_seconds (за замовчуванням 0.25 с), і якщо цикл не повернувся до
    наступного блоку, кільцевий буфер PortAudio переповнюється і звук
    ВТРАЧАЄТЬСЯ назавжди.

    Раніше `_save_recording()` викликався прямо з `feed()`, тобто
    `wave.open(...).writeframes(...)` — синхронний запис ~320 КБ на
    SD-карту — виконувався всередині цього бюджету, і саме в момент
    тривоги, тобто рівно тоді, коли втрачати аудіо найгірше.

    Тепер `feed()` лише складає готовий масив у чергу, а пише окремий
    потік-письменник. Поведінка запису (що саме зберігається, коли
    починається і коли завершується) не змінена; змінилось тільки те, ЯКИЙ
    потік торкається диска.

    Черга обмежена: якщо диск настільки повільний, що записи не встигають,
    краще втратити ЗАПИС, ніж втратити ДЕТЕКЦІЮ.
    """

    #: Скільки готових записів може чекати на диск. Кожен ~320 КБ.
    MAX_PENDING_WRITES = 4

    def __init__(self, save_dir: Path = DETECTIONS_DIR,
                 sr: int = SAMPLE_RATE,
                 buffer_sec: float = BUFFER_SEC,
                 record_after: float = RECORD_AFTER_SEC):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.buffer_sec = buffer_sec
        self.record_after = record_after

        # Кільцевий буфер: зберігає останні buffer_sec секунд
        max_chunks = int(buffer_sec / 0.5) + 2  # з запасом
        self._ring: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = Lock()

        # Стан запису
        self._recording = False
        self._record_start: float = 0.0
        self._record_chunks: list[np.ndarray] = []
        # Upper bound on a single recording: pre-roll + post-roll, with
        # generous slack. Blocks are 0.5 s, so this caps one capture at a
        # bounded number of chunks no matter what the clock does.
        self._max_chunks = int((buffer_sec + record_after) / 0.5) + 8
        self._alarm_handled = False   # щоб не зберігати двічі на одну тривогу

        self.last_saved: str | None = None

        # ── Потік-письменник ──
        self._writes: "queue.Queue[tuple | None]" = queue.Queue(
            maxsize=self.MAX_PENDING_WRITES)
        self.dropped_writes = 0
        #: Час ОСТАННЬОГО фактичного запису на диск, мілісекунди. Вимірюється
        #: у потоці-письменнику, тому це реальна вартість дискового I/O на
        #: цій машині — не оцінка. Аудіо-потік її більше не платить.
        self.last_write_ms: float | None = None
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="audio-logger", daemon=True)
        self._writer.start()

    def feed(self, audio_block: np.ndarray) -> None:
        """Додає блок аудіо у кільцевий буфер."""
        mono = audio_block.mean(axis=1) if audio_block.ndim > 1 else audio_block
        with self._lock:
            self._ring.append(mono.copy())

            if self._recording:
                self._record_chunks.append(mono.copy())
                # ⚠️ monotonic, not time.time(): a Raspberry Pi has no RTC
                # and its wall clock jumps by decades at the first NTP sync.
                # A backward jump made `elapsed` permanently negative, so
                # this recording never terminated and _record_chunks grew
                # without bound in the audio thread (~64 KB/s forever).
                elapsed = time.monotonic() - self._record_start
                # Hard cap as a second line of defence, so the buffer is
                # bounded even if the time source misbehaves in some way not
                # anticipated here.
                if (elapsed >= self.record_after
                        or len(self._record_chunks) >= self._max_chunks):
                    self._save_recording()

    def on_status(self, state: str, angle_deg: float | None,
                  distance_str: str) -> None:
        """Викликається з основного циклу радара."""
        with self._lock:
            if state == "ALARM" and not self._alarm_handled:
                self._alarm_handled = True
                self._start_recording(angle_deg, distance_str)
            # ALARM_COASTING — та сама тривога, у якої тимчасово зник
            # сигнал. Якби вона скидала прапорець, кожен порив вітру
            # створював би новий WAV-файл на ту саму ціль.
            elif state not in ("ALARM", "TRACK", "ALARM_COASTING"):
                self._alarm_handled = False

    def _start_recording(self, angle: float | None, dist: str) -> None:
        """Починає запис: копіює буфер + продовжує записувати."""
        self._recording = True
        self._record_start = time.monotonic()
        # Копіюємо кільцевий буфер (аудіо ДО тривоги)
        self._record_chunks = list(self._ring)
        self._angle = angle
        self._dist = dist

    def _save_recording(self) -> None:
        """
        Завершує запис і ПЕРЕДАЄ його потоку-письменнику.

        ⚠️ Викликається з real-time аудіо-потоку, тому тут не має права
        траплятись жодного дискового I/O. Усе, що робиться тут — це
        конкатенація вже наявних масивів у пам'яті (дешево і детерміновано)
        і неблокуюче складання в чергу.
        """
        self._recording = False

        if not self._record_chunks:
            return

        audio = np.concatenate(self._record_chunks)
        self._record_chunks = []

        # Ім'я файлу з часовою міткою
        ts = time.strftime("%Y%m%d_%H%M%S")
        angle_s = f"_{self._angle:.0f}deg" if self._angle is not None else ""
        filename = f"drone_{ts}{angle_s}.wav"

        try:
            # put_nowait, НЕ put: блокування тут повернуло б рівно ту
            # проблему, заради якої існує цей потік.
            self._writes.put_nowait((filename, audio))
        except queue.Full:
            self.dropped_writes += 1
            print(f"   [REC] DROPPED {filename}: {self.MAX_PENDING_WRITES} "
                  f"recordings already waiting on disk (total dropped: "
                  f"{self.dropped_writes}). Detection is unaffected.")

    # ── Потік-письменник ───────────────────────────────────────

    def _writer_loop(self) -> None:
        while True:
            item = self._writes.get()
            if item is None:
                return
            filename, audio = item
            try:
                self._write_wav(filename, audio)
            except Exception as exc:
                # Провал запису не має права зупинити письменника, і тим
                # більше — торкнутись детекції.
                print(f"   [REC] write failed for {filename}: {exc}")

    def _write_wav(self, filename: str, audio: np.ndarray) -> None:
        """Фактичний дисковий I/O. Виконується ЛИШЕ у потоці-письменнику."""
        t0 = time.monotonic()
        filepath = self.save_dir / filename

        # Нормалізація до int16
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio_int16 = (audio * 32767).astype(np.int16)

        # Запис WAV
        with wave.open(str(filepath), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(audio_int16.tobytes())

        self.last_write_ms = (time.monotonic() - t0) * 1000.0
        duration = len(audio) / self.sr
        self.last_saved = filename
        print(f"   [REC] Saved {filename} ({duration:.1f}s, "
              f"disk write {self.last_write_ms:.0f} ms, writer thread)")

    def close(self, timeout: float = 5.0) -> None:
        """Дописати те, що вже в черзі, і зупинити письменника."""
        try:
            self._writes.put_nowait(None)
        except queue.Full:
            pass
        self._writer.join(timeout)

    @property
    def detection_count(self) -> int:
        """Кількість збережених детекцій."""
        return len(list(self.save_dir.glob("drone_*.wav")))

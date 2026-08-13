#!/usr/bin/env python3
"""
latency.py — Measured end-to-end timing of the detection chain.

Every number this module reports is measured on the running machine. None
of it is derived from configuration or from intent, because the two have
already disagreed once in this project: the detector was documented as
raising the alarm in 2.5 s and actually took 3.5 s, and nothing in the
system was measuring the difference.

═══════════════════════════════════════════════════════════════════
THE STAGES, AND WHY EACH ONE IS SEPARATE
═══════════════════════════════════════════════════════════════════

    audio_wait      Time the audio thread spends blocked inside
                    stream.read(). This is NOT a cost of our code — it is
                    the microphone filling a block — but it IS latency: a
                    drone that becomes audible just after a read returns
                    waits a whole block before anything sees it. Mean is
                    half a block, worst case a full block.

    window_fill     STRUCTURAL, not measured here. The classifier judges a
                    2 s window. A drone that has been audible for only
                    0.5 s occupies a quarter of that window, and the model
                    was trained on windows that were entirely drone. No
                    threshold change can remove this; only a shorter window
                    (which needs retraining) or a shorter hop can.

    features        Mel front-end on the 2 s window.

    inference       ONNX forward pass.

    confirm         Blocks the detector requires above P_START. Exact and
                    configuration-derived: confirm_blocks x BLOCK_SEC.

    publish_to_fuse Age of the acoustic observation when the fusion layer
                    actually consumed it. This is where UI rate limiting
                    shows up, and it is the one people forget.

    hud_render      Compositing one frame.

    led_write       Time from submit() to the hardware write completing.
                    Dominated by a subprocess spawn.

TOTAL is not the sum of the means: the stages overlap (the LED write runs
on its own thread while the UI renders). The reported total is the measured
age of the acoustic observation at the moment each output consumed it,
plus the confirmation time that preceded it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stage:
    """One measured stage. Bounded memory, thread-safe."""

    name: str
    unit: str = "ms"
    #: Ring of recent samples. Bounded so a long run cannot grow without
    #: limit — this station is expected to run for hours.
    _samples: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _count: int = 0
    _max: float = 0.0
    MAX_SAMPLES = 512

    def add(self, value: float) -> None:
        with self._lock:
            self._count += 1
            if value > self._max:
                self._max = value
            self._samples.append(value)
            if len(self._samples) > self.MAX_SAMPLES:
                del self._samples[:len(self._samples) - self.MAX_SAMPLES]

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def stats(self) -> Optional[tuple]:
        """(mean, p50, p95, max) over the retained window, or None."""
        with self._lock:
            if not self._samples:
                return None
            data = sorted(self._samples)
            n = len(data)
            mean = sum(data) / n
            p50 = data[n // 2]
            p95 = data[min(n - 1, int(n * 0.95))]
            return mean, p50, p95, self._max

    def line(self) -> str:
        stats = self.stats()
        if stats is None:
            return f"   {self.name:<18} NOT MEASURED (no samples)"
        mean, p50, p95, mx = stats
        return (f"   {self.name:<18} mean {mean:7.1f}  p50 {p50:7.1f}  "
                f"p95 {p95:7.1f}  max {mx:7.1f}  {self.unit}  "
                f"(n={self.count})")


class LatencyBudget:
    """
    All stages, in one place, shared between threads.

    Deliberately cheap: `record()` is a lock, an append and a compare. It is
    called at most a few hundred times a second on a machine whose whole
    problem is that it has no CPU to spare.
    """

    ORDER = ("audio_wait", "features", "inference", "block_total",
             "publish_to_fuse", "hud_render", "led_write")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stages: Dict[str, Stage] = {}
        self._lock = threading.Lock()
        #: Set once, from the configuration, so the report can state the
        #: confirmation time exactly rather than estimating it.
        self.block_sec: Optional[float] = None
        self.confirm_blocks: Optional[int] = None
        self.window_sec: Optional[float] = None

    def stage(self, name: str, unit: str = "ms") -> Stage:
        with self._lock:
            stage = self._stages.get(name)
            if stage is None:
                stage = Stage(name, unit)
                self._stages[name] = stage
            return stage

    def record(self, name: str, value_ms: float) -> None:
        if self.enabled:
            self.stage(name).add(value_ms)

    def describe_config(self, block_sec: float, confirm_blocks: int,
                        window_sec: float) -> None:
        self.block_sec = block_sec
        self.confirm_blocks = confirm_blocks
        self.window_sec = window_sec

    # ── Reporting ──────────────────────────────────────────────

    def report(self) -> List[str]:
        """The full budget, as lines. Absent stages say so explicitly."""
        out = ["LATENCY BUDGET (measured on this machine)"]

        if self.block_sec is not None:
            block_ms = self.block_sec * 1000.0
            out.append(f"   {'audio block':<18} {block_ms:7.0f} ms "
                       f"(configuration)")
            out.append(f"   {'analysis window':<18} "
                       f"{(self.window_sec or 0) * 1000:7.0f} ms "
                       f"(configuration)")
            if self.confirm_blocks:
                out.append(
                    f"   {'confirmation':<18} "
                    f"{self.confirm_blocks * block_ms:7.0f} ms "
                    f"({self.confirm_blocks} blocks x {block_ms:.0f} ms, "
                    f"exact)")

        for name in self.ORDER:
            stage = self._stages.get(name)
            out.append(stage.line() if stage is not None
                       else f"   {name:<18} NOT MEASURED — REQUIRES A RUN "
                            f"WITH THIS SUBSYSTEM ACTIVE")

        # The honest total: what an operator waits, from the drone becoming
        # audible to the indication moving.
        parts: List[float] = []
        if self.block_sec is not None:
            parts.append(self.block_sec * 1000.0 / 2.0)      # mean read wait
            if self.confirm_blocks:
                parts.append(self.confirm_blocks * self.block_sec * 1000.0)
        for name in ("block_total", "publish_to_fuse"):
            stats = self._stages[name].stats() if name in self._stages else None
            if stats:
                parts.append(stats[0])
        if parts:
            out.append(f"   {'-' * 60}")
            out.append(f"   {'TOTAL (mean)':<18} {sum(parts):7.0f} ms "
                       f"from audible to indication, EXCLUDING window fill")
            out.append("   window fill: a drone audible for less than the "
                       "analysis window is")
            out.append("   diluted inside it and scores lower — see the "
                       "module docstring.")
            out.append("   THAT TERM IS NOT MEASURED HERE: it needs a real "
                       "drone and a")
            out.append("   reference microphone, i.e. PHYSICAL HARDWARE "
                       "VERIFICATION REQUIRED.")
        return out


#: The station's single budget. A module-level instance so the audio
#: thread, the UI thread and the LED thread all write into the same object
#: without having to thread it through every constructor.
BUDGET = LatencyBudget()


if __name__ == "__main__":
    import random
    import time

    print("=" * 70)
    print("latency.py — self-test")
    print("=" * 70)

    BUDGET.describe_config(block_sec=0.5, confirm_blocks=2, window_sec=2.0)
    for _ in range(200):
        BUDGET.record("audio_wait", random.gauss(250, 40))
        BUDGET.record("features", random.gauss(4, 1))
        BUDGET.record("inference", random.gauss(7, 2))
        BUDGET.record("block_total", random.gauss(11, 3))
        BUDGET.record("publish_to_fuse", random.gauss(120, 60))
        BUDGET.record("hud_render", random.gauss(6, 2))
    print()
    for line in BUDGET.report():
        print(line)

    print("\nA stage nobody recorded is reported as NOT MEASURED, never as 0:")
    empty = LatencyBudget()
    print(empty.stage("led_write").line())

    print("\nBounded memory under a long run:")
    s = Stage("soak")
    t0 = time.monotonic()
    for i in range(200_000):
        s.add(float(i))
    print(f"   200,000 samples -> {len(s._samples)} retained, "
          f"count={s.count:,}, {time.monotonic() - t0:.2f}s")

#!/usr/bin/env python3
"""
station_logging.py — Logging setup for the unified station.

Two problems this solves that plain `logging.basicConfig` does not:

1. RATE LIMITING. The camera loop runs at ~30 Hz and the audio loop at
   2 Hz. A single unguarded DEBUG line inside either produces thousands of
   lines per minute, which makes the log useless and costs real CPU on a
   Pi. `RateLimitedLogger.every()` emits a message at most once per
   interval, per call site.

2. CHANGE-ONLY EVENTS. During a long track, "bearing is 142°" is not worth
   logging 2x/second. `log_on_change()` emits only when a value moves by
   more than a configured amount, so the event log reads as a narrative of
   what actually happened.

Console output stays terse (level + message). The optional rotating file
handler keeps full timestamps and module names for post-flight analysis.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from fusion_config import LoggingConfig

_CONSOLE_FMT = "%(levelname).1s %(name)-16s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def setup(config: LoggingConfig, base_dir: Optional[Path] = None
          ) -> logging.Logger:
    """
    Configure the root 'station' logger. Idempotent — safe to call twice.

    Returns the root station logger.
    """
    root = logging.getLogger("station")
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    root.propagate = False

    if root.handlers:                       # already configured
        return root

    # ── Console ──
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    # Windows terminals and some Pi consoles are not UTF-8; a UnicodeEncode
    # error raised from inside a log call would otherwise take down the
    # thread that logged it.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    root.addHandler(console)

    # ── Rotating file ──
    if config.to_file:
        try:
            log_dir = Path(base_dir or Path(__file__).parent) / config.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / "station.log", maxBytes=config.max_bytes,
                backupCount=config.backup_count, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FILE_FMT))
            fh.setLevel(logging.DEBUG)
            root.addHandler(fh)
            root.debug("file logging -> %s", log_dir / "station.log")
        except OSError as exc:
            # A read-only filesystem must not prevent the station running.
            root.warning("file logging disabled: %s", exc)

    return root


class RateLimiter:
    """
    Allows an action at most once per `interval` seconds, per key.

    Used both for logging and for any other per-frame side effect that
    should not run at frame rate.
    """

    def __init__(self, interval: float):
        self.interval = float(interval)
        self._last: Dict[str, float] = {}

    def allow(self, key: str, interval: Optional[float] = None) -> bool:
        now = time.monotonic()
        gap = self.interval if interval is None else interval
        previous = self._last.get(key)
        if previous is None or now - previous >= gap:
            self._last[key] = now
            return True
        return False

    def reset(self, key: str) -> None:
        self._last.pop(key, None)


class EventLogger:
    """
    Logs significant target events, and only significant ones.

    Wraps a stdlib logger with:
      • `rate(key, level, msg, ...)`  — at most once per interval
      • `on_change(key, value, ...)`  — only when the value moved enough
    """

    def __init__(self, logger: logging.Logger, config: LoggingConfig):
        self.log = logger
        self.config = config
        self._limiter = RateLimiter(config.rate_limit_s)
        self._values: Dict[str, float] = {}

    # ── Plain pass-through ──
    def debug(self, msg, *a): self.log.debug(msg, *a)
    def info(self, msg, *a): self.log.info(msg, *a)
    def warning(self, msg, *a): self.log.warning(msg, *a)
    def error(self, msg, *a): self.log.error(msg, *a)
    def exception(self, msg, *a): self.log.exception(msg, *a)

    # ── Rate limited ──
    def rate(self, key: str, level: int, msg: str, *args,
             interval: Optional[float] = None) -> None:
        if self._limiter.allow(key, interval):
            self.log.log(level, msg, *args)

    # ── Change triggered ──
    def on_change(self, key: str, value: Optional[float], msg: str,
                  *args, min_delta: float = 0.0,
                  min_frac: float = 0.0, level: int = logging.INFO) -> None:
        """
        Log only when `value` has moved by more than min_delta (absolute) or
        min_frac (relative to the previous value), or when it appears for
        the first time. Passing None forgets the previous value, so the next
        real reading is logged as new rather than compared against a stale
        one.
        """
        if value is None:
            self._values.pop(key, None)
            return
        previous = self._values.get(key)
        if previous is None:
            self._values[key] = value
            self.log.log(level, msg, *args)
            return
        delta = abs(value - previous)
        moved = (delta >= min_delta if min_delta > 0 else False)
        if not moved and min_frac > 0 and abs(previous) > 1e-9:
            moved = delta / abs(previous) >= min_frac
        if moved:
            self._values[key] = value
            self.log.log(level, msg, *args)

    def forget(self, *keys: str) -> None:
        for key in keys:
            self._values.pop(key, None)


if __name__ == "__main__":
    from fusion_config import load as load_config

    cfg = load_config()
    cfg.logging.to_file = False
    root = setup(cfg.logging)
    events = EventLogger(logging.getLogger("station.demo"), cfg.logging)

    print("=" * 66)
    print("station_logging.py — self-test")
    print("=" * 66)

    print("\n1. 500 rate-limited calls in a tight loop (interval 0.2 s):")
    limited = EventLogger(logging.getLogger("station.rate"), cfg.logging)

    class Counter(logging.Handler):
        count = 0

        def emit(self, record):
            Counter.count += 1

    counter = Counter()
    logging.getLogger("station.rate").addHandler(counter)
    start = time.monotonic()
    for _ in range(500):
        limited.rate("tick", logging.INFO, "tick", interval=0.2)
    print(f"   500 calls in {time.monotonic()-start:.3f}s -> "
          f"{Counter.count} line(s) actually emitted")

    print("\n2. Change-triggered bearing log (min_delta 10°):")
    for bearing in (142.0, 143.0, 145.0, 158.0, 159.0, 175.0):
        events.on_change("bearing", bearing,
                         "target bearing now %.0f deg", bearing,
                         min_delta=cfg.logging.bearing_change_deg)

    print("\n3. Change-triggered distance log (min_frac 25%):")
    for dist in (200.0, 195.0, 180.0, 140.0, 138.0, 100.0):
        events.on_change("distance", dist,
                         "target distance now ~%.0f m", dist,
                         min_frac=cfg.logging.distance_change_frac)
    print()

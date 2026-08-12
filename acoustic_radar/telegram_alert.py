#!/usr/bin/env python3
"""
telegram_alert.py — Сповіщення про дрон у Telegram.

Використання:
    1. Надішліть /start своєму боту у Telegram.
    2. Запустіть: python telegram_alert.py   (одноразово, для реєстрації chat_id)
    3. Далі модуль працює автоматично з radar_gui.py.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from threading import Thread, Lock

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CONFIG_PATH = Path(__file__).with_name("telegram_config.json")

# Мінімальний інтервал між сповіщеннями (секунди).
# Щоб не спамити під час тривалого ALARM.
COOLDOWN_SEC = 30.0


class TelegramAlerter:
    """Надсилає тривоги в Telegram із захистом від спаму."""

    def __init__(self, token: str, config_path: Path = CONFIG_PATH):
        self.token = token
        self.config_path = config_path
        self.chat_ids: list[int] = []
        self._last_sent: float = 0.0
        self._alarm_active = False
        self._lock = Lock()

        self._load_config()

    # ── Конфіг ─────────────────────────────────────────────

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.chat_ids = data.get("chat_ids", [])
            except Exception:
                pass

    def _save_config(self) -> None:
        data = {"chat_ids": self.chat_ids}
        self.config_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    # ── Telegram API ───────────────────────────────────────

    def _api(self, method: str, params: dict | None = None) -> dict | None:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        if params:
            import urllib.parse
            url += "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def _send_message(self, chat_id: int, text: str) -> bool:
        result = self._api("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        })
        return result is not None and result.get("ok", False)

    # ── Реєстрація (пошук /start) ─────────────────────────

    def discover_chats(self) -> int:
        """Опитує бота на наявність нових /start повідомлень.
        Повертає кількість нових чатів."""
        result = self._api("getUpdates", {"timeout": 1, "limit": 50})
        if not result or not result.get("ok"):
            return 0

        added = 0
        max_id = 0
        for upd in result.get("result", []):
            max_id = max(max_id, upd.get("update_id", 0))
            msg = upd.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "")
            if chat_id and chat_id not in self.chat_ids:
                self.chat_ids.append(chat_id)
                added += 1
                name = msg.get("chat", {}).get("first_name", "")
                print(f"   Telegram: +chat {chat_id} ({name})")

        # Підтверджуємо прочитані оновлення
        if max_id > 0:
            self._api("getUpdates", {"offset": max_id + 1, "limit": 1})

        if added:
            self._save_config()
        return added

    @property
    def ready(self) -> bool:
        return len(self.chat_ids) > 0

    # ── Надсилання тривоги ─────────────────────────────────

    def on_status(self, state: str, angle_deg: float | None,
                  distance_str: str, confidence: float) -> None:
        """Викликається з основного циклу радара."""
        with self._lock:
            if state == "ALARM":
                if not self._alarm_active:
                    self._alarm_active = True
                    self._try_send(angle_deg, distance_str, confidence)
            else:
                self._alarm_active = False

    def _try_send(self, angle: float | None, dist: str, conf: float) -> None:
        now = time.time()
        if now - self._last_sent < COOLDOWN_SEC:
            return
        self._last_sent = now

        ts = time.strftime("%H:%M:%S")
        angle_s = f"{angle:.0f}" if angle is not None else "n/a"

        text = (
            f"\xF0\x9F\x9A\x81 <b>DRONE DETECTED!</b>\n\n"
            f"\xF0\x9F\x93\x8D Direction: {angle_s}\xC2\xB0\n"
            f"\xF0\x9F\x93\x8F Distance: {dist}\n"
            f"\xF0\x9F\x8E\xAF Confidence: {conf:.0%}\n"
            f"\xE2\x8F\xB0 Time: {ts}"
        )

        # Надсилаємо у фоновому потоці, щоб не блокувати радар
        Thread(target=self._broadcast, args=(text,), daemon=True).start()

    def _broadcast(self, text: str) -> None:
        for cid in self.chat_ids:
            self._send_message(cid, text)

    def send_startup(self) -> None:
        """Надсилає повідомлення про запуск радара."""
        ts = time.strftime("%H:%M:%S")
        text = f"\xE2\x9C\x85 Acoustic Radar started\n\xE2\x8F\xB0 {ts}"
        self._broadcast(text)


# ── Автономний запуск для реєстрації ──────────────────────

if __name__ == "__main__":
    TOKEN = "8875007303:AAGb8p-3-_h8qJaUFVAICO4I2sXSCpbTwOE"
    alerter = TelegramAlerter(TOKEN)

    print("=" * 50)
    print("Telegram Alert Setup")
    print("=" * 50)
    print(f"   Token: ...{TOKEN[-8:]}")
    print(f"   Saved chats: {alerter.chat_ids}")
    print()
    print("   Waiting for /start messages...")
    print("   (send /start to your bot in Telegram)")
    print()

    for attempt in range(60):
        n = alerter.discover_chats()
        if alerter.ready:
            print(f"\n   OK! Registered {len(alerter.chat_ids)} chat(s).")
            alerter.send_startup()
            print("   Startup message sent. Check Telegram!")
            break
        time.sleep(2)
    else:
        print("\n   Timeout. No /start received in 2 minutes.")
        print("   Send /start to the bot and run this again.")

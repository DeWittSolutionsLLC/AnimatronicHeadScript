"""
idle_animator.py
----------------
Runs a background thread that makes subtle random eye movements
while the head is not speaking, giving it a more lifelike appearance.

Config (settings.json idle section):
  enabled       — turn on/off
  interval_min  — minimum seconds between movements
  interval_max  — maximum seconds between movements
  jitter        — max degrees of random offset from neutral
"""

import threading
import random
import time
import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


def _load_idle_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f).get("idle", {})
    except Exception:
        return {}


class IdleAnimator:
    def __init__(self, serial_controller):
        self._serial   = serial_controller
        self._speaking = False
        self._active   = False
        self._thread   = None
        self._load_config()

    def _load_config(self):
        cfg = _load_idle_config()
        self._enabled      = cfg.get("enabled",      True)
        self._interval_min = cfg.get("interval_min", 2.0)
        self._interval_max = cfg.get("interval_max", 5.0)
        self._jitter       = cfg.get("jitter",       12)

    def reload_config(self):
        self._load_config()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self._enabled:
            return
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False

    # ── Speaking gate ─────────────────────────────────────────────────────────

    def set_speaking(self, speaking: bool):
        """Pause idle movements while the head is speaking."""
        self._speaking = speaking

    # ── Background loop ───────────────────────────────────────────────────────

    def _run(self):
        while self._active:
            delay = random.uniform(self._interval_min, self._interval_max)
            time.sleep(delay)

            if self._speaking or not self._serial.is_connected():
                continue

            j = self._jitter
            ud = 90 + random.randint(-j, j)
            lr = 90 + random.randint(-j, j)

            self._serial.eyes_ud(ud)
            time.sleep(0.2)
            self._serial.eyes_lr(lr)
            time.sleep(0.3)

            # Drift back toward neutral
            self._serial.eyes_ud(90)
            time.sleep(0.2)
            self._serial.eyes_lr(90)

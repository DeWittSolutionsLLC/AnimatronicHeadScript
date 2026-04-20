"""
serial_controller.py
--------------------
Manages the serial connection to the Arduino and sends
servo commands in the format expected by head_control.ino:

  M<angle>  — mouth
  U<angle>  — eyes up/down
  L<angle>  — eyes left/right
  R         — reset all to neutral

Set serial.debug = true in settings.json to log every sent command.
Arduino confirmation strings (HEAD_READY, M:60, etc.) are read back
in a background thread and printed when serial.debug is true.
"""

import serial
import serial.tools.list_ports
import threading
import time
import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


def _load_serial_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f).get("serial", {})
    except Exception:
        return {}


class SerialController:
    def __init__(self):
        cfg = _load_serial_config()
        self.port      = cfg.get("port",      "COM3")
        self.baud_rate = cfg.get("baud_rate", 9600)
        self.timeout   = cfg.get("timeout",   2.0)
        self._mock     = cfg.get("mock",      False)
        self._debug    = cfg.get("debug",     False)
        self._ser      = None
        self._reader   = None

    def reload_config(self):
        cfg = _load_serial_config()
        self._debug = cfg.get("debug", False)

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if self._mock:
            print("[serial] MOCK MODE — no hardware required")
            return True
        try:
            self._ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(self.timeout)
            while self._ser.in_waiting:
                self._ser.readline()
            print(f"[serial] Connected to Arduino on {self.port}")
            self._start_reader()
            return True
        except serial.SerialException as e:
            print(f"[serial] Connection failed: {e}")
            print(f"[serial] Available ports: {self._list_ports()}")
            self._ser = None
            return False

    def disconnect(self):
        if self._mock:
            return
        if self._reader:
            self._reader_active = False
        if self._ser and self._ser.is_open:
            self.reset()
            self._ser.close()
            print("[serial] Disconnected.")

    def is_connected(self) -> bool:
        if self._mock:
            return True
        return self._ser is not None and self._ser.is_open

    # ── Servo commands ────────────────────────────────────────────────────────

    def send(self, cmd: str):
        if not self.is_connected():
            return
        if self._mock:
            if self._debug:
                print(f"[serial] MOCK >> {cmd}")
            return
        try:
            self._ser.write((cmd + "\n").encode())
            if self._debug:
                print(f"[serial] >> {cmd}")
            time.sleep(0.04)
        except serial.SerialException as e:
            print(f"[serial] Send error: {e}")

    def mouth(self, angle: int):
        self.send(f"M{_clamp(angle)}")

    def eyes_ud(self, angle: int):
        self.send(f"U{_clamp(angle)}")

    def eyes_lr(self, angle: int):
        self.send(f"L{_clamp(angle)}")

    def reset(self):
        self.send("R")

    def apply_emotion(self, positions: dict):
        if "eyes_ud" in positions:
            self.eyes_ud(positions["eyes_ud"])
        if "eyes_lr" in positions:
            self.eyes_lr(positions["eyes_lr"])

    # ── Background confirmation reader ────────────────────────────────────────

    def _start_reader(self):
        self._reader_active = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        while self._reader_active and self._ser and self._ser.is_open:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="replace").strip()
                    if line and self._debug:
                        print(f"[serial] << {line}")
                else:
                    time.sleep(0.05)
            except serial.SerialException:
                break

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _list_ports() -> list:
        return [p.device for p in serial.tools.list_ports.comports()]


def _clamp(angle: int) -> int:
    return max(0, min(180, int(angle)))

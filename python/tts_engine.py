"""
tts_engine.py
-------------
Text-to-speech with mouth animation.

Supports two engines (set tts.engine in settings.json):
  "pyttsx3"  — free, fully offline, uses OS voices
  "edge-tts" — Microsoft neural voices, requires internet
               pip install edge-tts pygame

Mouth angle scales with word length: short words open less,
long words open fully, for more natural animation.
"""

import time
import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


def _load_tts_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            s = json.load(f)
        return {**s.get("tts", {}), **s.get("mouth", {})}
    except Exception:
        return {}


class TTSEngine:
    def __init__(self, serial_controller=None):
        self._serial = serial_controller
        self._load_config()
        self._init_engine()

    def _load_config(self):
        cfg = _load_tts_config()
        self._engine_type  = cfg.get("engine",        "pyttsx3")
        self._open_angle   = cfg.get("open_angle",    60)
        self._closed_angle = cfg.get("closed_angle",  90)
        self._word_ms      = cfg.get("word_open_ms",  80)
        self._rate         = cfg.get("rate",          150)
        self._volume       = cfg.get("volume",        1.0)
        self._voice_index  = cfg.get("voice_index",   0)
        self._edge_voice   = cfg.get("edge_voice",    "en-US-GuyNeural")

    def _init_engine(self):
        if self._engine_type == "edge-tts":
            self._engine = None
            try:
                import pygame
                pygame.mixer.init()
                self._pygame = pygame
            except ImportError:
                print("[tts] pygame not found — falling back to pyttsx3. Run: pip install pygame")
                self._engine_type = "pyttsx3"

        if self._engine_type == "pyttsx3":
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self._rate)
            self._engine.setProperty("volume", self._volume)
            voices = self._engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                self._engine.setProperty("voice", voices[self._voice_index].id)

    def reload_config(self):
        self._load_config()
        if self._engine_type == "pyttsx3" and self._engine:
            self._engine.setProperty("rate",   self._rate)
            self._engine.setProperty("volume", self._volume)

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        if self._engine_type == "edge-tts":
            import asyncio
            asyncio.run(self._speak_edge(text))
        else:
            if self._serial and self._serial.is_connected():
                self._speak_with_mouth(text)
            else:
                self._engine.say(text)
                self._engine.runAndWait()

    def list_voices(self):
        if self._engine_type == "pyttsx3" and self._engine:
            for i, v in enumerate(self._engine.getProperty("voices")):
                print(f"  [{i}] {v.name}  ({v.id})")
        else:
            print("  Voice listing only supported with pyttsx3 engine.")

    # ── pyttsx3 path ──────────────────────────────────────────────────────────

    def _word_angle(self, word_len: int) -> int:
        """Scale mouth open angle by word length (longer = wider open)."""
        t = min(word_len, 8) / 8.0
        span = self._closed_angle - self._open_angle
        return int(self._closed_angle - span * (0.4 + 0.6 * t))

    def _speak_with_mouth(self, text: str):
        word_ms = self._word_ms / 1000.0
        serial  = self._serial

        def on_word(name, location, length):
            word = text[location:location + length]
            angle = self._word_angle(len(word))
            serial.mouth(angle)
            time.sleep(word_ms)
            serial.mouth(self._closed_angle)

        token = self._engine.connect("started-word", on_word)
        self._engine.say(text)
        self._engine.runAndWait()
        self._engine.disconnect("started-word", token)

        serial.mouth(self._closed_angle)

    # ── edge-tts path ─────────────────────────────────────────────────────────

    async def _speak_edge(self, text: str):
        import edge_tts
        import tempfile
        import threading

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        communicate = edge_tts.Communicate(text, self._edge_voice)
        word_events = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(tmp_path, "ab") as f:
                    f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                word_events.append({
                    "offset_ms": chunk["offset"] / 10000,
                    "word":      chunk["text"],
                })

        pygame = self._pygame
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()

        start_ms  = time.time() * 1000
        word_idx  = 0
        word_ms   = self._word_ms / 1000.0

        while pygame.mixer.music.get_busy():
            current_ms = time.time() * 1000 - start_ms
            while word_idx < len(word_events) and word_events[word_idx]["offset_ms"] <= current_ms:
                if self._serial and self._serial.is_connected():
                    angle = self._word_angle(len(word_events[word_idx]["word"]))
                    self._serial.mouth(angle)
                    time.sleep(word_ms)
                    self._serial.mouth(self._closed_angle)
                word_idx += 1
            time.sleep(0.01)

        if self._serial and self._serial.is_connected():
            self._serial.mouth(self._closed_angle)

        pygame.mixer.music.unload()
        os.unlink(tmp_path)

"""
tts_engine.py
-------------
Text-to-speech with mouth animation.

Supports two engines (set tts.engine in settings.json):
  "pyttsx3"  — free, fully offline, uses OS voices
  "edge-tts" — Microsoft neural voices, requires internet
               pip install edge-tts pygame

Mouth angle scales with phonetic content of each word for natural animation.
"""

import os
import sys
import time
import json
import threading
import subprocess
import concurrent.futures

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
        self._serial   = serial_controller
        self._pygame   = None
        self._engine   = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
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
        self._edge_voice   = cfg.get("edge_voice",    "en-US-AndrewNeural")
        self._edge_rate    = cfg.get("edge_rate",     "-5%")
        self._edge_pitch   = cfg.get("edge_pitch",    "-3Hz")

    def _init_engine(self):
        if self._engine_type == "edge-tts":
            try:
                import edge_tts  # noqa: F401
            except ImportError:
                print("[tts] edge-tts not found — falling back to pyttsx3. Run: pip install edge-tts")
                self._engine_type = "pyttsx3"

        if self._engine_type == "edge-tts":
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.pre_init(44100, -16, 2, 512)
                    pygame.mixer.init()
                if not pygame.mixer.get_init():
                    raise RuntimeError("pygame mixer failed to initialize — check audio device")
                self._pygame = pygame
                print(f"[tts] edge-tts ready, pygame mixer: {pygame.mixer.get_init()}")
            except ImportError:
                print("[tts] pygame not found — falling back to pyttsx3. Run: pip install pygame")
                self._engine_type = "pyttsx3"
            except Exception as e:
                print(f"[tts] pygame error: {e} — falling back to pyttsx3")
                self._engine_type = "pyttsx3"

        if self._engine_type == "pyttsx3":
            try:
                import pyttsx3  # noqa: F401
            except ImportError:
                print("[tts] pyttsx3 not found. Run: pip install pyttsx3")

    def reload_config(self):
        old_type = self._engine_type
        self._load_config()
        if self._engine_type != old_type or (self._engine_type == "edge-tts" and self._pygame is None):
            self._pygame = None
            self._init_engine()

    # ── Public API ────────────────────────────────────────────────────────────

    def prefetch(self, text: str) -> "concurrent.futures.Future | None":
        """Start downloading TTS audio in the background. Returns a Future.

        Call this as soon as you have the text. Pass the returned Future to
        speak() so playback begins the moment the download finishes rather than
        waiting until speak() is called.
        """
        if self._engine_type != "edge-tts":
            return None
        return self._executor.submit(self._download_edge, text)

    def speak(self, text: str, future: "concurrent.futures.Future | None" = None):
        """Speak text, optionally using a pre-fetched audio Future."""
        try:
            if self._engine_type == "edge-tts":
                if future is None:
                    future = self.prefetch(text)
                result = future.result()  # blocks only if download isn't done yet
                if result:
                    self._play_edge(*result)
            else:
                if self._serial and self._serial.is_connected():
                    self._speak_with_mouth(text)
                else:
                    self._run_pyttsx3_subprocess(text)
        except Exception as e:
            print(f"[tts] speak error: {e}")

    def list_voices(self):
        if self._engine_type == "pyttsx3":
            code = (
                "import pyttsx3; e = pyttsx3.init(); "
                "[print(f'  [{i}] {v.name}') for i, v in enumerate(e.getProperty('voices'))]"
            )
            subprocess.run([sys.executable, "-c", code], check=False)
        else:
            print("  Voice listing only supported with pyttsx3 engine.")

    # ── Shared helpers ────────────────────────────────────────────────────────

    _VOWEL_OPEN = {'a': 1.0, 'o': 0.9, 'e': 0.65, 'i': 0.45, 'u': 0.40, 'y': 0.30}

    def _word_angle(self, word: str) -> int:
        clean = word.lower().strip(".,!?;:'\"()-")
        if not clean:
            return self._closed_angle
        peak  = max((self._VOWEL_OPEN.get(ch, 0.2) for ch in clean if ch.isalpha()), default=0.2)
        nudge = ((sum(ord(c) for c in clean) % 17) - 8) / 100.0
        t     = max(0.35, min(1.0, peak + nudge))
        return int(self._closed_angle - (self._closed_angle - self._open_angle) * t)

    def _animate_mouth(self, words: list, stop_event: threading.Event):
        word_ms          = self._word_ms / 1000.0
        seconds_per_word = 60.0 / max(self._rate, 1)
        serial           = self._serial
        for word in words:
            if stop_event.is_set():
                break
            serial.mouth(self._word_angle(word))
            time.sleep(word_ms)
            serial.mouth(self._closed_angle)
            gap = seconds_per_word - word_ms
            if gap > 0:
                stop_event.wait(gap)
        serial.mouth(self._closed_angle)

    # ── pyttsx3 path ──────────────────────────────────────────────────────────

    def _run_pyttsx3_subprocess(self, text: str):
        code = (
            f"import pyttsx3; e = pyttsx3.init(); "
            f"e.setProperty('rate', {self._rate}); "
            f"e.setProperty('volume', {self._volume}); "
            f"e.say({repr(text)}); e.runAndWait()"
        )
        subprocess.run([sys.executable, "-c", code], check=False)

    def _speak_with_mouth(self, text: str):
        words      = text.split()
        stop_event = threading.Event()
        anim       = threading.Thread(
            target=self._animate_mouth, args=(words, stop_event), daemon=True
        )
        anim.start()
        self._run_pyttsx3_subprocess(text)
        stop_event.set()
        anim.join(timeout=1.0)
        self._serial.mouth(self._closed_angle)

    # ── edge-tts path ─────────────────────────────────────────────────────────

    def _download_edge(self, text: str):
        """Download edge-tts audio synchronously. Returns (tmp_path, word_events) or None."""
        import asyncio
        import warnings
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._download_edge_async(text))
        except Exception as e:
            print(f"[tts] download error: {e}")
            return None
        finally:
            # Drain pending callbacks so the loop closes without unclosed-socket warnings.
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                loop.close()

    async def _download_edge_async(self, text: str):
        import edge_tts
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        communicate = edge_tts.Communicate(
            text, self._edge_voice,
            rate=self._edge_rate,
            pitch=self._edge_pitch,
        )
        word_events = []
        audio_bytes = 0

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(tmp_path, "ab") as f:
                    f.write(chunk["data"])
                audio_bytes += len(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                word_events.append({
                    "offset_ms": chunk["offset"] / 10000,
                    "word":      chunk["text"],
                })

        if audio_bytes == 0:
            print("[tts] edge-tts returned no audio — check internet and voice name")
            os.unlink(tmp_path)
            return None

        if not word_events:
            duration_ms = (audio_bytes * 8 / 24000) * 1000
            words_list  = text.split()
            if words_list:
                ms_per_word = duration_ms / len(words_list)
                for i, w in enumerate(words_list):
                    word_events.append({"offset_ms": i * ms_per_word, "word": w})

        return (tmp_path, word_events)

    def _play_edge(self, tmp_path: str, word_events: list):
        """Play a pre-downloaded edge-tts audio file with jaw animation."""
        pygame     = self._pygame
        word_ms    = self._word_ms / 1000.0
        serial_lead_ms = 80
        stop_event = threading.Event()

        def animate():
            while not stop_event.is_set() and not pygame.mixer.music.get_busy():
                time.sleep(0.005)
            if stop_event.is_set():
                return
            start_ms = time.time() * 1000
            word_idx = 0
            while not stop_event.is_set():
                current_ms = time.time() * 1000 - start_ms
                while (word_idx < len(word_events) and
                       word_events[word_idx]["offset_ms"] - serial_lead_ms <= current_ms):
                    if self._serial and self._serial.is_connected():
                        self._serial.mouth(self._word_angle(word_events[word_idx]["word"]))
                        time.sleep(word_ms)
                        self._serial.mouth(self._closed_angle)
                    word_idx += 1
                time.sleep(0.005)

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.set_volume(self._volume)

        anim = threading.Thread(target=animate, daemon=True)
        anim.start()
        pygame.mixer.music.play()

        deadline = time.time() + 1.0
        while not pygame.mixer.music.get_busy() and time.time() < deadline:
            time.sleep(0.005)
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)

        stop_event.set()
        anim.join(timeout=1.0)

        if self._serial and self._serial.is_connected():
            self._serial.mouth(self._closed_angle)

        pygame.mixer.music.unload()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

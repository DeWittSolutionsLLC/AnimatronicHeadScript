"""
tts_engine.py
-------------
Text-to-speech with mouth animation.

Supports two engines (set tts.engine in settings.json):
  "pyttsx3"  — free, fully offline, uses OS voices
  "edge-tts" — Microsoft neural voices, requires internet
               pip install edge-tts pygame

Jaw movement uses per-phoneme open angles derived from word boundary
events supplied by edge-tts, giving accurate lip-sync timing.
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
    def __init__(self, serial_controller=None, audio_ready_cb=None):
        self._serial        = serial_controller
        self._pygame        = None
        self._executor      = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._audio_ready_cb = audio_ready_cb
        self._load_config()
        self._init_engine()

    def _load_config(self):
        cfg = _load_tts_config()
        self._engine_type    = cfg.get("engine",          "pyttsx3")
        self._open_angle     = cfg.get("open_angle",      130)
        self._closed_angle   = cfg.get("closed_angle",    90)
        self._word_ms        = cfg.get("word_open_ms",    100)
        self._rate           = cfg.get("rate",            150)
        self._volume         = cfg.get("volume",          1.0)
        self._voice_index    = cfg.get("voice_index",     0)
        self._edge_voice     = cfg.get("edge_voice",      "en-US-EricNeural")
        self._edge_rate      = cfg.get("edge_rate",       "-5%")
        self._edge_pitch     = cfg.get("edge_pitch",      "-35Hz")
        self._serial_lead_ms = cfg.get("serial_lead_ms",  60)

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
        """Start downloading TTS audio in the background. Returns a Future."""
        if self._engine_type != "edge-tts":
            return None
        return self._executor.submit(self._download_edge, text)

    def speak(self, text: str, future: "concurrent.futures.Future | None" = None):
        """Speak text, optionally using a pre-fetched audio Future."""
        try:
            if self._engine_type == "edge-tts":
                if future is None:
                    future = self.prefetch(text)
                result = future.result()
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

    # ── Jaw angle calculation ─────────────────────────────────────────────────

    _PHONEME_OPEN: dict[str, float] = {
        # Vowels
        'a': 1.00, 'â': 1.00,
        'o': 0.85,
        'e': 0.65, 'é': 0.65,
        'i': 0.35,
        'u': 0.30,
        # Approximants / liquids
        'l': 0.45, 'r': 0.45,
        'w': 0.40, 'y': 0.35,
        # Fricatives
        'f': 0.30, 'v': 0.30,
        's': 0.25, 'z': 0.25,
        'h': 0.50,
        # Plosives
        'p': 0.00, 'b': 0.00,
        't': 0.10, 'd': 0.10,
        'k': 0.10, 'g': 0.10,
        # Nasals
        'm': 0.05, 'n': 0.10,
    }
    _DEFAULT_CONSONANT_OPEN = 0.20

    def _char_openness(self, ch: str) -> float:
        return self._PHONEME_OPEN.get(ch.lower(), self._DEFAULT_CONSONANT_OPEN)

    def _word_openness(self, word: str) -> float:
        clean = word.lower().strip(".,!?;:'\"()-—…")
        if not clean:
            return 0.0
        return max(self._char_openness(ch) for ch in clean if ch.isalpha()) if any(
            ch.isalpha() for ch in clean) else 0.0

    def _jaw_angle(self, openness: float) -> int:
        openness = max(0.0, min(1.0, openness))
        return int(self._closed_angle + (self._open_angle - self._closed_angle) * openness)

    _VOWELS = set('aeiou')

    def _phoneme_jaw_sequence(self, word: str, duration_ms: float) -> list[tuple[int, float]]:
        """Per-character jaw positions weighted by phoneme type, filling the word's spoken duration."""
        clean = word.lower().strip(".,!?;:'\"()-—…")
        letters = [c for c in clean if c.isalpha()]
        if not letters:
            return [(self._closed_angle, duration_ms / 1000.0)]
        weights    = [2.5 if ch in self._VOWELS else 1.0 for ch in letters]
        total_w    = sum(weights)
        speak_ms   = duration_ms * 0.82
        close_s    = (duration_ms - speak_ms) / 1000.0
        seq = [
            (self._jaw_angle(self._char_openness(ch)), (speak_ms * w / total_w) / 1000.0)
            for ch, w in zip(letters, weights)
        ]
        seq.append((self._closed_angle, close_s))
        return seq

    def _jaw_sequence(self, word: str) -> list[tuple[int, float]]:
        clean = word.lower().strip(".,!?;:'\"()-—…")
        word_ms = self._word_ms / 1000.0

        first_alpha = next((c for c in clean if c.isalpha()), "")
        is_plosive_start = first_alpha in "pbtdkg"

        openness = self._word_openness(word)
        peak     = self._jaw_angle(openness)
        closed   = self._closed_angle

        if is_plosive_start and openness > 0.1:
            return [
                (closed, word_ms * 0.15),
                (peak,   word_ms * 0.50),
                (closed, word_ms * 0.35),
            ]
        else:
            return [
                (peak,   word_ms * 0.60),
                (closed, word_ms * 0.40),
            ]

    # ── pyttsx3 path ──────────────────────────────────────────────────────────

    def _run_pyttsx3_subprocess(self, text: str):
        code = (
            f"import pyttsx3; e = pyttsx3.init(); "
            f"e.setProperty('rate', {self._rate}); "
            f"e.setProperty('volume', {self._volume}); "
            f"e.say({repr(text)}); e.runAndWait()"
        )
        subprocess.run([sys.executable, "-c", code], check=False)

    def _animate_mouth_words(self, words: list, stop_event: threading.Event):
        serial = self._serial
        ms_per_word = (60.0 / max(self._rate, 1)) * 1000
        for word in words:
            if stop_event.is_set():
                break
            for angle, hold in self._phoneme_jaw_sequence(word, ms_per_word):
                if stop_event.is_set():
                    break
                serial.mouth(angle)
                time.sleep(hold)
        serial.mouth(self._closed_angle)

    def _speak_with_mouth(self, text: str):
        words      = text.split()
        stop_event = threading.Event()
        anim       = threading.Thread(
            target=self._animate_mouth_words, args=(words, stop_event), daemon=True
        )
        anim.start()
        self._run_pyttsx3_subprocess(text)
        stop_event.set()
        anim.join(timeout=1.0)
        self._serial.mouth(self._closed_angle)

    # ── edge-tts path ─────────────────────────────────────────────────────────

    def _download_edge(self, text: str):
        """Download edge-tts audio. Returns (tmp_path, word_events) or None."""
        import asyncio
        import warnings
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._download_edge_async(text))
        except Exception as e:
            print(f"[tts] download error: {e}")
            return None
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                pass
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", ResourceWarning)
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
                    "offset_ms":   chunk["offset"] / 10000,
                    "duration_ms": chunk.get("duration", 0) / 10000,
                    "word":        chunk["text"],
                })

        if audio_bytes == 0:
            print("[tts] edge-tts returned no audio — check internet and voice name")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return None

        if not word_events:
            duration_ms = (audio_bytes * 8 / 24000) * 1000
            words_list  = text.split()
            if words_list:
                ms_per_word = duration_ms / len(words_list)
                for i, w in enumerate(words_list):
                    word_events.append({
                        "offset_ms":   i * ms_per_word,
                        "duration_ms": ms_per_word * 0.8,
                        "word":        w,
                    })

        return (tmp_path, word_events)

    def _estimate_duration_ms(self, tmp_path: str) -> float:
        try:
            return (os.path.getsize(tmp_path) / 3000.0) * 1000 + 3000
        except OSError:
            return 15000

    def _play_edge(self, tmp_path: str, word_events: list):
        """
        Play pre-downloaded audio with accurate jaw animation.
        Uses absolute wall-clock fire times per word so sleep errors
        within one word's phoneme sequence never cascade into the next.
        """
        pygame     = self._pygame
        stop_event = threading.Event()

        # Clip each word's animation duration to the gap before the next word.
        for i in range(len(word_events) - 1):
            gap_ms = word_events[i + 1]["offset_ms"] - word_events[i]["offset_ms"]
            word_events[i]["duration_ms"] = min(
                word_events[i].get("duration_ms", gap_ms), gap_ms * 0.88
            )

        # Main thread sets play_wall[0] just before play(); animate thread
        # spins briefly until the value is written, then drives jaw on wall-clock.
        # No dependency on pygame.mixer.get_busy() so jaw works even when
        # audio output is handled externally (e.g. browser via web server).
        play_wall = [0.0]

        def animate():
            # Wait up to 1 s for play_wall to be stamped by the main thread.
            deadline = time.time() + 1.0
            while play_wall[0] == 0.0 and time.time() < deadline:
                time.sleep(0.002)
            if play_wall[0] == 0.0 or stop_event.is_set():
                return

            t0     = play_wall[0]
            lead_s = self._serial_lead_ms / 1000.0

            for ev in word_events:
                if stop_event.is_set():
                    break

                # Absolute wall-clock moment this word's jaw should start moving.
                fire_at   = t0 + ev["offset_ms"] / 1000.0 - lead_s
                remaining = fire_at - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                if stop_event.is_set():
                    break

                duration_ms = ev.get("duration_ms", 0)
                seq = (self._phoneme_jaw_sequence(ev["word"], duration_ms)
                       if duration_ms > 20 else self._jaw_sequence(ev["word"]))

                for angle, hold in seq:
                    if stop_event.is_set():
                        break
                    if self._serial and self._serial.is_connected():
                        self._serial.mouth(angle)
                    time.sleep(hold)

            if self._serial and self._serial.is_connected():
                self._serial.mouth(self._closed_angle)

        # ── Playback ──────────────────────────────────────────────────────────
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.set_volume(self._volume)

        anim = threading.Thread(target=animate, daemon=True)
        anim.start()

        if self._audio_ready_cb:
            try:
                self._audio_ready_cb(tmp_path)
            except Exception:
                pass

        play_wall[0] = time.time()
        try:
            pygame.mixer.music.play()
        except Exception:
            pass  # no local audio device — browser handles playback

        t0 = time.time()
        while not pygame.mixer.music.get_busy() and time.time() - t0 < 1.0:
            time.sleep(0.01)

        timeout_ms = self._estimate_duration_ms(tmp_path)
        play_start = time.time()
        while pygame.mixer.music.get_busy():
            if (time.time() - play_start) * 1000 > timeout_ms:
                print("[tts] playback timeout — forcing stop")
                pygame.mixer.music.stop()
                break
            time.sleep(0.02)

        stop_event.set()
        anim.join(timeout=1.0)

        if self._serial and self._serial.is_connected():
            self._serial.mouth(self._closed_angle)

        pygame.mixer.music.unload()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
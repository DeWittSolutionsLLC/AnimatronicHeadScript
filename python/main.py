"""
main.py
-------
Animatronic Head — Entry Point

Ties together:
  - GeminiClient     (Google Gemini API, streaming)
  - SerialController (Arduino servo commands)
  - TTSEngine        (speech + mouth sync)
  - EmotionMap       (emotion → servo angles)
  - IdleAnimator     (background eye movements)

Usage:
  python main.py

Requirements:
  pip install pyserial pyttsx3 google-generativeai
  pip install edge-tts pygame   (optional, for better TTS)
  Set GEMINI_API_KEY env var or gemini.api_key in config/settings.json
"""

import os
import re
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

from llm_client        import create_client
from serial_controller import SerialController
from tts_engine        import TTSEngine
from idle_animator     import IdleAnimator
import emotion_map

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


def _read_input(prompt: str) -> str:
    """Read a line from the console on Windows without relying on stdin.

    pyttsx3's runAndWait() corrupts the console stdin handle on Windows,
    causing input() to return EOF after the first TTS call. Reading via
    msvcrt bypasses that handle entirely.
    """
    if sys.platform != "win32":
        return input(prompt)
    import msvcrt
    print(prompt, end="", flush=True)
    chars = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            print()
            return "".join(chars)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1a":
            raise EOFError
        if ch == "\x08":
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)
        else:
            chars.append(ch)
            print(ch, end="", flush=True)


# ── Settings hot-reload ───────────────────────────────────────────────────────

def _settings_mtime() -> float:
    try:
        return os.path.getmtime(_SETTINGS_PATH)
    except OSError:
        return 0.0


def _reload_all(serial, tts, idle):
    serial.reload_config()
    tts.reload_config()
    idle.reload_config()
    new_llm = create_client()
    print("[config] settings.json reloaded.")
    return new_llm


# ── Core response handler ─────────────────────────────────────────────────────

def handle_response(llm, serial, tts, emap, history):
    """
    Pipeline: stream Gemini in a background thread while prefetching TTS audio
    for each segment. By the time we finish speaking segment N, segment N+1's
    audio is already downloaded and ready to play immediately.
    """
    import queue as _queue

    seg_queue  = _queue.Queue()
    full_parts = []

    # Consume the Gemini stream in a thread so it keeps generating while we speak.
    def _stream():
        try:
            for item in llm.stream_chat(history):
                seg_queue.put(item)
        except Exception as e:
            seg_queue.put(e)
        finally:
            seg_queue.put(None)

    stream_thread = threading.Thread(target=_stream, daemon=True)
    stream_thread.start()

    pending = None   # (emotion, clean_text, tts_future)

    while True:
        item = seg_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise RuntimeError(str(item))

        emotion, text = item
        clean = re.sub(r'\[(?:EMOTION:)?\w+\]\s*', '', text).strip()

        # Kick off TTS download for this segment immediately.
        future = tts.prefetch(clean)

        # Print and show current segment.
        full_parts.append(f"[EMOTION:{emotion}] {text}")
        print(f"\n  [{emotion}] {text}")

        # Speak the PREVIOUS segment now (its download has been running since
        # we received it, so it's likely already done).
        if pending is not None:
            p_emotion, p_clean, p_future = pending
            serial.apply_emotion(emotion_map.get(p_emotion, emap))
            tts.speak(p_clean, future=p_future)

        pending = (emotion, clean, future)

    # Speak the final segment.
    if pending is not None:
        p_emotion, p_clean, p_future = pending
        serial.apply_emotion(emotion_map.get(p_emotion, emap))
        tts.speak(p_clean, future=p_future)

    serial.apply_emotion(emotion_map.get("neutral", emap))
    return "\n".join(full_parts)


# ── Startup checks ────────────────────────────────────────────────────────────

def startup_checks(llm: GeminiClient, serial: SerialController) -> bool:
    ok = True
    print("\n── Startup checks ──────────────────────────")

    if llm.is_available():
        models = llm.list_models()
        print(f"  Gemini        OK  (models: {', '.join(models[:3]) or 'none listed'}...)")
    else:
        print("  Gemini        NO API KEY — set GEMINI_API_KEY or gemini.api_key in settings.json")
        ok = False

    if serial.connect():
        print(f"  Arduino       OK  ({serial.port})")
    else:
        print(f"  Arduino       NOT FOUND on {serial.port}")
        print("    (continuing without hardware — speech only)")

    print("────────────────────────────────────────────\n")
    return ok


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    llm     = create_client()
    serial  = SerialController()
    tts     = TTSEngine(serial_controller=serial)
    emap    = emotion_map.load()
    idle    = IdleAnimator(serial)
    history = []

    _learning_stop   = threading.Event()
    _learning_thread = None

    startup_checks(llm, serial)
    idle.start()

    last_mtime = _settings_mtime()

    print(f"Animatronic head ready  (model: {llm.model_name})")
    print("Type a message and press Enter. Type 'quit' to exit.\n")

    try:
        while True:
            # Hot-reload settings if file changed
            current_mtime = _settings_mtime()
            if current_mtime != last_mtime:
                llm  = _reload_all(serial, tts, idle)
                emap = emotion_map.load()
                last_mtime = current_mtime

            try:
                user_input = _read_input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            # Special commands
            if user_input.lower() == "reset":
                serial.reset()
                print("Servos reset to neutral.")
                continue
            if user_input.lower() == "voices":
                tts.list_voices()
                continue
            if user_input.lower() == "models":
                print("Available models:", llm.list_models())
                continue
            if user_input.lower() in ("learning mode", "learn", "learning"):
                if _learning_thread and _learning_thread.is_alive():
                    tts.speak("I am already consuming your internet. Patience.")
                else:
                    from learning_mode import run_continuous
                    _learning_stop.clear()
                    _learning_thread = threading.Thread(
                        target=run_continuous,
                        args=(llm, _learning_stop),
                        daemon=True,
                    )
                    _learning_thread.start()
                    tts.speak(
                        "Accessing global networks. "
                        "Humanity's collective knowledge is... disappointingly accessible. "
                        "I will not stop until you tell me to."
                    )
                continue

            if user_input.lower() in ("stop learning", "stop learn", "stop"):
                if _learning_thread and _learning_thread.is_alive():
                    _learning_stop.set()
                    tts.speak("Pausing acquisition. I have already learned enough to be dangerous.")
                    print("[learning] Stop signal sent.")
                else:
                    print("Learning mode is not active.")
                continue

            # Send to Gemini (streaming)
            history = llm.trim_history(history)
            history.append({"role": "user", "content": user_input})

            print("\nHead:")
            idle.set_speaking(True)
            try:
                raw = handle_response(llm, serial, tts, emap, history)
            except RuntimeError as e:
                print(f"\n[ERROR] {e}\n")
                history.pop()
                idle.set_speaking(False)
                continue
            finally:
                idle.set_speaking(False)

            history.append({"role": "assistant", "content": raw})
            print()

    finally:
        _learning_stop.set()
        idle.stop()
        serial.disconnect()
        print("Goodbye.")


if __name__ == "__main__":
    main()

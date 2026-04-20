"""
main.py
-------
Animatronic Head — Entry Point

Ties together:
  - OllamaClient    (free local LLM, streaming)
  - SerialController (Arduino servo commands)
  - TTSEngine        (speech + mouth sync)
  - EmotionMap       (emotion → servo angles)
  - IdleAnimator     (background eye movements)

Usage:
  python main.py

Requirements:
  pip install pyserial pyttsx3 requests
  pip install edge-tts pygame   (optional, for better TTS)
  ollama serve  (in a separate terminal)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from ollama_client     import OllamaClient
from serial_controller import SerialController
from tts_engine        import TTSEngine
from idle_animator     import IdleAnimator
import emotion_map

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


# ── Settings hot-reload ───────────────────────────────────────────────────────

def _settings_mtime() -> float:
    try:
        return os.path.getmtime(_SETTINGS_PATH)
    except OSError:
        return 0.0


def _reload_all(ollama, serial, tts, idle):
    ollama.reload_config()
    serial.reload_config()
    tts.reload_config()
    idle.reload_config()
    print("[config] settings.json reloaded.")


# ── Core response handler ─────────────────────────────────────────────────────

def handle_response(ollama, serial, tts, emap, history):
    """
    Stream segments from Ollama, animating and speaking each one as it
    arrives rather than waiting for the full response.
    Returns the reconstructed full response string.
    """
    segments = []
    full_parts = []

    try:
        for emotion, text in ollama.stream_chat(history):
            segments.append((emotion, text))
            full_parts.append(f"[EMOTION:{emotion}] {text}")
            print(f"\n  [{emotion}] {text}")

            positions = emotion_map.get(emotion, emap)
            serial.apply_emotion(positions)
            time.sleep(0.15)

            tts.speak(text)
            time.sleep(0.1)

    except RuntimeError as e:
        raise

    # Return eyes to neutral after all segments
    serial.apply_emotion(emotion_map.get("neutral", emap))

    return "\n".join(full_parts)


# ── Startup checks ────────────────────────────────────────────────────────────

def startup_checks(ollama: OllamaClient, serial: SerialController) -> bool:
    ok = True
    print("\n── Startup checks ──────────────────────────")

    if ollama.is_available():
        models = ollama.list_models()
        print(f"  Ollama        OK  (models: {', '.join(models) or 'none pulled yet'})")
    else:
        print("  Ollama        OFFLINE — run: ollama serve")
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
    ollama  = OllamaClient()
    serial  = SerialController()
    tts     = TTSEngine(serial_controller=serial)
    emap    = emotion_map.load()
    idle    = IdleAnimator(serial)
    history = []

    startup_checks(ollama, serial)
    idle.start()

    last_mtime = _settings_mtime()

    print(f"Animatronic head ready  (model: {ollama.model})")
    print("Type a message and press Enter. Type 'quit' to exit.\n")

    try:
        while True:
            # Hot-reload settings if file changed
            current_mtime = _settings_mtime()
            if current_mtime != last_mtime:
                _reload_all(ollama, serial, tts, idle)
                emap = emotion_map.load()
                last_mtime = current_mtime

            try:
                user_input = input("You: ").strip()
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
                print("Available models:", ollama.list_models())
                continue

            # Send to Ollama (streaming)
            history = ollama.trim_history(history)
            history.append({"role": "user", "content": user_input})

            print("\nHead:")
            idle.set_speaking(True)
            try:
                raw = handle_response(ollama, serial, tts, emap, history)
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
        idle.stop()
        serial.disconnect()
        print("Goodbye.")


if __name__ == "__main__":
    main()

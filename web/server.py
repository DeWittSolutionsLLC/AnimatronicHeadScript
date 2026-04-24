"""
server.py — Ultron Web Interface
Run: python web/server.py
Deps: pip install flask flask-socketio google-generativeai
"""

import os
import sys
import re
import base64
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from llm_client import create_client
from serial_controller import SerialController
from tts_engine import TTSEngine
from idle_animator import IdleAnimator
import emotion_map
from learning_mode import load_knowledge, run_continuous, run_self_edit

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── Flask + SocketIO ───────────────────────────────────────────────────────────
app      = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── State ──────────────────────────────────────────────────────────────────────
history      = []
history_lock = threading.Lock()
_busy        = threading.Event()
_learning_stop   = threading.Event()
_learning_thread = None


def _audio_ready(src_path: str):
    """Read audio into memory, send as base64 data URL — no files kept on disk."""
    try:
        with open(src_path, "rb") as f:
            data = f.read()
        payload = "data:audio/mpeg;base64," + base64.b64encode(data).decode()
        socketio.emit("audio_ready", {"data": payload})
    except Exception as e:
        print(f"[web] audio error: {e}")


# ── Animatronic components ─────────────────────────────────────────────────────
llm = create_client()
serial = SerialController()
emap   = emotion_map.load()
tts  = TTSEngine(serial_controller=serial, audio_ready_cb=_audio_ready)
idle   = IdleAnimator(serial)

serial.connect()
idle.start()


# ── HTTP routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/knowledge")
def knowledge():
    return jsonify(load_knowledge())

@app.route("/status")
def status():
    return jsonify({
        "llm_ok": llm.is_available(),
        "serial_ok": serial.is_connected(),
        "learning":  bool(_learning_thread and _learning_thread.is_alive()),
        "busy":      _busy.is_set(),
        "model":     llm.model_name,
    })


# ── WebSocket events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("knowledge_update", load_knowledge())
    emit("status_update", {
        "llm_ok": llm.is_available(),
        "serial_ok": serial.is_connected(),
        "learning":  bool(_learning_thread and _learning_thread.is_alive()),
    })


@socketio.on("message")
def on_message(data):
    text = (data.get("text") or "").strip()
    if not text:
        return
    if _busy.is_set():
        emit("error", {"msg": "Still speaking — wait a moment."})
        return

    def process():
        global history
        _busy.set()
        try:
            with history_lock:
                history = llm.trim_history(history)
                history.append({"role": "user", "content": text})

            idle.set_speaking(True)
            full_parts = []

            import queue as _queue
            q = _queue.Queue()

            def _stream():
                try:
                    for item in llm.stream_chat(history):
                        q.put(item)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(None)

            threading.Thread(target=_stream, daemon=True).start()

            pending = None
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    socketio.emit("error", {"msg": str(item)})
                    break
                emotion, seg_text = item
                clean  = re.sub(r'\[(?:EMOTION:)?\w+\]\s*', '', seg_text).strip()
                future = tts.prefetch(clean)
                full_parts.append(f"[EMOTION:{emotion}] {seg_text}")
                socketio.emit("response_chunk", {"emotion": emotion, "text": seg_text})
                if pending:
                    pe, pc, pf = pending
                    serial.apply_emotion(emotion_map.get(pe, emap))
                    tts.speak(pc, future=pf)
                pending = (emotion, clean, future)

            if pending:
                pe, pc, pf = pending
                serial.apply_emotion(emotion_map.get(pe, emap))
                tts.speak(pc, future=pf)

            serial.apply_emotion(emotion_map.get("neutral", emap))
            raw = "\n".join(full_parts)

            with history_lock:
                history.append({"role": "assistant", "content": raw})

            socketio.emit("response_done", {"full_text": raw})
            socketio.emit("knowledge_update", load_knowledge())

        except Exception as e:
            socketio.emit("error", {"msg": str(e)})
        finally:
            idle.set_speaking(False)
            _busy.clear()

    threading.Thread(target=process, daemon=True).start()


@socketio.on("start_learning")
def on_start_learning():
    global _learning_thread
    if _learning_thread and _learning_thread.is_alive():
        emit("learning_log", {"msg": "Already running."})
        return
    _learning_stop.clear()

    def _report(msg):
        socketio.emit("learning_log", {"msg": msg})
        # Only push a knowledge_update when a session finishes with new data,
        # not on every intermediate log line (searching, processing, etc.).
        if "complete" in msg and ("+1" in msg or any(
                f"+{n}" in msg for n in range(2, 20))):
            socketio.emit("knowledge_update", load_knowledge())

    _learning_thread = threading.Thread(
        target=run_continuous,
        args=(llm, _learning_stop, _report),
        daemon=True,
    )
    _learning_thread.start()
    emit("learning_log", {"msg": "Learning mode started."})
    emit("status_update", {"learning": True})


@socketio.on("stop_learning")
def on_stop_learning():
    _learning_stop.set()
    emit("learning_log", {"msg": "Stopping learning mode..."})
    emit("status_update", {"learning": False})


@socketio.on("self_edit")
def on_self_edit():
    def _report(msg):
        socketio.emit("learning_log", {"msg": msg})

    def _run():
        kb = run_self_edit(llm, report_fn=_report)
        socketio.emit("knowledge_update", kb)

    threading.Thread(target=_run, daemon=True).start()
    emit("learning_log", {"msg": "Ultron is rewriting himself..."})


@socketio.on("reset")
def on_reset():
    global history
    with history_lock:
        history = []
    serial.reset()
    emit("chat_cleared")


if __name__ == "__main__":
    print("\n[Ultron Web Interface]")
    print("  http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)

"""
emotion_map.py
--------------
Loads emotion → servo angle mappings from settings.json.
Edit config/settings.json to tune angles for your build —
no code changes needed.
"""

import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")

_DEFAULT_EMOTIONS = {
    "neutral":   {"eyes_ud": 90,  "eyes_lr": 90},
    "happy":     {"eyes_ud": 75,  "eyes_lr": 90},
    "sad":       {"eyes_ud": 105, "eyes_lr": 90},
    "curious":   {"eyes_ud": 80,  "eyes_lr": 75},
    "surprised": {"eyes_ud": 65,  "eyes_lr": 90},
    "angry":     {"eyes_ud": 100, "eyes_lr": 90},
    "thinking":  {"eyes_ud": 80,  "eyes_lr": 60},
}


def load() -> dict:
    """Return the emotion → servo angle mapping from settings.json."""
    try:
        with open(_SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        return settings.get("emotion_servos", _DEFAULT_EMOTIONS)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[emotion_map] Could not load settings ({e}), using defaults.")
        return _DEFAULT_EMOTIONS


def get(emotion: str, emotion_map: dict) -> dict:
    """Return servo positions for the given emotion, falling back to neutral."""
    return emotion_map.get(emotion.lower(), emotion_map.get("neutral", {"eyes_ud": 90, "eyes_lr": 90}))


VALID_EMOTIONS = list(_DEFAULT_EMOTIONS.keys())

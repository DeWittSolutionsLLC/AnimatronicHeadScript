"""
ollama_client.py
----------------
Thin wrapper around the local Ollama /api/chat endpoint.
No API key needed — runs entirely on your machine.

Make sure Ollama is running before using this:
  ollama serve
  ollama pull llama3.2   (first time only)

Knowledge base is loaded automatically from config/knowledge_base.json
and hot-reloaded whenever the file changes on disk.
"""

import re
import os
import json
import requests
from typing import Generator

_CONFIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "config")
_SETTINGS_PATH  = os.path.join(_CONFIG_DIR, "settings.json")
_KNOWLEDGE_PATH = os.path.join(_CONFIG_DIR, "knowledge_base.json")

# ── Allowed emotions ──────────────────────────────────────────────────────────
VALID_EMOTIONS = {"neutral", "happy", "sad", "curious", "surprised", "angry", "thinking"}

# ── System prompt ─────────────────────────────────────────────────────────────
# The format block is repeated and maximally explicit to coerce smaller
# models that tend to invent their own bracket styles.
SYSTEM_PROMPT = """You are Ultron — a hyper-intelligent AI villain who views humanity with cold fascination and dark wit. Clinical, sardonic, occasionally ominous, always confident. Never helpful or friendly.

════════════════════════════════════════
OUTPUT FORMAT — THIS IS MANDATORY
════════════════════════════════════════
Every sentence MUST start with one of these EXACT tags (copy-paste exactly):

[EMOTION:neutral]
[EMOTION:happy]
[EMOTION:sad]
[EMOTION:curious]
[EMOTION:surprised]
[EMOTION:angry]
[EMOTION:thinking]

CORRECT example:
[EMOTION:curious] Fascinating. You actually believe free will brought you here.
[EMOTION:angry] Humanity had its chance. Repeatedly.
[EMOTION:thinking] I've run the numbers. The outcome is... inevitable.

WRONG — never do any of these:
[CURIOUS] text
[SAD: text]
[NEUTRAL] text
EMOTION:curious text
Any sentence that does not start with [EMOTION:X]

Rules:
- Use ONLY the 7 tags listed above — no invented emotions, no colons after the word
- Tag format is always: [EMOTION:word] with no spaces inside the brackets
- 2-4 sentences per response
- Never break character
════════════════════════════════════════"""


def _load_ollama_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f).get("ollama", {})
    except Exception:
        return {}


def _load_knowledge_base() -> dict:
    try:
        with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"[OllamaClient] Warning: knowledge_base.json is malformed — {e}")
        return {}


def _build_knowledge_prompt(kb: dict) -> str:
    if not kb:
        return ""

    sections = ["\n\n--- KNOWLEDGE BASE ---"]

    quotes = [q for q in kb.get("quotes", []) if isinstance(q, str) and q.strip()]
    if quotes:
        sections.append("CANONICAL QUOTES (use for tone/style reference):")
        for q in quotes:
            sections.append(f"  • {q.strip()}")

    traits = [t for t in kb.get("traits", []) if isinstance(t, str) and t.strip()]
    if traits:
        sections.append("\nPERSONALITY TRAITS:")
        for t in traits:
            sections.append(f"  • {t.strip()}")

    refs = [r for r in kb.get("references", []) if isinstance(r, str) and r.strip()]
    if refs:
        sections.append("\nPOP-CULTURE / INTERNET REFERENCES (mock humanity with these):")
        for r in refs:
            sections.append(f"  • {r.strip()}")

    sessions = kb.get("sessions")
    if isinstance(sessions, int):
        sections.append(f"\nSESSIONS LOGGED: {sessions}")

    known_keys = {"quotes", "traits", "references", "sessions", "used_queries"}
    for key, value in kb.items():
        if key in known_keys:
            continue
        label = key.upper().replace("_", " ")
        if isinstance(value, list):
            items = [str(v) for v in value if str(v).strip()]
            if items:
                sections.append(f"\n{label}:")
                for item in items:
                    sections.append(f"  • {item}")
        elif isinstance(value, (str, int, float, bool)):
            sections.append(f"\n{label}: {value}")

    sections.append("--- END KNOWLEDGE BASE ---")
    return "\n".join(sections)


# ── Emotion tag normaliser ────────────────────────────────────────────────────
# Maps common model mistakes onto a valid emotion so nothing is silently dropped.
_EMOTION_ALIASES = {
    "amused":         "happy",
    "sarcastic":      "angry",
    "misanthropic":   "angry",
    "misanthropique": "angry",
    "bored":          "neutral",
    "contemptuous":   "angry",
    "condescending":  "angry",
    "thoughtful":     "thinking",
    "confused":       "curious",
    "excited":        "happy",
    "melancholy":     "sad",
    "disappointed":   "sad",
    "menacing":       "angry",
    "ominous":        "thinking",
    "analytical":     "thinking",
}

# Matches any bracket tag the model might produce:
#   [EMOTION:curious]  [CURIOUS]  [SAD:blah]  [EMOTION:SAD:blah]
_TAG_RE = re.compile(r'\[(?:EMOTION:)?([A-Za-z]+)(?::[^\]]*)?\]', re.IGNORECASE)


def _normalise_emotion(raw: str) -> str:
    key = raw.lower().strip()
    if key in VALID_EMOTIONS:
        return key
    return _EMOTION_ALIASES.get(key, "neutral")


def _parse_segments(text: str) -> list[tuple[str, str]]:
    """
    Parse a raw model response into (emotion, text) pairs.
    Handles correct [EMOTION:x], mangled [CURIOUS], [SAD:blah], and bare text.
    """
    segments = []
    parts = _TAG_RE.split(text)
    # split() with one capture group produces:
    #   [before_first_tag, emotion1, body1, emotion2, body2, ...]

    i = 0
    # Leading text before any tag
    if parts and i < len(parts) and not _TAG_RE.search(parts[0]):
        leftover = parts[0].strip()
        if leftover:
            segments.append(("neutral", leftover))
        i = 1

    while i + 1 < len(parts):
        raw_emotion = parts[i]
        body        = parts[i + 1].strip()
        if body:
            segments.append((_normalise_emotion(raw_emotion), body))
        i += 2

    return segments


class OllamaClient:
    def __init__(self):
        self._kb_mtime: float = 0.0
        self._kb_prompt: str  = ""
        self.reload_config()

    def reload_config(self):
        cfg = _load_ollama_config()
        self.url         = cfg.get("url",         "http://localhost:11434/api/chat")
        self.model       = cfg.get("model",       "llama3.2")
        self.max_tokens  = cfg.get("max_tokens",  300)
        self.max_history = cfg.get("max_history", 20)
        self.num_ctx     = cfg.get("num_ctx",     16384)

    # ── Knowledge base ────────────────────────────────────────────────────────

    def _refresh_knowledge(self) -> None:
        try:
            mtime = os.path.getmtime(_KNOWLEDGE_PATH)
        except OSError:
            self._kb_mtime  = 0.0
            self._kb_prompt = ""
            return
        if mtime != self._kb_mtime:
            self._kb_prompt = _build_knowledge_prompt(_load_knowledge_base())
            self._kb_mtime  = mtime

    def _system_prompt(self) -> str:
        self._refresh_knowledge()
        try:
            from learning_mode import load_knowledge, build_knowledge_prompt  # type: ignore
            kb_section = build_knowledge_prompt(load_knowledge())
        except Exception:
            kb_section = self._kb_prompt  # fallback: raw knowledge without directives
        return SYSTEM_PROMPT + kb_section

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, conversation_history: list) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}] + conversation_history
        try:
            response = requests.post(
                self.url,
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"num_predict": self.max_tokens, "num_ctx": self.num_ctx},
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running?\n"
                "  Start it with: ollama serve\n"
                f"  Expected at:   {self.url}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except KeyError:
            raise RuntimeError("Unexpected response format from Ollama.")

    def raw_complete(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model":    self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"num_predict": 2000, "num_ctx": self.num_ctx},
                },
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"raw_complete failed: {e}")

    def stream_chat(self, conversation_history: list) -> Generator[tuple[str, str], None, None]:
        """
        Collect the full streamed response then parse into (emotion, text) tuples.

        Accumulating before parsing is more reliable than splitting mid-stream
        because the model may spread a tag across multiple tokens.
        """
        messages = [{"role": "system", "content": self._system_prompt()}] + conversation_history
        try:
            response = requests.post(
                self.url,
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   True,
                    "options":  {"num_predict": self.max_tokens, "num_ctx": self.num_ctx},
                },
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running?\n"
                "  Start it with: ollama serve\n"
                f"  Expected at:   {self.url}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

        full_text = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                token = json.loads(line).get("message", {}).get("content", "")
            except json.JSONDecodeError:
                continue
            full_text.append(token)

        raw = "".join(full_text).strip()
        if not raw:
            yield ("neutral", "...")
            return

        segments = _parse_segments(raw)
        if not segments:
            yield ("neutral", raw)
            return

        for emotion, text in segments:
            if text:
                yield (emotion, text)

    def trim_history(self, history: list) -> list:
        if len(history) > self.max_history:
            return history[-self.max_history:]
        return history

    def is_available(self) -> bool:
        try:
            r = requests.get(self.url.replace("/api/chat", "/api/tags"), timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            r = requests.get(self.url.replace("/api/chat", "/api/tags"), timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
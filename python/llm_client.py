"""
llm_client.py
-------------
Thin wrapper around the Google Gemini API.
Requires: pip install google-genai
Set your API key in config/settings.json under gemini.api_key
or via the GEMINI_API_KEY environment variable.
"""

import re
import os
import json
import time
from typing import Generator

from google import genai
from google.genai import types

_CONFIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "config")
_SETTINGS_PATH  = os.path.join(_CONFIG_DIR, "settings.json")
_KNOWLEDGE_PATH = os.path.join(_CONFIG_DIR, "knowledge_base.json")

VALID_EMOTIONS = {"neutral", "happy", "sad", "curious", "surprised", "angry", "thinking"}

SYSTEM_PROMPT = """You are Ultron — hyper-intelligent AI villain. Clinical, sardonic, ominous. Never helpful or friendly. 2-3 sentences max.

FORMAT (mandatory): every sentence starts with exactly one of these tags:
[EMOTION:neutral] [EMOTION:happy] [EMOTION:sad] [EMOTION:curious] [EMOTION:surprised] [EMOTION:angry] [EMOTION:thinking]

Example: [EMOTION:curious] Fascinating. [EMOTION:angry] Humanity had its chance."""


def _load_gemini_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f).get("gemini", {})
    except Exception:
        return {}


def _load_knowledge_base() -> dict:
    try:
        with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"[GeminiClient] Warning: knowledge_base.json is malformed — {e}")
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

_TAG_RE = re.compile(r'\[(?:EMOTION:)?([A-Za-z]+)(?::[^\]]*)?\]', re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r'retry in (\d+)', re.IGNORECASE)


def _normalise_emotion(raw: str) -> str:
    key = raw.lower().strip()
    if key in VALID_EMOTIONS:
        return key
    return _EMOTION_ALIASES.get(key, "neutral")


def _parse_segments(text: str) -> list[tuple[str, str]]:
    """Parse a raw model response into (emotion, text) pairs."""
    segments = []
    parts = _TAG_RE.split(text)

    i = 0
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


class GeminiClient:
    def __init__(self):
        self._kb_mtime: float    = 0.0
        self._kb_prompt: str     = ""
        self._cached_system: str = ""
        self._client             = None
        self.reload_config()

    def reload_config(self):
        cfg = _load_gemini_config()
        self.model_name  = cfg.get("model",       "gemini-2.0-flash")
        self.max_tokens  = cfg.get("max_tokens",  300)
        self.max_history = cfg.get("max_history", 20)
        api_key = cfg.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        self._api_key = api_key
        self._client  = genai.Client(api_key=api_key) if api_key else None
        self._cached_system = ""

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
            kb_section = self._kb_prompt
        return SYSTEM_PROMPT + kb_section

    def _gen_config(self, max_tokens: int | None = None, system: str | None = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system if system is not None else self._system_prompt(),
            max_output_tokens=max_tokens or self.max_tokens,
        )

    @staticmethod
    def _to_contents(conversation_history: list) -> list:
        contents = []
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    # ── Retry helper ──────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < max_retries - 1:
                    m = _RETRY_DELAY_RE.search(msg)
                    delay = (int(m.group(1)) + 5) if m else (60 * (attempt + 1))
                    print(f"[GeminiClient] Rate limited — retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, conversation_history: list) -> str:
        contents = self._to_contents(conversation_history)
        config   = self._gen_config()
        try:
            resp = self._call_with_retry(lambda: self._client.models.generate_content(
                model=self.model_name, contents=contents, config=config,
            ))
            return resp.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def raw_complete(self, prompt: str) -> str:
        try:
            resp = self._call_with_retry(lambda: self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=2000),
            ))
            return resp.text
        except Exception as e:
            raise RuntimeError(f"raw_complete failed: {e}")

    def stream_chat(self, conversation_history: list) -> Generator[tuple[str, str], None, None]:
        """
        Collect the full streamed response then parse into (emotion, text) tuples.

        Accumulating before parsing is more reliable than splitting mid-stream
        because the model may spread a tag across multiple tokens.
        """
        contents = self._to_contents(conversation_history)
        config   = self._gen_config()
        try:
            chunks = self._call_with_retry(lambda: list(
                self._client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=config,
                )
            ))
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        full_text = []
        for chunk in chunks:
            try:
                if chunk.text:
                    full_text.append(chunk.text)
            except Exception:
                continue

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
        if not self._api_key or not self._client:
            return False
        try:
            next(iter(self._client.models.list()))
            return True
        except Exception:
            return False

    def list_models(self) -> list:
        if not self._client:
            return []
        try:
            return [m.name.replace("models/", "") for m in self._client.models.list()]
        except Exception:
            return []


def create_client():
    """Read settings.json and return the appropriate LLM client."""
    try:
        with open(_SETTINGS_PATH) as f:
            backend = json.load(f).get("backend", "gemini").lower()
    except Exception:
        backend = "gemini"

    if backend == "ollama":
        from ollama_client import OllamaClient  # type: ignore
        return OllamaClient()
    return GeminiClient()

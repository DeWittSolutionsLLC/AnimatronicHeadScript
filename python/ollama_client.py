"""
ollama_client.py
----------------
Thin wrapper around the local Ollama /api/chat endpoint.
No API key needed — runs entirely on your machine.

Make sure Ollama is running before using this:
  ollama serve
  ollama pull llama3.2   (first time only)
"""

import requests
import json
import re
import os
from typing import Generator

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")

SYSTEM_PROMPT = """You are an expressive animatronic character head.
When you speak, format EVERY response exactly like this:

[EMOTION:happy] Hello, it's great to meet you!
[EMOTION:curious] I wonder what we'll talk about today.

Rules:
- Use ONLY these emotions: neutral, happy, sad, curious, surprised, angry, thinking
- Put [EMOTION:X] at the start of each sentence or natural phrase
- Keep responses conversational and expressive, 2-5 sentences total
- Never add text outside the [EMOTION:X] format
- Never include explanations, preambles, or markdown"""


def _load_ollama_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f).get("ollama", {})
    except Exception:
        return {}


class OllamaClient:
    def __init__(self):
        self.reload_config()

    def reload_config(self):
        cfg = _load_ollama_config()
        self.url         = cfg.get("url",         "http://localhost:11434/api/chat")
        self.model       = cfg.get("model",       "llama3.2")
        self.max_tokens  = cfg.get("max_tokens",  300)
        self.max_history = cfg.get("max_history", 20)

    def chat(self, conversation_history: list) -> str:
        """Send conversation history and return the full assistant reply."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        try:
            response = requests.post(
                self.url,
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"num_predict": self.max_tokens},
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

    def stream_chat(self, conversation_history: list) -> Generator[tuple[str, str], None, None]:
        """
        Stream the response, yielding (emotion, text) tuples as each
        [EMOTION:X] segment completes. Starts animating before the full
        response is received.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        try:
            response = requests.post(
                self.url,
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   True,
                    "options":  {"num_predict": self.max_tokens},
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

        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                token = json.loads(line).get("message", {}).get("content", "")
            except json.JSONDecodeError:
                continue
            buffer += token

            # Yield any complete segments (a segment ends when the next [EMOTION: starts)
            while True:
                parts = re.split(r'(?=\[EMOTION:)', buffer, maxsplit=2)
                if len(parts) < 3:
                    break
                # parts[1] is a complete segment, parts[2] is the start of the next
                match = re.match(r'\[EMOTION:(\w+)\]\s*(.+)', parts[1].strip(), re.DOTALL)
                if match:
                    yield (match.group(1).lower(), match.group(2).strip())
                buffer = parts[2]

        # Yield any remaining segment after the stream ends
        if buffer.strip():
            match = re.match(r'\[EMOTION:(\w+)\]\s*(.+)', buffer.strip(), re.DOTALL)
            if match:
                yield (match.group(1).lower(), match.group(2).strip())
            elif not buffer.strip().startswith("[EMOTION:"):
                yield ("neutral", buffer.strip())

    def trim_history(self, history: list) -> list:
        """Keep history within max_history messages (preserves pairs)."""
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

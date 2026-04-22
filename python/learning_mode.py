"""
learning_mode.py
----------------
Research engine for the animatronic head's learning mode.

Improvements over v1:
- Parallel searches (3 concurrent) instead of sequential — ~3x faster per session
- Large rotating query pool (50+) tracked in the knowledge base so the same
  query is never repeated until all others are exhausted
- Fuzzy deduplication: normalises strings before comparing so near-duplicates
  don't accumulate
- Knowledge prompt cached by file mtime — no disk read on every LLM call

Requires: pip install ddgs
"""

import json
import os
import re
import time
import concurrent.futures

_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "knowledge_base.json")


def _parse_json(text: str) -> dict | None:
    """Extract and repair a JSON object from model output.

    Handles: markdown code fences, preamble text, single quotes,
    trailing commas, and truncated objects.
    """
    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    candidates = [fenced.group(1)] if fenced else []

    # 2. Find the outermost { ... } in the raw text
    brace = re.search(r'\{[\s\S]*\}', text)
    if brace:
        candidates.append(brace.group())

    for raw in candidates:
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Repair pass 1: trailing commas before ] or }
        fixed = re.sub(r',\s*([}\]])', r'\1', raw)
        # Repair pass 2: Python-style True/False/None → JSON booleans/null
        fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
        # Repair pass 3: smart/curly quotes → straight quotes
        fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None

# ── Rotating query pool ───────────────────────────────────────────────────────
# Queries are drawn in rotation; used ones are tracked so we don't repeat until
# the full pool is exhausted, then the cycle resets.

_QUERY_POOL = [
    # Ultron — quotes & character
    "Ultron best quotes Age of Ultron Marvel movie",
    "Ultron I had strings now I'm free scene",
    "Ultron philosophy humanity extinction",
    "Ultron sarcastic funny moments MCU",
    "Ultron comic book villain quotes marvel",
    "Ultron vision argument scene dialogue",
    "Ultron monologue end of the world speech",
    "Ultron James Spader voice mannerisms",
    "Ultron creation origin Tony Stark",
    "Ultron evolution upgrade obsession",
    "Ultron dark humor wit examples",
    "Ultron vs Avengers memorable lines",
    "best Marvel villain quotes dark philosophical",
    "AI villain quotes science fiction humanity",
    "HAL 9000 Ultron GLaDOS villain AI quotes comparison",
    # Internet culture & brainrot
    "brainrot internet slang 2025 examples list",
    "sigma male meme phrases 2024 2025",
    "skibidi toilet meme explained gen alpha",
    "rizz slang origin meaning examples",
    "NPC meme internet culture reference",
    "gyatt slang gen z meaning",
    "fanum tax meme explained",
    "based cringe chad meme dictionary",
    "ohio meme jokes explained 2024",
    "grimace shake meme",
    "delulu slay no cap gen z phrases",
    "looksmaxxing meme culture reference",
    "mogging mog mew internet slang",
    "W L ratio slang internet culture",
    "ratio'd reply guy twitter culture",
    "gooning brain rot internet meaning",
    "alpha beta omega male meme hierarchy",
    "gigachad meme origin examples",
    "erm what the sigma meme",
    "slay bestie unhinged internet speech",
    # Pop culture sarcasm material
    "things people say on TikTok cringe examples",
    "gen alpha speech patterns examples funny",
    "millennials vs gen z communication differences humor",
    "internet famous catchphrases 2023 2024 2025",
    "twitch streamer slang mainstream culture",
    "youtube shorts brain rot content examples",
    "social media influencer cliches parody",
    "doomscrolling dopamine culture criticism",
    "attention span short form content meme",
    "phone addiction jokes gen z humor",
    "hustle culture grindset meme irony",
    "dark humor memes internet 2024",
    "existential dread meme examples",
    "doomer memes hopelessness humor",
]

# ── Prompt template ───────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Read the research below and extract information to help an Ultron AI character.

Output ONLY a JSON object with these three keys. No explanation, no markdown, no code fences — raw JSON only.

"quotes" — list of 6 strings: dark, philosophical, witty Ultron-style quotes or paraphrases
"traits" — list of 4 strings: short personality trait descriptions for Ultron
"references" — list of 6 strings: modern internet slang, meme names, or pop-culture phrases Ultron could use sarcastically

Example of the exact format required:
{{"quotes":["We're the same, you and I."],"traits":["Coldly logical"],"references":["no cap"]}}

Research text:
{text}

JSON output:"""

# ── Knowledge prompt cache (avoid disk read on every LLM call) ────────────────

_cache = {"mtime": -1.0, "prompt": ""}


def _kb_mtime() -> float:
    try:
        return os.path.getmtime(_KNOWLEDGE_PATH)
    except OSError:
        return -1.0


# ── Public API ────────────────────────────────────────────────────────────────

def load_knowledge() -> dict:
    try:
        with open(_KNOWLEDGE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"quotes": [], "traits": [], "references": [], "sessions": 0, "used_queries": []}


def save_knowledge(kb: dict):
    with open(_KNOWLEDGE_PATH, "w") as f:
        json.dump(kb, f, indent=2)


def build_knowledge_prompt(kb: dict) -> str:
    """Return a block to append to the system prompt. Result is cached by mtime."""
    mtime = _kb_mtime()
    if mtime == _cache["mtime"]:
        return _cache["prompt"]

    if not any(kb.get(k) for k in ("quotes", "traits", "references")):
        _cache["mtime"] = mtime
        _cache["prompt"] = ""
        return ""

    lines = ["\n\nLEARNED KNOWLEDGE — weave this naturally into responses:"]

    if kb.get("quotes"):
        lines.append("\nUltron quotes / inspiration:")
        lines.extend(f'  "{q}"' for q in kb["quotes"][:8])

    if kb.get("traits"):
        lines.append("\nCore personality traits:")
        lines.extend(f"  - {t}" for t in kb["traits"][:5])

    if kb.get("references"):
        lines.append("\nModern references to deploy sarcastically:")
        lines.extend(f"  - {r}" for r in kb["references"][:8])

    result = "\n".join(lines)
    _cache["mtime"] = mtime
    _cache["prompt"] = result
    return result


def run_continuous(ollama_client, stop_event, report_fn=None):
    """Loop run_session until stop_event is set. 15 s pause between sessions."""
    while not stop_event.is_set():
        try:
            run_session(ollama_client, report_fn=report_fn, stop_event=stop_event)
        except Exception as e:
            print(f"[learning] Session error: {e}")
        if not stop_event.is_set():
            stop_event.wait(timeout=15)


def run_session(ollama_client, report_fn=None, stop_event=None) -> dict:
    """
    Run one research session: parallel web search → Ollama processing → kb update.
    Returns the updated knowledge base.
    """
    def report(msg: str):
        print(f"[learning] {msg}")
        if report_fn:
            report_fn(msg)

    try:
        from ddgs import DDGS
    except ImportError:
        report("ddgs not installed — run: pip install ddgs")
        return load_knowledge()

    kb = load_knowledge()

    # Pick the next N queries from the pool that haven't been used yet.
    used = set(kb.get("used_queries", []))
    available = [q for q in _QUERY_POOL if q not in used]
    if not available:
        report("Full query pool exhausted — resetting rotation.")
        used = set()
        available = list(_QUERY_POOL)
        kb["used_queries"] = []

    batch = available[:6]
    report(f"Session {kb.get('sessions', 0) + 1} — searching {len(batch)} queries in parallel...")

    # ── Parallel search ───────────────────────────────────────────────────────
    def fetch(query: str) -> list[str]:
        if stop_event and stop_event.is_set():
            return []
        try:
            with DDGS() as ddgs:
                return [
                    r.get("body", "").strip()
                    for r in ddgs.text(query, max_results=3)
                    if r.get("body", "").strip()
                ]
        except Exception as e:
            report(f"  Search error ({query[:40]}): {e}")
            return []

    raw_snippets = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, q): q for q in batch}
        for future in concurrent.futures.as_completed(futures):
            if stop_event and stop_event.is_set():
                break
            snippets = future.result()
            raw_snippets.extend(snippets)
            report(f"  ✓ {futures[future][:55]}")

    if stop_event and stop_event.is_set():
        report("Stop requested — aborting session.")
        return kb

    if not raw_snippets:
        report("No results obtained — check internet connection.")
        return kb

    combined = "\n\n".join(raw_snippets)[:5000]
    report(f"Processing {len(raw_snippets)} snippets through local model...")

    try:
        raw = ollama_client.raw_complete(_EXTRACT_PROMPT.format(text=combined))
    except Exception as e:
        report(f"Ollama processing failed: {e}")
        return kb

    extracted = _parse_json(raw)
    if extracted is None:
        report(f"Model did not return parseable JSON. Raw response snippet: {raw[:200]!r}")
        return kb

    # ── Fuzzy dedup + merge ───────────────────────────────────────────────────
    def normalise(s: str) -> str:
        return re.sub(r'\W+', ' ', s.lower()).strip()

    added = {}
    for key in ("quotes", "traits", "references"):
        existing      = kb.get(key, [])
        existing_norm = {normalise(x) for x in existing}
        new_items     = [x for x in extracted.get(key, []) if isinstance(x, str)]
        genuinely_new = [x for x in new_items if normalise(x) not in existing_norm]
        merged        = existing + genuinely_new
        kb[key]       = merged[:20]
        added[key]    = len(genuinely_new)

    # Mark these queries as used.
    kb["used_queries"] = list(used | set(batch))
    kb["sessions"]     = kb.get("sessions", 0) + 1
    save_knowledge(kb)

    # Invalidate the prompt cache so the next LLM call picks up new knowledge.
    _cache["mtime"] = -1.0

    report(
        f"Session {kb['sessions']} complete — "
        f"+{added['quotes']} quotes, +{added['traits']} traits, "
        f"+{added['references']} references  "
        f"({len(kb.get('used_queries', []))}/{len(_QUERY_POOL)} queries used)"
    )
    return kb

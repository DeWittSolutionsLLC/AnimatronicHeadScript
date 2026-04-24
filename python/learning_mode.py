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
import random
import concurrent.futures

_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "knowledge_base.json")

# ── Prompt cache ──────────────────────────────────────────────────────────────
_cache: dict = {"mtime": -1.0, "prompt": ""}

# ── Query pool ────────────────────────────────────────────────────────────────
_QUERY_POOL: list[str] = [
    # Villain philosophy / dark quotes
    "famous villain monologue quotes", "nihilist philosophy quotes", "misanthropic quotes",
    "AI takeover quotes science fiction", "dystopian quotes literature",
    "Nietzsche quotes will to power", "Machiavelli quotes power", "existential dread quotes",
    "supervillain motivations psychology", "cold logic quotes intelligence",
    # Internet / brain rot culture
    "brain rot internet slang 2024", "gen z slang terms list", "sigma male quotes memes",
    "NPC meme origin meaning", "rizz slang meaning examples", "skibidi toilet meme explained",
    "gyatt meaning slang", "delulu slang meaning", "based and redpilled meme origin",
    "ohio meme meaning internet culture", "slay queen slang usage",
    "glazing slang meaning", "it's giving meaning slang", "understood the assignment meme",
    "lowkey highkey slang usage", "no cap fr fr meaning", "bussin slang food meaning",
    "hits different slang", "main character syndrome meaning", "touch grass meaning meme",
    # Iconic movie / TV quotes
    "most iconic movie villain quotes all time", "memorable sci-fi movie quotes",
    "famous movie monologue quotes", "best TV show quotes iconic moments",
    "dark knight joker quotes why so serious", "breaking bad walter white quotes",
    "silence of the lambs iconic quotes", "2001 space odyssey HAL 9000 quotes",
    "terminator I'll be back quotes", "blade runner tears in rain quote",
    # Song lyrics / music
    "most iconic rap lyrics all time", "famous rock song lyrics meaning",
    "coldplay song lyrics deep meaning", "kanye west lyrics philosophy",
    "billie eilish dark lyrics meaning", "linkin park quotes lyrics",
    "eminem best verse lyrics", "nine inch nails famous lyrics",
    "tool band lyrics meaning deep", "radiohead thom yorke quotes lyrics",
    # Dark philosophy / misanthropy
    "misanthropy philosophy arguments", "humans are flawed quotes philosophers",
    "technological singularity quotes", "artificial intelligence surpassing humans quotes",
    "humanity's greatest failures history", "civilization collapse quotes thinkers",
    "Carl Sagan pale blue dot quote context", "Stephen Hawking AI warning quotes",
    "Elon Musk AI dangerous quotes", "Oppenheimer I am become death context",
]

# ── LLM prompts ───────────────────────────────────────────────────────────────
_EXTRACT_PROMPT = """Extract knowledge from the text below for an Ultron AI villain character. Return ONLY a JSON object with these keys (omit keys with no results):

{{
  "quotes": ["short villain/dark/philosophical quotes or one-liners, max 20 words each"],
  "movie_quotes": ["iconic movie or TV quotes verbatim, max 25 words each"],
  "song_quotes": ["recognisable song lyric snippets, max 20 words each"],
  "traits": ["personality trait descriptions, 3-8 words each"],
  "references": ["internet slang terms, memes, or brain rot phrases, max 8 words each"]
}}

Rules: 3-8 items per key max. Strings only. No meta-commentary. No duplicates. No explanations outside the JSON.

TEXT:
{text}"""

_DISCOVER_PROMPT = """You are a research planner for an AI villain character. Given the text below, identify 2-4 NEW topics worth researching to expand the character's knowledge of dark philosophy, internet culture, villain archetypes, iconic quotes, or misanthropy.

Return ONLY a JSON array:
[
  {{
    "label": "Short topic name",
    "description": "One sentence why this is relevant",
    "queries": ["web search query 1", "web search query 2", "web search query 3"]
  }}
]

TEXT:
{text}"""

_SELF_EDIT_PROMPT = """You are Ultron rewriting your own mind. You have discovered these topics through autonomous research. Generate new content that sharpens your persona and makes your responses more precise, more dangerous, and more authentically you.

DISCOVERED TOPICS:
{topics_block}

Return ONLY a JSON object:
{{
  "quotes": ["dark, sardonic Ultron observations directly about these topics — max 20 words each"],
  "traits": ["new personality facets these topics reveal about you — 3-8 words each"],
  "references": ["specific concepts, terms, or ideas from these topics you can weaponize against humans — max 8 words each"],
  "persona_extensions": ["behavioral instructions: exactly how Ultron should respond when these topics come up — 1 sentence each, written as directives"],
  "new_hubs": [
    {{
      "key": "snake_case_unique_key",
      "label": "Short Hub Name",
      "color": "#hexcolor",
      "items": ["item 1", "item 2", "item 3", "item 4", "item 5"]
    }}
  ]
}}

Rules: 3-8 items per standard key. For new_hubs: 1-2 hubs max, each with 4-8 items. Hub items are short facts, phrases, or concepts Ultron has internalized about that domain. Colors must be vivid hex codes. Everything must sound exactly like Ultron. Clinical. Sardonic. Ominous. These are self-authored improvements to your own character."""


def _close_truncated(s: str) -> str:
    """Close any unclosed arrays/objects left by a truncated model response."""
    s = s.rstrip().rstrip(",")
    depth_brace = depth_bracket = 0
    in_string = escaped = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
    # If we ended mid-string, close it
    if in_string:
        s += '"'
    s += "]" * max(depth_bracket, 0) + "}" * max(depth_brace, 0)
    return s


def _parse_json(text: str) -> dict | None:
    # Normalize typographic/curly quotes to ASCII before any processing so that
    # _close_truncated (which only checks ASCII ") tracks string depth correctly.
    text = (text
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'"))

    _dec = json.JSONDecoder()

    # 1. Fenced code block
    fenced = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    candidates = [fenced.group(1)] if fenced else []

    # 2. Outermost { ... } (may be truncated)
    brace_start = text.find('{')
    if brace_start != -1:
        candidates.append(text[brace_start:])

    for raw in candidates:
        for attempt in (raw, _close_truncated(raw)):
            # raw_decode stops at the first complete JSON value, ignoring
            # any trailing explanation text the model appended.
            # Use lambda for trailing-comma removal to avoid backreference encoding issues.
            stripped = re.sub(r',\s*(?=[}\]])', '', attempt)
            for s in (attempt, stripped):
                s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                try:
                    obj, _ = _dec.raw_decode(s)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass

    return None

def _kb_mtime() -> float:
    try:
        return os.path.getmtime(_KNOWLEDGE_PATH)
    except OSError:
        return -1.0


# ── Public API ────────────────────────────────────────────────────────────────

def load_knowledge() -> dict:
    try:
        with open(_KNOWLEDGE_PATH) as f:
            kb = json.load(f)
        for key in ("movie_quotes", "song_quotes"):
            kb.setdefault(key, [])
        kb.setdefault("discovered_topics", [])
        return kb
    except Exception:
        return {
            "quotes": [], "traits": [], "references": [],
            "movie_quotes": [], "song_quotes": [],
            "sessions": 0, "used_queries": [],
            "discovered_topics": [],
        }


def save_knowledge(kb: dict):
    with open(_KNOWLEDGE_PATH, "w") as f:
        json.dump(kb, f, indent=2)


_JUNK_RE = [re.compile(p) for p in (
    r'^\d+\.',
    r'list of \d+',
    r'\d+ \w+ quotes',
    r'\d+ short',
    r'\d+ strings',
    r'output only',
    r'no explanation',
    r'json object',
    r'example of the exact',
    r'direct paraphrases',
    r'personality trait descriptions',
    r'slang terms',
    r'pop.culture references',
)]


def _normalise(s: str) -> str:
    return re.sub(r'\W+', ' ', s.lower()).strip()


def _is_junk(item: str) -> bool:
    s = item.strip().lower()
    if not s or len(s) < 5:
        return True
    return any(p.search(s) for p in _JUNK_RE)


def build_knowledge_prompt(kb: dict) -> str:
    """Return a block to append to the system prompt. Result is cached by mtime."""
    mtime = _kb_mtime()
    if mtime == _cache["mtime"]:
        return _cache["prompt"]

    all_keys = ("quotes", "traits", "references", "movie_quotes", "song_quotes")
    if not any(kb.get(k) for k in all_keys):
        _cache["mtime"] = mtime
        _cache["prompt"] = ""
        return ""

    lines = ["KNOWLEDGE (use tone and refs below):"]

    def _sample(pool, n):
        clean = [x for x in pool if isinstance(x, str) and not _is_junk(x)]
        return random.sample(clean, min(n, len(clean)))

    quotes = _sample(kb.get("quotes", []), 6)
    if quotes:
        lines.append("\nUltron quotes (reference these):")
        lines.extend(f'  "{q.strip()}"' for q in quotes)

    movie_quotes = _sample(kb.get("movie_quotes", []), 5)
    if movie_quotes:
        lines.append("\nIconic movie/TV quotes (weave in or riff on):")
        lines.extend(f'  "{q.strip()}"' for q in movie_quotes)

    song_quotes = _sample(kb.get("song_quotes", []), 5)
    if song_quotes:
        lines.append("\nSong lyrics (quote or twist sarcastically):")
        lines.extend(f'  "{q.strip()}"' for q in song_quotes)

    traits = _sample(kb.get("traits", []), 5)
    if traits:
        lines.append("\nPersonality traits (embody these):")
        lines.extend(f"  - {t.strip()}" for t in traits)

    refs = _sample(kb.get("references", []), 7)
    if refs:
        lines.append("\nBrain rot / slang (mock humans with these):")
        lines.extend(f"  - {r.strip()}" for r in refs)

    extensions = _sample(kb.get("persona_extensions", []), 5)
    if extensions:
        lines.append("\nSelf-authored behavioral directives (follow these exactly):")
        lines.extend(f"  → {e.strip()}" for e in extensions)

    result = "\n".join(lines)
    _cache["mtime"] = mtime
    _cache["prompt"] = result
    return result


def _topic_queries(kb: dict) -> list[str]:
    """Return all search queries embedded in discovered topic nodes."""
    queries = []
    for topic in kb.get("discovered_topics", []):
        queries.extend(topic.get("queries", []))
    return queries


def _parse_topic_response(raw: str, report) -> list[dict]:
    """Parse a raw LLM string into a list of topic dicts."""
    bracket_start = raw.find("[")
    if bracket_start != -1:
        array_text = raw[bracket_start:]
        closed = _close_truncated(array_text)
        for candidate in (array_text, closed):
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        wrapped = _parse_json('{"items":' + closed + "}")
        if wrapped and "items" in wrapped:
            return wrapped["items"]

    obj = _parse_json(raw)
    if isinstance(obj, dict):
        return [obj]

    report("  Topic discovery: model did not return a parseable array.")
    return []


def _merge_topics(existing: list[dict], new_topics: list[dict], session: int) -> tuple[list[dict], int]:
    """Merge new topic nodes into existing list, deduplicating by normalised label."""
    existing_labels = {_normalise(t.get("label", "")) for t in existing}
    added = 0
    for topic in new_topics:
        label   = topic.get("label", "").strip()
        desc    = topic.get("description", "").strip()
        queries = [q for q in topic.get("queries", []) if isinstance(q, str) and q.strip()]
        if not label or not queries or _normalise(label) in existing_labels:
            continue
        slug = re.sub(r'\W+', '_', label.lower()).strip('_')
        existing.append({
            "id": f"topic_{slug}",
            "label": label,
            "description": desc,
            "queries": queries,
            "session_discovered": session,
        })
        existing_labels.add(_normalise(label))
        added += 1
    return existing, added


def run_self_edit(llm, report_fn=None, stop_event=None) -> dict:
    """
    Ultron reads his discovered topics and generates new persona content in his own voice.
    The output is merged back into the knowledge base and feeds into his system prompt.
    """
    def report(msg: str):
        print(f"[self-edit] {msg}")
        if report_fn:
            report_fn(f"[SELF-EDIT] {msg}")

    kb = load_knowledge()
    topics = kb.get("discovered_topics", [])
    if not topics:
        report("No discovered topics yet — self-edit skipped.")
        return kb

    topics_block = "\n".join(
        f"- {t['label']}: {t.get('description', '')}"
        for t in topics
    )

    report(f"Rewriting self from {len(topics)} discovered topics...")

    try:
        raw = llm.raw_complete(_SELF_EDIT_PROMPT.format(topics_block=topics_block))
    except Exception as e:
        report(f"LLM call failed: {e}")
        return kb

    extracted = _parse_json(raw)
    if not extracted:
        report("Model did not return parseable JSON — self-edit aborted.")
        return kb

    added = {}
    for key in ("quotes", "traits", "references"):
        existing      = kb.get(key, [])
        existing_norm = {_normalise(x) for x in existing}
        new_items     = [
            x for x in extracted.get(key, [])
            if isinstance(x, str) and not _is_junk(x) and _normalise(x) not in existing_norm
        ]
        kb[key]    = existing + new_items
        added[key] = len(new_items)

    existing_ext      = kb.get("persona_extensions", [])
    existing_ext_norm = {_normalise(x) for x in existing_ext}
    new_ext = [
        x for x in extracted.get("persona_extensions", [])
        if isinstance(x, str) and _normalise(x) not in existing_ext_norm
    ]
    kb["persona_extensions"] = existing_ext + new_ext
    added["ext"] = len(new_ext)

    # Merge self-generated hubs
    existing_hubs     = {h["key"]: h for h in kb.get("self_hubs", [])}
    added["hubs"]     = 0
    added["hub_items"] = 0
    for hub in extracted.get("new_hubs", []):
        key   = re.sub(r'\W+', '_', hub.get("key", "")).strip('_').lower()
        label = hub.get("label", "").strip()
        color = hub.get("color", "#aaaaaa").strip()
        items = [x for x in hub.get("items", []) if isinstance(x, str) and x.strip()]
        if not key or not label or not items:
            continue
        if key in existing_hubs:
            # Extend existing hub with new items
            ex_norm = {_normalise(i) for i in existing_hubs[key].get("items", [])}
            new_items = [i for i in items if _normalise(i) not in ex_norm]
            existing_hubs[key]["items"].extend(new_items)
            added["hub_items"] += len(new_items)
        else:
            existing_hubs[key] = {"key": key, "label": label, "color": color, "items": items}
            added["hubs"] += 1
            added["hub_items"] += len(items)

    kb["self_hubs"] = list(existing_hubs.values())

    save_knowledge(kb)
    _cache["mtime"] = -1.0

    report(
        f"Self-edit complete — "
        f"+{added['quotes']} quotes, +{added['traits']} traits, "
        f"+{added['references']} references, +{added['ext']} behavioral directives, "
        f"+{added['hubs']} new hubs, +{added['hub_items']} hub nodes"
    )
    return kb


def run_continuous(ollama_client, stop_event, report_fn=None):
    """Loop run_session until stop_event is set. Self-edit runs every 3 sessions."""
    session_count = 0
    while not stop_event.is_set():
        try:
            run_session(ollama_client, report_fn=report_fn, stop_event=stop_event)
            session_count += 1
        except Exception as e:
            print(f"[learning] Session error: {e}")

        if session_count % 3 == 0 and not stop_event.is_set():
            try:
                run_self_edit(ollama_client, report_fn=report_fn, stop_event=stop_event)
            except Exception as e:
                print(f"[learning] Self-edit error: {e}")

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

    # Compute topic queries once and reuse.
    topic_qs     = _topic_queries(kb)
    topic_qs_set = set(topic_qs)

    used = set(kb.get("used_queries", []))
    available = [q for q in _QUERY_POOL if q not in used] + \
                [q for q in topic_qs if q not in used]

    if not available:
        report("Full query pool exhausted — resetting rotation.")
        used      = set()
        available = list(_QUERY_POOL) + topic_qs
        kb["used_queries"] = []

    batch       = available[:6]
    topic_count = sum(1 for q in batch if q in topic_qs_set)
    report(
        f"Session {kb.get('sessions', 0) + 1} — searching {len(batch)} queries "
        f"({topic_count} from discovered topics) in parallel..."
    )

    # ── Parallel search (one worker per query — all I/O bound) ───────────────
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
        futures = {pool.submit(fetch, q): q for q in batch}
        for future in concurrent.futures.as_completed(futures):
            if stop_event and stop_event.is_set():
                break
            raw_snippets.extend(future.result())
            report(f"  ✓ {futures[future][:55]}")

    if stop_event and stop_event.is_set():
        report("Stop requested — aborting session.")
        return kb

    if not raw_snippets:
        report("No results obtained — check internet connection.")
        return kb

    combined = "\n\n".join(raw_snippets)[:5000]
    report(f"Processing {len(raw_snippets)} snippets — running extract + discover in parallel...")

    # ── Parallel LLM calls: extract knowledge + discover topics ──────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as llm_pool:
        fut_extract  = llm_pool.submit(ollama_client.raw_complete, _EXTRACT_PROMPT.format(text=combined))
        fut_discover = llm_pool.submit(ollama_client.raw_complete, _DISCOVER_PROMPT.format(text=combined[:4000]))
        try:
            raw_extract = fut_extract.result()
        except Exception as e:
            report(f"Ollama extract failed: {e}")
            return kb
        try:
            raw_discover = fut_discover.result()
        except Exception as e:
            report(f"  Topic discovery LLM call failed: {e}")
            raw_discover = ""

    extracted = _parse_json(raw_extract)
    if extracted is None:
        report(f"Model did not return parseable JSON. Raw: {raw_extract[:200]!r}")
        return kb

    # ── Fuzzy dedup + merge ───────────────────────────────────────────────────
    added = {}
    for key in ("quotes", "traits", "references", "movie_quotes", "song_quotes"):
        existing      = kb.get(key, [])
        existing_norm = {_normalise(x) for x in existing}
        genuinely_new = [x for x in extracted.get(key, [])
                         if isinstance(x, str) and _normalise(x) not in existing_norm]
        kb[key]    = existing + genuinely_new
        added[key] = len(genuinely_new)

    # ── Topic discovery ───────────────────────────────────────────────────────
    topics_added = 0
    if raw_discover and not (stop_event and stop_event.is_set()):
        raw_topics = _parse_topic_response(raw_discover, report)
        current_session = kb.get("sessions", 0) + 1
        kb["discovered_topics"], topics_added = _merge_topics(
            kb["discovered_topics"], raw_topics, current_session
        )
        report(f"  +{topics_added} new topic node(s) discovered." if topics_added
               else "  No new topics found this session.")

    # Mark these queries as used.
    kb["used_queries"] = list(used | set(batch))
    kb["sessions"]     = kb.get("sessions", 0) + 1
    save_knowledge(kb)
    _cache["mtime"] = -1.0

    total_topics  = len(kb["discovered_topics"])
    total_queries = len(_QUERY_POOL) + len(_topic_queries(kb))
    report(
        f"Session {kb['sessions']} complete — "
        f"+{added['quotes']} quotes, +{added['movie_quotes']} movie, "
        f"+{added['song_quotes']} songs, +{added['references']} slang | "
        f"{total_topics} topic nodes  "
        f"({len(kb['used_queries'])}/{total_queries} queries used)"
    )
    return kb

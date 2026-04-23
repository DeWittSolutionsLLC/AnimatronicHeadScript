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

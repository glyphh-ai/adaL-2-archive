"""
Semantic primes — the decomposition ontology at the center of the rotor.

Every idea is taken down to the primes it relates to (the NSM universal
primitives: KNOW, WANT, FEEL, DO, HAPPEN, WHERE, WHEN, GOOD, BAD, …).
Primes are stored on turns and events at write time; at recall time the
prompt's own primes select and group what comes back — that grouping is
the fact tree.

Decomposition is heuristic-first: a deterministic keyword map that is
always available and costs nothing. When a local model is up it sharpens
the heuristic set (validated against the prime list — the model can only
choose primes, never invent them).
"""

from __future__ import annotations

import re

# The NSM semantic primes, grouped as in the literature. The rotor uses
# them as a closed vocabulary — every decomposition is a subset of this.
PRIMES: dict[str, list[str]] = {
    "substantives": ["I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY"],
    "relational": ["KIND", "PART"],
    "determiners": ["THIS", "SAME", "OTHER"],
    "quantifiers": ["ONE", "TWO", "SOME", "ALL", "MUCH", "LITTLE"],
    "evaluators": ["GOOD", "BAD"],
    "descriptors": ["BIG", "SMALL"],
    "mental": ["THINK", "KNOW", "WANT", "DONT_WANT", "FEEL", "SEE", "HEAR"],
    "speech": ["SAY", "WORDS", "TRUE"],
    "actions": ["DO", "HAPPEN", "MOVE"],
    "existence": ["BE_SOMEWHERE", "THERE_IS", "BE_SOMEONE"],
    "life": ["LIVE", "DIE"],
    "time": ["WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME",
             "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT"],
    "space": ["WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR",
              "SIDE", "INSIDE", "TOUCH"],
    "logic": ["NOT", "MAYBE", "CAN", "BECAUSE", "IF"],
    "intensifier": ["VERY", "MORE"],
    "similarity": ["LIKE"],
}

ALL_PRIMES: frozenset[str] = frozenset(p for group in PRIMES.values() for p in group)

# Deterministic keyword → prime map. Deliberately conservative: a hit
# means the idea plausibly RELATES to the prime, nothing stronger.
_KEYWORDS: dict[str, str] = {
    # mental predicates
    "know": "KNOW", "knew": "KNOW", "remember": "KNOW", "recall": "KNOW",
    "forget": "KNOW", "learn": "KNOW", "understand": "KNOW",
    "think": "THINK", "thought": "THINK", "believe": "THINK",
    "idea": "THINK", "plan": "THINK", "decide": "THINK", "consider": "THINK",
    "want": "WANT", "need": "WANT", "wish": "WANT", "hope": "WANT",
    "goal": "WANT", "prefer": "WANT", "intend": "WANT",
    "feel": "FEEL", "felt": "FEEL", "love": "FEEL", "hate": "FEEL",
    "afraid": "FEEL", "happy": "FEEL", "sad": "FEEL", "angry": "FEEL",
    "worry": "FEEL", "excited": "FEEL",
    "see": "SEE", "saw": "SEE", "look": "SEE", "watch": "SEE", "show": "SEE",
    "hear": "HEAR", "heard": "HEAR", "listen": "HEAR", "sound": "HEAR",
    # speech
    "say": "SAY", "said": "SAY", "tell": "SAY", "told": "SAY", "ask": "SAY",
    "asked": "SAY", "talk": "SAY", "call": "SAY", "name": "SAY",
    "word": "WORDS", "true": "TRUE", "false": "TRUE", "correct": "TRUE",
    "wrong": "TRUE", "fact": "TRUE",
    # actions & events
    "do": "DO", "did": "DO", "make": "DO", "made": "DO", "build": "DO",
    "built": "DO", "create": "DO", "work": "DO", "run": "DO", "use": "DO",
    "happen": "HAPPEN", "happened": "HAPPEN", "occur": "HAPPEN",
    "event": "HAPPEN", "became": "HAPPEN",
    "move": "MOVE", "go": "MOVE", "went": "MOVE", "come": "MOVE",
    "leave": "MOVE", "arrive": "MOVE", "travel": "MOVE", "drive": "MOVE",
    # people & things
    "i": "I", "me": "I", "my": "I", "mine": "I", "myself": "I",
    "you": "YOU", "your": "YOU",
    "people": "PEOPLE", "person": "SOMEONE", "who": "SOMEONE",
    "someone": "SOMEONE", "somebody": "SOMEONE",
    "wife": "SOMEONE", "husband": "SOMEONE", "friend": "SOMEONE",
    "family": "PEOPLE", "team": "PEOPLE", "kid": "SOMEONE",
    "child": "SOMEONE", "children": "PEOPLE", "son": "SOMEONE",
    "daughter": "SOMEONE", "mother": "SOMEONE", "father": "SOMEONE",
    "something": "SOMETHING", "thing": "SOMETHING", "what": "SOMETHING",
    "body": "BODY", "hand": "BODY", "head": "BODY", "health": "BODY",
    # evaluation & description
    "good": "GOOD", "great": "GOOD", "best": "GOOD", "right": "GOOD",
    "better": "GOOD", "favorite": "GOOD",
    "bad": "BAD", "worst": "BAD", "worse": "BAD", "problem": "BAD",
    "broken": "BAD", "fail": "BAD", "bug": "BAD", "error": "BAD",
    "big": "BIG", "large": "BIG", "huge": "BIG",
    "small": "SMALL", "little": "LITTLE", "tiny": "SMALL",
    # time
    "when": "WHEN", "time": "WHEN", "date": "WHEN", "year": "WHEN",
    "month": "WHEN", "week": "WHEN", "day": "WHEN", "schedule": "WHEN",
    "now": "NOW", "today": "NOW", "currently": "NOW",
    "before": "BEFORE", "ago": "BEFORE", "yesterday": "BEFORE",
    "was": "BEFORE", "were": "BEFORE", "past": "BEFORE", "history": "BEFORE",
    "after": "AFTER", "tomorrow": "AFTER", "next": "AFTER", "will": "AFTER",
    "future": "AFTER", "soon": "AFTER",
    "always": "A_LONG_TIME", "forever": "A_LONG_TIME", "never": "NOT",
    # space
    "where": "WHERE", "place": "WHERE", "location": "WHERE",
    "address": "WHERE", "city": "WHERE", "home": "WHERE", "house": "WHERE",
    "here": "HERE", "near": "NEAR", "close": "NEAR", "far": "FAR",
    "inside": "INSIDE", "in": "INSIDE",
    # existence & life
    "is": "THERE_IS", "are": "THERE_IS", "exist": "THERE_IS",
    "have": "THERE_IS", "has": "THERE_IS", "own": "THERE_IS",
    "live": "LIVE", "life": "LIVE", "born": "LIVE",
    "die": "DIE", "died": "DIE", "death": "DIE", "dead": "DIE",
    # quantity
    "one": "ONE", "two": "TWO", "some": "SOME", "few": "SOME",
    "all": "ALL", "every": "ALL", "everything": "ALL",
    "many": "MUCH", "much": "MUCH", "most": "MUCH", "lot": "MUCH",
    "count": "MUCH", "number": "MUCH", "how many": "MUCH",
    # logic & modality
    "not": "NOT", "no": "NOT", "none": "NOT", "without": "NOT",
    "maybe": "MAYBE", "might": "MAYBE", "perhaps": "MAYBE",
    "could": "MAYBE", "possibly": "MAYBE",
    "can": "CAN", "able": "CAN", "cannot": "NOT",
    "because": "BECAUSE", "why": "BECAUSE", "reason": "BECAUSE",
    "since": "BECAUSE", "cause": "BECAUSE",
    "if": "IF", "unless": "IF", "whether": "IF",
    "very": "VERY", "really": "VERY", "extremely": "VERY",
    "more": "MORE", "less": "MORE", "most of": "MORE",
    "like": "LIKE", "similar": "LIKE", "same": "SAME", "as": "LIKE",
    "different": "OTHER", "other": "OTHER", "another": "OTHER",
    "this": "THIS", "kind": "KIND", "type": "KIND", "part": "PART",
}

_WORD_RE = re.compile(r"[a-z']+")


def _lookup(word: str) -> str | None:
    """Exact match, then light suffix-stripping (wants→want, moved→move,
    thinking→think). Deterministic — no stemmer dependency."""
    p = _KEYWORDS.get(word)
    if p is not None:
        return p
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)]
            p = _KEYWORDS.get(stem) or _KEYWORDS.get(stem + "e")
            if p is not None:
                return p
    return None


def decompose(text: str) -> list[str]:
    """Deterministic decomposition: text → the primes it relates to.
    Order = first appearance in the text (stable, reproducible)."""
    seen: dict[str, None] = {}
    for w in _WORD_RE.findall(text.lower()):
        p = _lookup(w)
        if p is not None and p not in seen:
            seen[p] = None
    # numbers relate to quantity
    if re.search(r"\d", text) and "MUCH" not in seen:
        seen["MUCH"] = None
    return list(seen)


def validate(primes: list[str]) -> list[str]:
    """Keep only real primes (a model may only choose, never invent)."""
    out: dict[str, None] = {}
    for p in primes:
        p = p.strip().upper().replace(" ", "_").replace("-", "_")
        if p in ALL_PRIMES and p not in out:
            out[p] = None
    return list(out)


async def decompose_sharp(text: str, complete) -> list[str]:
    """Heuristics + optional local-model sharpening. `complete` is an
    async (prompt) -> str|None; None (model down) → heuristics stand."""
    base = decompose(text)
    raw = await complete(
        "You map text to semantic primes. From this closed list only:\n"
        + ", ".join(sorted(ALL_PRIMES))
        + "\n\nText:\n" + text[:1200]
        + "\n\nReply with ONLY a comma-separated list of the 3-8 most "
          "relevant primes from the list. No other words."
    )
    if not raw:
        return base
    sharp = validate(raw.replace("\n", ",").split(","))
    if not sharp:
        return base
    # union, model's ordering first — heuristics never get dropped
    return validate(sharp + base)

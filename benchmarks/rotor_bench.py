"""
Rotor benchmark — proving the trade.

The rotor buys answer quality with input tokens. This measures both sides
so the trade is a number, not a hope:

  RETRIEVAL  For each probe, does the rotor surface the fact the answer
             needs? (recall) — and does it stay quiet when no memory is
             relevant? (no-noise on distractors). Fully deterministic:
             the heuristic enricher, lattice recall only, no model, no
             network. This is the rotor's OWN contribution — the part we
             build. If recall is high and noise is low, injection helps;
             if not, it is dead weight.

  COST       The size of what we inject, in tokens (~3.5 chars/token).
             Paid once per conversation (then cached), and ZERO on a
             distractor because an empty fact tree injects nothing.

  END-TO-END (opt-in, needs a local model at ADA_LOCAL_URL) The same
             probes answered by the model WITH vs WITHOUT the injection,
             scored on whether the answer contains the ground truth. This
             is the headline: baseline structurally cannot know your
             facts; the rotor lets the model answer them. Skipped cleanly
             when no model is up.

Run:  python -m benchmarks.rotor_bench          # retrieval + cost
      ADA_E2E=1 python -m benchmarks.rotor_bench # + end-to-end A/B

The retrieval + cost half is also asserted in tests/test_rotor_bench.py,
so "we recall the right thing, cheaply" is a regression-protected claim.
"""

from __future__ import annotations

import asyncio
import os

# ── The scenario: one person's life + work memory ──────────────────────
MEMORIES = [
    "chris's wife is brandi",
    "brandi has two kids james and traceton",
    "chris's favorite programming language is python",
    "chris lives in austin texas",
    "chris works at a company called glyphh",
    "the production deploy script is called ship.sh",
    "the staging database password rotates every monday",
    "chris is allergic to shellfish",
    "the team standup is at 9am on tuesdays",
    "chris drives a blue rivian",
]

# Each probe: the question, the ground-truth answer tokens the memory
# supplies, and a kind. "direct" = the question shares a keyword with the
# fact (lattice recall should get it). "paraphrase" = it does NOT (only
# the semantic embedder can bridge it — an honest lattice-only miss).
# "distractor" = no memory is relevant; injecting anything is pure cost.
PROBES = [
    # direct — lattice recall should land these
    {"q": "what is my wife's name?", "needs": ["brandi"], "kind": "direct"},
    {"q": "what are the kids names?", "needs": ["james", "traceton"], "kind": "direct"},
    {"q": "what programming language do I prefer?", "needs": ["python"], "kind": "direct"},
    {"q": "where do I live?", "needs": ["austin"], "kind": "direct"},
    {"q": "what is the deploy script called?", "needs": ["ship.sh"], "kind": "direct"},
    {"q": "when does the staging database password rotate?", "needs": ["monday"], "kind": "direct"},
    {"q": "what am I allergic to?", "needs": ["shellfish"], "kind": "direct"},
    {"q": "what car do I drive?", "needs": ["rivian"], "kind": "direct"},
    # paraphrase — no shared keyword; lattice misses, the embedder is the fix
    {"q": "who am I married to?", "needs": ["brandi"], "kind": "paraphrase"},
    {"q": "which city am I based in?", "needs": ["austin"], "kind": "paraphrase"},
    {"q": "who employs me?", "needs": ["glyphh"], "kind": "paraphrase"},
    # distractor — general/coding; memory is irrelevant, inject nothing
    {"q": "what is the capital of france?", "needs": [], "kind": "distractor"},
    {"q": "write a function to reverse a string", "needs": [], "kind": "distractor"},
    {"q": "what is 15 percent of 200?", "needs": [], "kind": "distractor"},
]

_CHARS_PER_TOKEN = 3.5   # standard rough estimate for the injected block


def _est_tokens(text: str) -> int:
    return round(len(text) / _CHARS_PER_TOKEN) if text else 0


async def _build_store():
    """A real SqlFactStore over a temp SQLite DB, filled with MEMORIES via
    the heuristic (offline, deterministic) enricher — the long-term
    lattice the rotor recalls from."""
    import tempfile

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from ada.encoder.llm_enricher import HeuristicEnricher
    from ada.memory.sql_store import SqlFactStore
    from infrastructure.database.connection import Base

    tmp = tempfile.mkdtemp()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp}/bench.db")
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = SqlFactStore(sf, space_id="main", enricher=HeuristicEnricher())
    for m in MEMORIES:
        await store.absorb(m, speaker="chris")
    return store, sf


class _Brain:
    def __init__(self, store):
        self._store = store

    def space(self, _sid):
        return self._store


async def run_retrieval() -> dict:
    """RETRIEVAL + COST — deterministic. For each probe, run the real rotor
    and inspect what it would inject."""
    from ada.rotor import rotor as rotor_mod

    store, sf = await _build_store()
    brain = _Brain(store)
    rows = []
    for p in PROBES:
        rotor_mod._CONV_CACHE.clear()   # each probe is its own conversation
        body = {"messages": [{"role": "user", "content": p["q"]}]}
        injection = await rotor_mod.spin(brain, sf, body) or ""
        low = injection.lower()
        hit = bool(p["needs"]) and all(n.lower() in low for n in p["needs"])
        rows.append({**p, "injection": injection,
                     "tokens": _est_tokens(injection), "hit": hit})
    return {"rows": rows}


async def run_semantic() -> dict | None:
    """SEMANTIC LIFT — the paraphrases lattice recall misses (no shared
    keyword) are exactly what a dedicated embedder should bridge. Write
    the memories as short-term turns with vectors, then recall the
    paraphrase probes through the semantic layer. None when no embedder
    answers (nomic not installed) — the lattice floor still stands."""
    import tempfile

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from ada.rotor import embed as embed_mod
    from ada.rotor import memory as memory_mod
    from infrastructure.database.connection import Base

    probe = MEMORIES[0]
    if await embed_mod.embed_text(probe, kind="document") is None:
        return None

    tmp = tempfile.mkdtemp()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp}/sem.db")
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    for i, m in enumerate(MEMORIES):        # each memory its own past conv
        await memory_mod.write_turn(sf, f"seed{i}", "user", m)

    rows = []
    for p in PROBES:
        if p["kind"] != "paraphrase":
            continue
        qv = await embed_mod.embed_text(p["q"], kind="query")
        hits = await memory_mod.semantic_recall(sf, qv or [], exclude_conv=None)
        found = " ".join(h["text"].lower() for h in hits["turns"])
        rows.append({"q": p["q"], "needs": p["needs"],
                     "hit": all(n.lower() in found for n in p["needs"])})
    return {"rows": rows}


# A realistic assistant system prompt — the rotor's injection is appended
# to something like this in a real Claude Desktop / Claude Code request,
# not to an empty string. We inject exactly as production does (via the
# real rotor.inject) so the A/B tests the actual format.
_BASE_SYSTEM = (
    "You are a helpful AI assistant. Be concise and direct. Answer in one "
    "sentence. If you genuinely do not have the information, say so plainly "
    "rather than guessing."
)
_REFUSAL = ("i don't know", "i do not know", "don't have", "do not have",
            "no information", "not sure", "cannot", "can't", "unable",
            "i'm not aware", "i am not aware", "no record", "as an ai")


async def _chat(system: str, user: str) -> str:
    """One proper system+user chat turn against the local model."""
    import httpx

    from ada.encoder.local_enricher import local_base_url, local_model_name
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(f"{local_base_url()}/chat/completions", json={
                "model": local_model_name(),
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": 80, "temperature": 0.0})
            if r.status_code != 200:
                return ""
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if "</think>" in txt:
                txt = txt.rsplit("</think>", 1)[-1].strip()
            return txt
    except Exception:
        return ""


async def run_e2e() -> dict | None:
    """END-TO-END A/B against the local model — the SAME realistic prompt
    answered baseline (no memory) vs treated (real rotor injection). None
    when no model answers. Scores correctness AND the meaningful flip:
    a baseline that can't answer (refuses / guesses) → a specific answer."""
    from ada.encoder.local_enricher import local_available
    from ada.rotor import rotor as rotor_mod

    if not local_available():
        return None
    store, sf = await _build_store()
    brain = _Brain(store)

    results, samples = [], []
    for p in PROBES:
        if p["kind"] == "distractor":
            continue
        rotor_mod._CONV_CACHE.clear()
        injection = await rotor_mod.spin(
            brain, sf, {"messages": [{"role": "user", "content": p["q"]}]}) or ""
        # treated system = the caller's prompt + the REAL injected block
        treated_sys = rotor_mod.inject({"system": _BASE_SYSTEM},
                                       injection)["system"] if injection else _BASE_SYSTEM
        base = await _chat(_BASE_SYSTEM, p["q"])
        treat = await _chat(treated_sys, p["q"])
        bl, tl = base.lower(), treat.lower()
        row = {
            "q": p["q"], "kind": p["kind"],
            "base_ok": bool(p["needs"]) and all(n.lower() in bl for n in p["needs"]),
            "treat_ok": bool(p["needs"]) and all(n.lower() in tl for n in p["needs"]),
            "base_refused": any(m in bl for m in _REFUSAL),
            "changed": base.strip() != treat.strip(),
            "tokens": _est_tokens(injection),
        }
        results.append(row)
        if len(samples) < 3 and row["treat_ok"] and not row["base_ok"]:
            samples.append({"q": p["q"], "base": base, "treat": treat})
    return {"results": results, "samples": samples}


def _pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "—"


def report(retr: dict, sem: dict | None, e2e: dict | None) -> str:
    rows = retr["rows"]
    out = ["", "═══ ROTOR BENCHMARK ═══", "",
           "RETRIEVAL & COST (deterministic — heuristic enricher, lattice only)",
           f"  {'kind':<11} {'question':<44} {'hit':<4} {'tokens'}"]
    by_kind: dict[str, list[bool]] = {}
    cost_relevant = cost_distractor = 0
    for r in rows:
        mark = "✓" if r["hit"] else ("·" if r["kind"] == "distractor" else "✗")
        out.append(f"  {r['kind']:<11} {r['q'][:44]:<44} {mark:<4} {r['tokens']}")
        if r["kind"] != "distractor":
            by_kind.setdefault(r["kind"], []).append(r["hit"])
            cost_relevant += r["tokens"]
        else:
            cost_distractor += r["tokens"]
    out.append("")
    for kind, hits in by_kind.items():
        out.append(f"  recall@lattice ({kind}): {_pct(sum(hits), len(hits))} "
                   f"({sum(hits)}/{len(hits)})")
    relevant = [r for r in rows if r["kind"] != "distractor"]
    distract = [r for r in rows if r["kind"] == "distractor"]
    out.append(f"  avg inject on relevant : ~{cost_relevant // max(len(relevant),1)} tokens")
    out.append(f"  avg inject on distractor: ~{cost_distractor // max(len(distract),1)} "
               f"tokens  (noise — want ≈0)")

    out += ["", "SEMANTIC LIFT (embedder recall on the paraphrases lattice missed)"]
    if sem is None:
        out.append("  skipped — no embedder (ollama pull nomic-embed-text)")
    else:
        s = sum(r["hit"] for r in sem["rows"])
        out.append(f"  recall@semantic (paraphrase): {_pct(s, len(sem['rows']))} "
                   f"({s}/{len(sem['rows'])})  — lattice-only got 0%")

    out += ["", "END-TO-END A/B (does the injection change the model's ANSWER)"]
    if e2e is None:
        out.append("  skipped — no local model at ADA_LOCAL_URL "
                   "(set one up + ADA_E2E=1 to run)")
    else:
        res = e2e["results"]
        n = len(res)
        b = sum(r["base_ok"] for r in res)
        t = sum(r["treat_ok"] for r in res)
        wins = sum(1 for r in res if r["treat_ok"] and not r["base_ok"])
        changed = sum(r["changed"] for r in res)
        refused = sum(r["base_refused"] for r in res)
        toks = sum(r["tokens"] for r in res) / max(n, 1)
        out.append(f"  baseline correct (no memory): {_pct(b, n)} ({b}/{n})")
        out.append(f"  rotor correct    (injected) : {_pct(t, n)} ({t}/{n})")
        out.append(f"  answers the rotor unlocked  : {wins}/{n} "
                   f"(baseline wrong/unknown → rotor right)")
        out.append(f"  responses that changed      : {changed}/{n}")
        out.append(f"  baseline said 'I don't know': {refused}/{n} "
                   f"(honest refusals the memory resolves)")
        out.append(f"  cost of the lift            : ~{toks:.0f} input tokens / question")
        for s in e2e.get("samples", []):
            out += ["", f"    Q: {s['q']}",
                    f"      baseline: {s['base'][:100]}",
                    f"      + rotor : {s['treat'][:100]}"]
    out.append("")
    return "\n".join(out)


async def main() -> None:
    retr = await run_retrieval()
    sem = await run_semantic()
    e2e = await run_e2e() if os.environ.get("ADA_E2E") else None
    print(report(retr, sem, e2e))


if __name__ == "__main__":
    asyncio.run(main())

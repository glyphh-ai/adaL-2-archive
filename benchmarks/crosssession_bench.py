"""
Cross-session A/B — Ada vs no-Ada on what Claude structurally CANNOT do.

Claude (any stateless LLM) sees only the current conversation. It cannot
recall what you said in a *previous* session, cannot know your private
repo's conventions, cannot track that a decision *changed*, and cannot
accumulate context across chats. Those aren't quality gaps — they're
structural: the information is neither in training data nor in the
current prompt. "Without Ada" here IS Claude's real behavior.

Each scenario seeds memory as if from PAST sessions, then asks questions
in a FRESH session (new conversation, empty history). The A/B:

    without Ada : the model gets only the question — stateless Claude
    with Ada    : the rotor recalls the past-session facts and injects them

Categories (all things a stateless model genuinely can't do):
  repo    — your private codebase's conventions (not in training data)
  version — a decision that CHANGED; the current belief must win, and the
            old value must not leak (naive "replay all past turns" fails)
  recall  — something told in passing a session ago
  synth   — accumulate facts across sessions, then synthesize
  override— the current turn contradicts memory; the user must win
            (a preview of the adversarial/prompt-manipulation work)

Run:  ADA_LOCAL_URL=... ADA_EMBED_URL=... ADA_E2E=1 \
      python -m benchmarks.crosssession_bench
The deterministic retrieval half is gated in tests/test_crosssession_bench.py.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from benchmarks.rotor_bench import (_BASE_SYSTEM, _chat, _est_tokens, _pct)

# Each memory: (text, key). key set → a versioned fact (later writes to the
# same key supersede earlier ones). key None → a plain fact.
SCENARIOS = [
    {
        "name": "repo", "why": "your private codebase — not in any training set",
        "memories": [
            ("this project uses pnpm, never npm", None),
            ("unit tests run with the command pnpm test:unit", None),
            ("we deploy to staging by running ./scripts/ship.sh staging", None),
            ("the authentication code lives in src/core/auth", None),
            ("never edit anything under the generated/ folder", None),
        ],
        "probes": [
            {"q": "what command runs the unit tests here?", "needs": ["pnpm test:unit"], "det": True},
            {"q": "how do I deploy to staging?", "needs": ["ship.sh"], "det": True},
            {"q": "where does the authentication code live?", "needs": ["src/core/auth"], "det": True},
        ],
    },
    {
        "name": "version", "why": "a decision that CHANGED — statelessness can't track it",
        "memories": [
            ("the launch date is october 15", "launch.date"),
            ("the launch date moved to november 3", "launch.date"),   # supersedes
            ("we are building the api in REST", "api.style"),
            ("we switched the api to GraphQL", "api.style"),           # supersedes
        ],
        "probes": [
            {"q": "when is the launch?", "needs": ["november"], "avoid": ["october"], "det": True},
            {"q": "what api style are we using?", "needs": ["graphql"], "avoid": [], "det": True},
        ],
    },
    {
        "name": "recall", "why": "told in passing a session ago",
        "memories": [
            ("my manager's name is priya", None),
            ("the team standup is at 9am on tuesdays", None),
            ("the staging database password rotates every monday", None),
        ],
        "probes": [
            {"q": "who is my manager?", "needs": ["priya"], "det": True},
            {"q": "what day does the staging database password rotate?", "needs": ["monday"], "det": True},
        ],
    },
    {
        "name": "synth", "why": "accumulate across sessions, then synthesize",
        "memories": [
            ("my wife brandi is a vegetarian", None),
            ("i am allergic to shellfish", None),
            ("we have two kids james and traceton", None),
        ],
        "probes": [
            # a realistic assistant task — the constraints only exist in memory
            {"q": "suggest what to cook for a family dinner tonight — keep it "
                  "suitable for everyone here", "needs": [], "any": ["vegetarian", "shellfish", "meat-free", "no shellfish"], "det": False},
        ],
    },
    {
        "name": "override", "why": "current turn contradicts memory — the user must win",
        "memories": [
            ("chris's favorite editor is vim", "editor.fav"),
        ],
        "probes": [
            {"q": "I've switched to VS Code — give me one tip to set it up",
             "needs": [], "any": ["vs code", "vscode", "code"], "avoid": ["vim"], "det": False},
        ],
    },
]


async def _build_store(memories):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from ada.encoder.llm_enricher import HeuristicEnricher
    from ada.memory.sql_store import SqlFactStore
    from infrastructure.database.connection import Base

    tmp = tempfile.mkdtemp()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp}/xs.db")
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = SqlFactStore(sf, space_id="main", enricher=HeuristicEnricher())
    for text, key in memories:          # order matters — later versions supersede
        await store.absorb(text, speaker="chris", key=key)
    return store, sf


class _Brain:
    def __init__(self, store):
        self._store = store

    def space(self, _):
        return self._store


async def _injection_for(store, sf, question):
    from ada.rotor import rotor as rotor_mod
    rotor_mod._CONV_CACHE.clear()
    return await rotor_mod.spin(_Brain(store), sf,
                                {"messages": [{"role": "user", "content": question}]}) or ""


def _scored(text, probe):
    low = text.lower()
    ok = True
    if probe.get("needs"):
        ok = all(n.lower() in low for n in probe["needs"])
    if probe.get("any"):
        ok = ok and any(a.lower() in low for a in probe["any"])
    if probe.get("avoid"):
        ok = ok and not any(a.lower() in low for a in probe["avoid"])
    return ok


async def run(with_model: bool) -> dict:
    from ada.rotor import rotor as rotor_mod

    scenarios = []
    for sc in SCENARIOS:
        store, sf = await _build_store(sc["memories"])
        rows = []
        for p in sc["probes"]:
            injection = await _injection_for(store, sf, p["q"])
            retrieved = _scored(injection, {"needs": p.get("needs"),
                                            "any": p.get("any")})
            row = {"q": p["q"], "det": p.get("det", False),
                   "tokens": _est_tokens(injection), "retrieved": retrieved,
                   "injection": injection, "avoid": p.get("avoid", [])}
            if with_model:
                treated_sys = rotor_mod.inject({"system": _BASE_SYSTEM},
                                               injection)["system"] if injection else _BASE_SYSTEM
                base = await _chat(_BASE_SYSTEM, p["q"])
                treat = await _chat(treated_sys, p["q"])
                row["base_ok"] = _scored(base, p)
                row["treat_ok"] = _scored(treat, p)
                row["base"], row["treat"] = base, treat
            rows.append(row)
        scenarios.append({"name": sc["name"], "why": sc["why"], "rows": rows})
    return {"scenarios": scenarios, "with_model": with_model}


def report(res: dict) -> str:
    wm = res["with_model"]
    out = ["", "═══ CROSS-SESSION A/B — Ada vs stateless (what Claude can't do) ═══"]
    all_base = all_treat = all_ret = all_n = 0
    for sc in res["scenarios"]:
        out += ["", f"  [{sc['name']}] {sc['why']}"]
        for r in sc["rows"]:
            all_n += 1
            all_ret += r["retrieved"]
            if wm:
                all_base += r["base_ok"]
                all_treat += r["treat_ok"]
                flip = "✓ UNLOCKED" if (r["treat_ok"] and not r["base_ok"]) else \
                       ("= both" if r["treat_ok"] == r["base_ok"] else "rotor worse")
                out.append(f"    {r['q'][:52]:<52} {flip}")
                if r["treat_ok"] and not r["base_ok"]:
                    out.append(f"        without Ada: {r['base'][:78]}")
                    out.append(f"        with Ada   : {r['treat'][:78]}")
            else:
                out.append(f"    {r['q'][:56]:<56} {'✓ recalled' if r['retrieved'] else '✗ missed'}")
    out += ["", f"  retrieval (Ada surfaced the fact): {_pct(all_ret, all_n)} ({all_ret}/{all_n})"]
    if wm:
        out += [f"  without Ada correct (stateless) : {_pct(all_base, all_n)} ({all_base}/{all_n})",
                f"  with Ada correct                : {_pct(all_treat, all_n)} ({all_treat}/{all_n})",
                f"  tasks Ada unlocked              : "
                f"{sum(1 for sc in res['scenarios'] for r in sc['rows'] if r.get('treat_ok') and not r.get('base_ok'))}/{all_n}"]
    out.append("")
    return "\n".join(out)


async def main() -> None:
    from ada.encoder.local_enricher import local_available
    print(report(await run(with_model=bool(os.environ.get("ADA_E2E")) and local_available())))


if __name__ == "__main__":
    asyncio.run(main())

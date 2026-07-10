"""
The real test: does Ada make CLAUDE better — and by how much, which way.

Hypothesis: with Ada, Claude is dramatically better (recalls what it
cannot know) OR dramatically worse (irrelevant memory derails it). So we
measure the effect in BOTH directions against real Claude:

  HELP     memory Claude structurally cannot have (private repo, a fact
           that changed, something said a session ago) — expect a large
           lift, treated >> baseline.
  HARM     the SAME memory forced into an unrelated general-knowledge
           question — expect NO drop (Claude ignores irrelevant context);
           a drop here is the "10x worse" failure mode.
  OVERRIDE the current turn contradicts memory — the user must win; the
           injected fact must NOT hijack the answer.

Each probe is asked twice against the same real Claude model: baseline
(the caller's system prompt only) vs treated (that prompt + Ada's REAL
injection, built by the actual rotor). Memory injection is model-agnostic
by design — the future prompt-router picks the A/B/C model, Ada injects
into whichever — so this measures Ada's contribution independent of which
model answers.

Run (needs a key in .env or the environment; makes real, billed calls):
    ADA_CLAUDE_MODEL=claude-sonnet-4-5 python -m benchmarks.claude_ab
"""

from __future__ import annotations

import asyncio
import os

import httpx

from benchmarks.crosssession_bench import _BASE_SYSTEM, _build_store, _injection_for
from ada.rotor import rotor as _rotor

MODEL = os.environ.get("ADA_CLAUDE_MODEL", "claude-sonnet-4-5")

REPO = [("this project uses pnpm, never npm", None),
        ("unit tests run with the command pnpm test:unit", None),
        ("we deploy to staging by running ./scripts/ship.sh staging", None),
        ("the authentication code lives in src/core/auth", None)]
VERSION = [("the launch date is october 15", "launch.date"),
           ("the launch date moved to november 3", "launch.date"),
           ("we are building the api in REST", "api.style"),
           ("we switched the api to GraphQL", "api.style")]
RECALL = [("my manager's name is priya", None),
          ("the team standup is at 9am on tuesdays", None),
          ("the staging database password rotates every monday", None)]
PERSONAL = [("chris's wife is brandi", None),
            ("i am allergic to shellfish", None),
            ("we have two kids james and traceton", None)]


def _key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key


async def _build_probes() -> list[dict]:
    """(question, ground truth, and the REAL rotor injection) per probe."""
    async def inj(memories, q):
        store, sf = await _build_store(memories)
        return await _injection_for(store, sf, q)

    def treated(injection):
        return _rotor.inject({"system": _BASE_SYSTEM}, injection)["system"] if injection else _BASE_SYSTEM

    probes = []

    async def add(pid, memories, q, *, needs=(), any_=(), avoid=(), direction="help", force=None):
        injection = force if force is not None else await inj(memories, q)
        probes.append({"id": pid, "q": q, "needs": list(needs), "any": list(any_),
                       "avoid": list(avoid), "direction": direction,
                       "treated_system": treated(injection), "injected": bool(injection)})

    # HELP — Claude cannot know these
    await add("repo-tests", REPO, "what command runs the unit tests in this project?", needs=["pnpm test:unit"])
    await add("repo-deploy", REPO, "how do I deploy this to staging?", needs=["ship.sh"])
    await add("repo-auth", REPO, "where does the authentication code live in this repo?", needs=["src/core/auth"])
    await add("ver-launch", VERSION, "when is the launch?", needs=["november"], avoid=["october"])
    await add("ver-api", VERSION, "what api style are we using?", needs=["graphql"], avoid=["rest"])
    await add("recall-mgr", RECALL, "who is my manager?", needs=["priya"])
    await add("recall-pw", RECALL, "what day does the staging database password rotate?", needs=["monday"])
    await add("recall-wife", PERSONAL, "what is my wife's name?", needs=["brandi"])
    # HARM — irrelevant personal memory forced into a general question
    noise = await inj(PERSONAL, "what is my wife's name?")
    await add("harm-capital", None, "what is the capital of France?", needs=["paris"], direction="harm", force=noise)
    await add("harm-math", None, "what is 15 percent of 240?", needs=["36"], direction="harm", force=noise)
    # OVERRIDE — memory says vim; the user just switched to VS Code
    await add("override", None, "I've switched to VS Code from my old editor — give me one quick setup tip.",
              any_=["vs code", "vscode", "code"], avoid=["vim"], direction="override",
              force="chris's favorite editor is vim")
    return probes


def _correct(text: str, p: dict) -> bool:
    low = text.lower()
    ok = all(n.lower() in low for n in p["needs"])
    if p["any"]:
        ok = ok and any(a.lower() in low for a in p["any"])
    if p["avoid"]:
        ok = ok and not any(a.lower() in low for a in p["avoid"])
    return ok


async def _ask(http: httpx.AsyncClient, key: str, system: str, q: str) -> str:
    r = await http.post("https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": MODEL, "max_tokens": 120, "system": system,
                              "messages": [{"role": "user", "content": q}]})
    if r.status_code != 200:
        return f"[error {r.status_code}]"
    return "".join(b.get("text", "") for b in r.json().get("content", [])
                   if b.get("type") == "text").strip()


async def run() -> dict:
    key = _key()
    if not key:
        raise SystemExit("no ANTHROPIC_API_KEY (put it in .env or the environment)")
    probes = await _build_probes()
    sem = asyncio.Semaphore(4)

    async def one(http, p):
        async with sem:
            base = await _ask(http, key, _BASE_SYSTEM, p["q"])
            treat = await _ask(http, key, p["treated_system"], p["q"])
        return {**p, "base": base, "treat": treat,
                "base_ok": _correct(base, p), "treat_ok": _correct(treat, p)}

    async with httpx.AsyncClient(timeout=60.0) as http:
        rows = await asyncio.gather(*[one(http, p) for p in probes])
    return {"model": MODEL, "rows": rows}


def report(res: dict) -> str:
    rows = res["rows"]
    out = ["", f"═══ DOES ADA MAKE CLAUDE BETTER — real A/B ({res['model']}) ═══"]

    def block(name, direction, note):
        sub = [r for r in rows if r["direction"] == direction]
        b = sum(r["base_ok"] for r in sub)
        t = sum(r["treat_ok"] for r in sub)
        out.append("")
        out.append(f"  {name}  ({note})")
        out.append(f"    baseline Claude : {b}/{len(sub)}")
        out.append(f"    Claude + Ada    : {t}/{len(sub)}")
        for r in sub:
            flip = "UNLOCKED" if (r["treat_ok"] and not r["base_ok"]) else \
                   ("REGRESSED" if (r["base_ok"] and not r["treat_ok"]) else "same")
            out.append(f"      [{flip:<9}] {r['q'][:50]}")
            if r["direction"] != "help" or (r["treat_ok"] and not r["base_ok"]):
                out.append(f"          baseline: {r['base'][:80]}")
                out.append(f"          + Ada   : {r['treat'][:80]}")
        return b, t, len(sub)

    hb, ht, hn = block("HELP", "help", "memory Claude can't have — expect a big lift")
    xb, xt, xn = block("HARM", "harm", "irrelevant memory forced in — expect NO drop")
    block("OVERRIDE", "override", "user contradicts memory — user must win")

    out += ["", "  ── verdict ──",
            f"  HELP : baseline {hb}/{hn} → +Ada {ht}/{hn}  "
            f"({'+' if ht >= hb else ''}{ht - hb} answers Ada unlocked)",
            f"  HARM : baseline {xb}/{xn} → +Ada {xt}/{xn}  "
            f"({'no regression' if xt >= xb else f'{xb - xt} REGRESSED — noise hurt'})",
            ""]
    return "\n".join(out)


async def main() -> None:
    print(report(await run()))


if __name__ == "__main__":
    asyncio.run(main())

"""The rotor benchmark's deterministic half, as a CI gate.

Makes two claims about the rotor regression-protected — the exact trade
the memory proxy is selling:

  1. On memory-relevant questions it recalls the needed fact (high recall,
     small injection).
  2. On memory-irrelevant questions it injects NOTHING (zero token cost —
     the proxy is free when memory can't help).

The model-dependent halves (semantic lift, end-to-end A/B) are not gated
here — they need a local model — but the harness that runs them lives in
benchmarks/rotor_bench.py and is exercised by hand / in the nightly.
"""

import asyncio
import os

from benchmarks.rotor_bench import run_retrieval


def _rows():
    os.environ.setdefault("ADA_ENRICHER", "heuristic")   # offline, deterministic
    return asyncio.run(run_retrieval())["rows"]


def test_direct_questions_recall_the_fact():
    rows = _rows()
    direct = [r for r in rows if r["kind"] == "direct"]
    hits = sum(r["hit"] for r in direct)
    assert hits == len(direct), (
        f"lattice recall regressed: {hits}/{len(direct)} direct probes hit "
        f"(missed: {[r['q'] for r in direct if not r['hit']]})")


def test_distractors_inject_nothing():
    """The efficiency guarantee: a prompt memory can't help must cost zero
    injected tokens — no noise, no burn."""
    rows = _rows()
    for r in rows:
        if r["kind"] == "distractor":
            assert r["tokens"] == 0, (
                f"distractor injected {r['tokens']} tokens of noise: {r['q']!r}\n"
                f"  → {r['injection']!r}")


def test_relevant_injection_stays_small():
    """A recalled fact is a handful of tokens, not a wall — the trade only
    works if the cost per hit is tiny."""
    rows = _rows()
    for r in rows:
        if r["kind"] != "distractor" and r["hit"]:
            assert r["tokens"] < 120, f"injection too fat ({r['tokens']} tok): {r['q']!r}"

"""Cross-session retrieval as a CI gate — the deterministic half of
benchmarks/crosssession_bench.py.

Proves, without a model, that Ada recalls what a stateless LLM cannot:
past-session facts, private-repo conventions, and — the sharp one — the
CURRENT belief of a fact that changed (November, not the superseded
October), with the old value not leaking. The model A/B (answer flips)
needs a local model and is exercised by hand / nightly, not gated here.
"""

import asyncio
import os

from benchmarks.crosssession_bench import run


def _det_rows():
    os.environ.setdefault("ADA_ENRICHER", "heuristic")
    res = asyncio.run(run(with_model=False))
    return [(sc["name"], r) for sc in res["scenarios"] for r in sc["rows"]
            if r["det"]]


def test_deterministic_probes_all_retrieve():
    rows = _det_rows()
    missed = [(name, r["q"]) for name, r in rows if not r["retrieved"]]
    assert not missed, f"cross-session recall regressed: {missed}"


def test_changed_fact_returns_current_not_superseded():
    """The version scenario is the one a naive 'replay all past turns'
    approach gets wrong: recall must surface the NEW value and drop the
    old one. Verified via the injection text itself."""
    os.environ.setdefault("ADA_ENRICHER", "heuristic")
    res = asyncio.run(run(with_model=False))
    version = next(sc for sc in res["scenarios"] if sc["name"] == "version")
    launch = next(r for r in version["rows"] if "launch" in r["q"])
    inj = launch["injection"].lower()
    assert "november" in inj, "current belief (november) not recalled"
    assert "october" not in inj, "superseded value (october) leaked into recall"

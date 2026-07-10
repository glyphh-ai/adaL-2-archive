"""auto_enricher honors ADA_ENRICHER and degrades gracefully.

Fully offline: the "endpoint up" case is monkeypatched; the "endpoint
down" cases point ADA_LOCAL_URL at a port nothing listens on, so the
probe fails instantly with connection-refused.
"""

import os
from contextlib import contextmanager

from ada.encoder.llm_enricher import HeuristicEnricher, auto_enricher
from ada.encoder.local_enricher import LocalUniversalEnricher

DEAD_URL = "http://127.0.0.1:9"  # discard port — nothing listens


@contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_heuristic_forced():
    with env(ADA_ENRICHER="heuristic"):
        assert isinstance(auto_enricher(), HeuristicEnricher)


def test_auto_uses_local_when_endpoint_up(monkeypatch):
    import ada.encoder.local_enricher as le
    monkeypatch.setattr(le, "local_available", lambda *a, **k: True)
    with env(ADA_ENRICHER=None):
        assert isinstance(auto_enricher(), LocalUniversalEnricher)


def test_auto_falls_to_heuristic_when_endpoint_down():
    with env(ADA_ENRICHER=None, ADA_LOCAL_URL=DEAD_URL):
        assert isinstance(auto_enricher(), HeuristicEnricher)


def test_local_forced_but_down_falls_back_not_crash():
    with env(ADA_ENRICHER="local", ADA_LOCAL_URL=DEAD_URL):
        enr = auto_enricher()
        assert isinstance(enr, HeuristicEnricher)  # graceful fallback

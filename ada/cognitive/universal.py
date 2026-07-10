"""
The universal schema — every fact maps into one fixed shape.

Instead of growing ad-hoc predicates ("favorite_color", "has_dream_loop",
"quantity_of_bones") as an LLM invents them, every fact gets MAPPED into
a fixed 7-layer lattice. The substrate stays stable; structured queries
(count / distribution / intersection) address facts by layer.role slot.

LAYERS (fixed):
    entity        — what kind of thing is this
    perceptual    — color / size / shape / sound / texture / temperature
    spatial       — location / origin / direction
    temporal      — time / duration / age / era / frequency
    relational    — subject / predicate / object / possessor / agent
    quantitative  — count / magnitude / unit
    epistemic     — source / certainty / modality

Every fact fills SOME of these slots (the rest are absent). An absent
slot is a structural ∅ — the substrate refuses rather than confabulates.

The write-time extractor that maps free text into this schema is the
LOCAL model enricher (ada/encoder/local_enricher.py — Ollama or
llama-server, no cloud key). It runs once per fact and is cached; the
read path never calls a model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── The universal schema ─────────────────────────────────────────────

UNIVERSAL_SCHEMA: dict[str, list[str]] = {
    "entity":       ["name", "kind", "subkind"],
    "perceptual":   ["color", "size", "shape", "texture", "sound",
                     "smell", "taste", "temperature"],
    "spatial":      ["location", "origin", "direction"],
    "temporal":     ["time", "duration", "age", "era", "frequency"],
    "relational":   ["subject", "predicate", "object", "possessor",
                     "agent", "patient", "instrument"],
    "quantitative": ["count", "magnitude", "unit", "ratio"],
    "epistemic":    ["source", "certainty", "modality"],
}

ALL_LAYERS = list(UNIVERSAL_SCHEMA)
ALL_ROLES = [(layer, role) for layer, roles in UNIVERSAL_SCHEMA.items() for role in roles]


# ── Enricher: LLM maps a fact INTO the fixed shape ──────────────────

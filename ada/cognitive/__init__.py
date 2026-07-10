"""
ada.cognitive — the universal schema and the cognitive surfaces on it.

    UNIVERSAL_SCHEMA   — the fixed 7-layer × 33-role lattice
    LocalUniversalEnricher — local-model text → schema extractor
    CognitiveSurface   — think / ask on a ThoughtSpace
    build_llm_renderer — optional LLM surface form for answers
"""

from ada.encoder.local_enricher import LocalUniversalEnricher
from .universal import ALL_LAYERS, ALL_ROLES, UNIVERSAL_SCHEMA
from .surface import Activation, Answer, CognitiveSurface
from .generate import build_llm_renderer

__all__ = [
    "UNIVERSAL_SCHEMA",
    "ALL_LAYERS",
    "ALL_ROLES",
    "LocalUniversalEnricher",
    "CognitiveSurface",
    "Activation",
    "Answer",
    "build_llm_renderer",
]

"""
ada.encoder — model-at-build-time fact enrichment, fully local.

    auto_enricher() — picks the best available enricher (the local model
                      at ADA_LOCAL_URL when it answers, regex heuristics
                      otherwise)
"""

from ada.encoder.llm_enricher import (
    FactEnricher,
    HeuristicEnricher,
    StructuredFact,
    auto_enricher,
)
from ada.encoder.local_enricher import (
    LocalUniversalEnricher,
    local_available,
    local_base_url,
    local_model_name,
)

__all__ = [
    "FactEnricher",
    "HeuristicEnricher",
    "StructuredFact",
    "LocalUniversalEnricher",
    "local_available",
    "local_base_url",
    "local_model_name",
    "auto_enricher",
]

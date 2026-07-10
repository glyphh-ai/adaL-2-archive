"""
LLM-at-build-time fact enricher.

Parses incoming facts into a structured `{subject, predicate, object, topic}`
tuple so recall can match on discriminative structure (predicate/subject)
instead of raw surface text. The substrate doesn't store raw sentences
alone — it stores parsed structure alongside them.

Two enrichers — both fully local; no cloud key exists in Ada:

  LocalUniversalEnricher — a small local model (Ollama / llama-server,
                       OpenAI-compatible endpoint) maps each fact into
                       the fixed universal schema, projected back to a
                       StructuredFact. Activates when the endpoint at
                       ADA_LOCAL_URL answers; see local_enricher.py.

  HeuristicEnricher  — regex patterns for common templates. Works with
                       nothing running at all; narrower coverage but
                       enough for demos and held-out tests on
                       similar-shaped corpora.

The enricher returns the SAME shape regardless of which backend is in use,
so the substrate stays agnostic. Failures degrade to topic="general" with
an empty triple, which is no worse than the un-enriched path.

Usage:
    from ada.encoder.llm_enricher import auto_enricher
    enr = auto_enricher()
    fact = enr.enrich("Bob's favorite color is red.")
    # → StructuredFact(subject="Bob", predicate="favorite_color",
    #                  object="red", topic="person.preference")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class StructuredFact:
    """Parsed representation of a single fact."""
    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    topic: str = "general"

    def is_structured(self) -> bool:
        return bool(self.subject or self.predicate or self.object)


# ── Protocol ─────────────────────────────────────────────────────────

class FactEnricher(Protocol):
    """Anything that turns text → StructuredFact."""
    def enrich(self, text: str) -> StructuredFact: ...


# ── Heuristic enricher (offline) ─────────────────────────────────────

def _strip_punct(s: str) -> str:
    return s.strip().strip("?.,!").lower()


def _try_facts(text: str) -> Optional[StructuredFact]:
    """Statement-shaped facts. Returns None if nothing matches."""

    # "X's favorite Y is Z."  ─ predicate = "favorite_Y"
    m = re.match(r"^(\w+)(?:'s|s)\s+favorite\s+(\w+)\s+is\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)),
            predicate=f"favorite_{m.group(2).lower()}",
            object=_strip_punct(m.group(3)),
            topic="person.preference")

    # "X is N years old."  ─ predicate = "age"
    m = re.match(r"^(\w+)\s+is\s+(\d+)\s+years?\s+old\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="age",
            object=m.group(2), topic="person.age")

    # "X lives in Y."  ─ predicate = "residence"
    m = re.match(r"^(\w+)\s+lives?\s+in\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="residence",
            object=_strip_punct(m.group(2)), topic="person.location")

    # "X works as (a|an) Y."  ─ predicate = "job"
    m = re.match(r"^(\w+)\s+works?\s+as\s+(?:a|an)?\s*(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="job",
            object=_strip_punct(m.group(2)), topic="person.job")

    # "The capital of X is Y."  ─ predicate = "capital"
    m = re.match(r"^The\s+capital\s+of\s+(.+?)\s+is\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="capital",
            object=_strip_punct(m.group(2)), topic="geography.capital")

    # "X stands for Y."  ─ predicate = "abbreviation_of"
    m = re.match(r"^(\w+)\s+stands\s+for\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="abbreviation_of",
            object=_strip_punct(m.group(2)), topic="knowledge.definition")

    # "The X has N Y."  ─ predicate = "quantity_of_Y"
    m = re.match(r"^The\s+(\w+(?:\s+\w+)?)\s+has\s+(\d+)\s+(\w+)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)),
            predicate=f"quantity_of_{m.group(3).lower()}",
            object=m.group(2), topic="knowledge.quantity")

    # "Ada uses N-dimensional Y."  ─ predicate = "dimension"
    m = re.match(r"^(\w+)\s+uses\s+(\d+)-dimensional\s+(.+?)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="dimension",
            object=m.group(2), topic="ada.architecture")

    # "The Ada server runs on port N."
    m = re.match(r"^The\s+(\w+)\s+server\s+runs\s+on\s+port\s+(\d+)\.?$", text, re.I)
    if m:
        return StructuredFact(text=text,
            subject=_strip_punct(m.group(1)), predicate="port",
            object=m.group(2), topic="ada.architecture")

    return None


def _try_queries(text: str) -> Optional[StructuredFact]:
    """Question-shaped queries → same triple shape so they hash to the
    same structural keys as the facts that should answer them."""

    # "What is X's age?" / "How old is X?"  ─ predicate = "age"
    m = re.match(r"^(?:what\s+is\s+(\w+)(?:'s|s)\s+age|how\s+old\s+is\s+(\w+))\??$",
                 text, re.I)
    if m:
        subj = m.group(1) or m.group(2)
        return StructuredFact(text=text, subject=_strip_punct(subj),
            predicate="age", topic="person.age")

    # "Where does X live?" / "Where is X?"  ─ predicate = "residence"
    m = re.match(r"^where\s+does\s+(\w+)\s+live\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="residence", topic="person.location")

    # "What job does X have?" / "What does X do?"  ─ predicate = "job"
    m = re.match(r"^(?:what\s+job\s+does\s+(\w+)\s+have|what\s+does\s+(\w+)\s+do)\??$",
                 text, re.I)
    if m:
        subj = m.group(1) or m.group(2)
        return StructuredFact(text=text, subject=_strip_punct(subj),
            predicate="job", topic="person.job")

    # "What is X's favorite Y?"  ─ predicate = "favorite_Y"
    m = re.match(r"^what\s+is\s+(\w+)(?:'s|s)\s+favorite\s+(\w+)\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate=f"favorite_{m.group(2).lower()}",
            topic="person.preference")
    # "What color does X like?"
    m = re.match(r"^what\s+color\s+does\s+(\w+)\s+like\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="favorite_color", topic="person.preference")

    # "What is the capital of X?" / "Capital of X?" / "Capital city of X?"
    m = re.match(r"^(?:what\s+is\s+the\s+capital\s+of|where\s+is\s+the\s+capital\s+of|"
                 r"capital\s+(?:city\s+)?of|what\s+is\s+(\w+)(?:'s|s)\s+capital)\s*(\w+)?\??$",
                 text, re.I)
    if m:
        subj = m.group(2) or m.group(1)
        if subj:
            return StructuredFact(text=text, subject=_strip_punct(subj),
                predicate="capital", topic="geography.capital")

    # "What does X stand for?"
    m = re.match(r"^what\s+does\s+(\w+)\s+stand\s+for\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="abbreviation_of", topic="knowledge.definition")

    # "What dimension are X's vectors?"
    m = re.match(r"^what\s+dimension\s+are\s+(\w+)(?:'s|s)\s+vectors\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="dimension", topic="ada.architecture")

    # "What port does X run on?"
    m = re.match(r"^what\s+port\s+does\s+(\w+)\s+run\s+on\??$", text, re.I)
    if m:
        return StructuredFact(text=text, subject=_strip_punct(m.group(1)),
            predicate="port", topic="ada.architecture")

    return None


class HeuristicEnricher:
    """Regex-based fact + query parser. Cheap, offline, narrow coverage."""

    def enrich(self, text: str) -> StructuredFact:
        clean = text.strip()
        return _try_facts(clean) or _try_queries(clean) or StructuredFact(text=clean)

    # predicate -> (layer, role) for projecting the parsed triple into
    # universal-schema slots, so offline `tell` still feeds the
    # structured queries (narrow coverage, same slots the LLM uses).
    _PRED_SLOT = {
        "age": ("temporal", "age"),
        "residence": ("spatial", "location"),
        "favorite_color": ("perceptual", "color"),
    }

    def universal(self, text: str) -> dict[str, dict[str, str]]:
        sf = self.enrich(text)
        if not sf.is_structured() or not sf.object:
            return {}
        mapped: dict[str, dict[str, str]] = {}
        if sf.subject:
            mapped["entity"] = {"name": sf.subject}
        layer_role = self._PRED_SLOT.get(sf.predicate)
        if layer_role:
            layer, role = layer_role
            mapped.setdefault(layer, {})[role] = sf.object
            if sf.subject:
                mapped["relational"] = {"subject": sf.subject,
                                        "predicate": sf.predicate}
        else:
            mapped["relational"] = {
                "subject": sf.subject, "predicate": sf.predicate,
                "object": sf.object,
            }
        return mapped


# ── Projection: universal mapping → StructuredFact ─────────────────────

def _project_to_structured(text: str, mapped: dict[str, dict[str, str]]) -> StructuredFact:
    """Collapse a universal-schema mapping into a StructuredFact.

    The projection picks the most discriminative role from the universal
    mapping to fill (subject, predicate, object, topic):

      subject   = relational.subject || entity.name || possessor
      predicate = the most specific role that's filled, expressed as
                  'layer.role' so recall's predicate-match still
                  distinguishes age vs color vs location
      object    = relational.object || the filled role's value
      topic     = '{entity.kind}.{primary_layer}' where primary_layer is
                  the most-filled non-relational layer
    """
    def get(layer: str, role: str) -> str:
        return mapped.get(layer, {}).get(role, "") or ""

    subject = (
        get("relational", "subject")
        or get("relational", "possessor")
        or get("entity", "name")
    ).strip().lower()

    # Pick the most "discriminative" role: prefer a perceptual/temporal/
    # spatial/quantitative filling over a generic relational one.
    PRIORITY = [
        ("perceptual", "color"),  ("perceptual", "size"),
        ("perceptual", "shape"),  ("perceptual", "sound"),
        ("perceptual", "texture"),("perceptual", "temperature"),
        ("temporal",   "age"),    ("temporal",   "time"),
        ("temporal",   "duration"),("temporal",  "frequency"),
        ("spatial",    "location"),("spatial",   "origin"),
        ("quantitative", "magnitude"), ("quantitative", "count"),
        ("relational", "predicate"),
    ]
    predicate = ""
    obj = get("relational", "object")
    for layer, role in PRIORITY:
        v = get(layer, role)
        if v:
            predicate = f"{layer}.{role}"
            if not obj:
                obj = v
            break

    # Topic: '{kind}.{primary_layer}'.
    kind = (get("entity", "kind") or "thing").strip().lower().replace(" ", "_")
    layer_fills = {
        layer: sum(1 for v in roles.values() if v)
        for layer, roles in mapped.items()
        if layer != "relational"
    }
    primary_layer = max(layer_fills, key=layer_fills.get) if layer_fills else "entity"
    topic = f"{kind}.{primary_layer}"

    return StructuredFact(
        text=text,
        subject=subject,
        predicate=predicate.lower(),
        object=str(obj).strip().lower(),
        topic=topic,
    )


# ── Factory ──────────────────────────────────────────────────────────

def auto_enricher() -> FactEnricher:
    """Pick the enricher. ADA_ENRICHER forces one explicitly:

      local     — the local model at ADA_LOCAL_URL (Ollama/llama-server),
                  zero external calls / no data egress
      heuristic — regex patterns, no model at all

    Unset → auto: local when the endpoint answers, else heuristic.
    A forced choice that can't initialize falls back with a warning —
    a boot never hard-fails over enrichment.
    """
    choice = (os.environ.get("ADA_ENRICHER") or "").strip().lower()

    if choice == "heuristic":
        logger.info("Enricher: HeuristicEnricher (forced, offline)")
        return HeuristicEnricher()

    from ada.encoder.local_enricher import (LocalUniversalEnricher,
                                            local_available,
                                            local_model_name)
    if local_available():
        logger.info("Enricher: LocalUniversalEnricher (%s — local, no external calls)",
                    local_model_name())
        return LocalUniversalEnricher()
    if choice == "local":
        logger.warning("ADA_ENRICHER=local but nothing answers at the local "
                       "endpoint — using heuristic (start Ollama or llama-server)")
    else:
        logger.info("Enricher: HeuristicEnricher (no local model endpoint up)")
    return HeuristicEnricher()

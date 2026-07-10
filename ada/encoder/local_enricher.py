"""
Local universal-schema enricher — zero external LLM calls, zero data
egress. A small local model maps prose into the fixed universal schema
at write time, served by any OpenAI-compatible local runtime: Ollama
(Mac/Windows/Linux installers) or llama.cpp's llama-server. No cloud
key exists anywhere in Ada.

Validated in the ada-benchmarks repo, local/REPORT.md: 100% extraction
accuracy at Qwen3-1.7B and 4B on the recall corpus. The few-shot prompt
below is the validated one — change it only behind a fresh benchmark
run (CONTRIBUTING's pre-registration rule).

Config (env):
  ADA_LOCAL_URL    OpenAI-compatible base URL.
                   default http://127.0.0.1:11434/v1  (Ollama)
                   llama-server default is http://127.0.0.1:8080/v1
  ADA_LOCAL_MODEL  model name at that endpoint. default qwen3:4b

Same .enrich()/.universal() interface as the heuristic enricher, so
auto_enricher() composes them freely.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3:4b"


def local_base_url() -> str:
    return (os.environ.get("ADA_LOCAL_URL") or DEFAULT_LOCAL_URL).rstrip("/")


def local_model_name() -> str:
    return os.environ.get("ADA_LOCAL_MODEL") or DEFAULT_LOCAL_MODEL


def local_available(timeout: float = 1.0) -> bool:
    """True when an OpenAI-compatible endpoint answers at ADA_LOCAL_URL."""
    try:
        r = httpx.get(f"{local_base_url()}/models", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


# The few-shot extraction prompt validated in ada-benchmarks local/.
_EXTRACT_SYS = """Extract a fact into compact universal-schema JSON.
Output ONLY filled roles, nothing else. Roles:
 entity{name,kind}, spatial{location,origin}, perceptual{color,size},
 temporal{age}, relational{subject,predicate,object},
 quantitative{count,magnitude,unit}.
Rules: entity.name = the main person (always). 'works as Y' ->
relational{subject,predicate:works_as,object:Y}. 'employed by Z' ->
relational{predicate:employed_by,object:Z}. 'drives V' ->
relational{predicate:drives,object:V}. weekends/hobby on H ->
relational{predicate:hobby,object:H}. favorite color X ->
perceptual{color:X}. location = where they are NOW; origin = born in/from.

Examples:
Fact: Ann lives in Boston.
{"entity":{"name":"ann","kind":"person"},"spatial":{"location":"boston"}}
Fact: Bo was born in Reno.
{"entity":{"name":"bo","kind":"person"},"spatial":{"origin":"reno"}}
Fact: Cy works as a chef.
{"entity":{"name":"cy","kind":"person"},"relational":{"subject":"cy","predicate":"works_as","object":"chef"}}
Fact: Di's favorite color is teal.
{"entity":{"name":"di","kind":"person"},"perceptual":{"color":"teal"}}"""


class LocalUniversalEnricher:
    """Local-endpoint enricher with the universal-schema interface."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self._model = model or local_model_name()
        self._url = (base_url or local_base_url()).rstrip("/")
        self._cache: dict[str, dict] = {}
        self._client = httpx.Client(timeout=30.0)

    def _gen(self, text: str) -> str:
        user = "Fact: " + text
        # Qwen3's soft switch: suppress thinking tokens on every runtime
        # without needing server-specific template flags.
        if "qwen3" in self._model.lower():
            user += " /no_think"
        r = self._client.post(
            f"{self._url}/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _EXTRACT_SYS},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 200,
            },
        )
        r.raise_for_status()
        j = r.json()
        return str(j["choices"][0]["message"]["content"] or "")

    def universal(self, text: str) -> dict[str, dict[str, str]]:
        """prose -> sanitized {layer:{role:value}} universal mapping."""
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]
        from ada.memory.thought_space import _sanitize_universal
        try:
            raw = self._gen(text)
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {}
        except Exception:
            logger.warning("local enrich failed for %r", text[:60],
                           exc_info=True)
            parsed = {}
        mapped = _sanitize_universal(parsed) if parsed else {}
        # backfill relational.subject from entity.name (the small model
        # often omits it — deterministic, not a guess)
        name = (mapped.get("entity") or {}).get("name")
        rel = mapped.get("relational")
        if name and isinstance(rel, dict) and rel.get("object") and not rel.get("subject"):
            rel["subject"] = name
        self._cache[key] = mapped
        return mapped

    def enrich(self, text: str):
        from ada.encoder.llm_enricher import _project_to_structured
        return _project_to_structured(text, self.universal(text))

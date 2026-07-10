"""
LLM renderer factory — the surface form on top of the substrate.

The substrate does the retrieval; the renderer turns recalled facts into
natural language. The model only ever sees what the substrate chose to
expose, so it can't invent facts — and the model is the LOCAL one
(Ollama / llama-server at ADA_LOCAL_URL). Nothing leaves the machine.
Without a local endpoint, the substrate's raw output is returned
directly.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# (facts_context, query) -> rendered text
Renderer = Callable[[str, str], str]

DEFAULT_SYSTEM = (
    "You are Ada's language surface. You receive recalled memories and a "
    "query. Answer the query in ONE short sentence using ONLY the recalled "
    "memories. If they don't contain the answer, say \"I don't know.\" "
    "Never invent facts."
)


def build_llm_renderer(
    model: str | None = None,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = 200,
) -> Optional[Renderer]:
    """Return a local-model renderer, or None when no endpoint answers.

    The substrate works without it — the renderer is just the surface form.
    """
    from ada.encoder.local_enricher import (local_available, local_base_url,
                                            local_model_name)

    if not local_available():
        logger.info("No local model endpoint at ADA_LOCAL_URL — answers "
                    "return substrate facts directly.")
        return None

    import httpx

    chosen = model or local_model_name()
    url = local_base_url()
    client = httpx.Client(timeout=30.0)

    def _render(context: str, query: str) -> str:
        user = (
            f"{context}\n\n"
            f"Answer this query using only the recalled memories: {query}"
        )
        if "qwen3" in chosen.lower():
            user += " /no_think"
        r = client.post(
            f"{url}/chat/completions",
            json={
                "model": chosen,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"] or "").strip()

    return _render

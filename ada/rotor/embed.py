"""
The semantic layer's embedder — a dedicated embedding model over any
OpenAI-compatible /embeddings endpoint (Ollama serves nomic-embed-text
on the same port as chat models: `ollama pull nomic-embed-text`).

Ported from the proven glyphh-apps design: chat-model embeddings are
unusable for retrieval (anisotropy swamps the signal); nomic with its
trained task prefixes works. Queries and documents live in ONE space but
enter it differently ("search_query: " / "search_document: ").

Vectors are unit-normalized so cosine similarity is a plain dot product.
Every failure path answers None and the semantic layer degrades to a
silent no-op — the rotor never blocks a proxied request on this.
"""

from __future__ import annotations

import logging
import struct

import httpx

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)

_MAX_CHARS = 2000   # ≈500 tokens — comfortably inside the serve window
_BATCH = 16


def _endpoint() -> str:
    s = get_settings()
    base = (s.embed_url or "").strip()
    if not base:
        from ada.encoder.local_enricher import local_base_url
        base = local_base_url()
    return base.rstrip("/") + "/embeddings"


def normalize(v: list[float]) -> list[float]:
    ssum = sum(x * x for x in v)
    if ssum <= 0:
        return [0.0] * len(v)
    inv = ssum ** -0.5
    return [x * inv for x in v]


def pack(v: list[float]) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def unpack(b: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def dot(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


async def embed_texts(texts: list[str], kind: str = "query",
                      fast: bool = False) -> list[list[float]] | None:
    """Embed texts (batched, prefixed, normalized). None = embedder down.
    `fast` uses a tight timeout for the request path (recall) so a slow or
    missing embedder degrades to no-injection instead of stalling the
    proxied call; the background cascade leaves it False."""
    if not texts:
        return []
    prefix = "search_query: " if kind == "query" else "search_document: "
    model = get_settings().embed_model
    timeout = 2.5 if fast else 30.0
    out: list[list[float]] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            for i in range(0, len(texts), _BATCH):
                batch = [prefix + t[:_MAX_CHARS] for t in texts[i:i + _BATCH]]
                r = await http.post(_endpoint(),
                                    json={"model": model, "input": batch})
                if r.status_code != 200:
                    return None
                rows = sorted(r.json().get("data", []),
                              key=lambda d: d.get("index", 0))
                if len(rows) != len(batch):
                    return None
                for row in rows:
                    out.append(normalize(row.get("embedding") or []))
    except Exception:
        return None
    return out


async def embed_text(text: str, kind: str = "query",
                     fast: bool = False) -> list[float] | None:
    r = await embed_texts([text], kind, fast=fast)
    return r[0] if r else None

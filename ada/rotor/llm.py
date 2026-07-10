"""
Local-model completion for rotor internals (summarize · distill ·
prime-sharpening). Whatever model the user installed and named in
ADA_LOCAL_MODEL, over the OpenAI-compatible endpoint at ADA_LOCAL_URL.
None on any failure — callers degrade to heuristics or wait.
"""

from __future__ import annotations

import httpx

from ada.encoder.local_enricher import local_base_url, local_model_name


async def complete(prompt: str, max_tokens: int = 400) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(
                f"{local_base_url()}/chat/completions",
                json={
                    "model": local_model_name(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            if r.status_code != 200:
                return None
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            # strip <think>…</think> from reasoning-style local models
            if "</think>" in text:
                text = text.rsplit("</think>", 1)[-1].strip()
            return text or None
    except Exception:
        return None

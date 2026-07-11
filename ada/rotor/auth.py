"""
Token gate for Ada's proxy endpoints — the "from Claude to Ada is our
key" boundary.

Ada already mints tokens (`ada token create` → `ada_<random>`, stored as
a sha256 hash in the Token table). This is the check side: a presented
`ada_…` token is hashed and looked up; active + unexpired passes. Bearer
and x-api-key both carry it (clients differ). No LLM, no external call —
one indexed hash lookup.

Bootstrap-friendly: if NO tokens exist yet, the gate is open, so a fresh
install (and `ada token create` itself) is reachable. The moment the
first token is minted, the endpoints require one.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select

from domains.models.db_models import Token


def _bare(raw: str) -> str:
    raw = raw.strip()
    return raw[7:].strip() if raw[:7].lower() == "bearer " else raw


async def any_tokens_exist(session_factory) -> bool:
    async with session_factory() as s:
        return (await s.execute(select(Token.id).limit(1))).first() is not None


async def validate_token(raw: str, session_factory) -> bool:
    """True if `raw` is a live Ada token (active, not expired)."""
    token = _bare(raw or "")
    if not token:
        return False
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with session_factory() as s:
        tok = (await s.execute(
            select(Token).where(Token.token_hash == token_hash,
                                Token.status == "active"))).scalar_one_or_none()
    if tok is None:
        return False
    if tok.expires_at and tok.expires_at < datetime.utcnow():
        return False
    return True


async def gate_ok(presented: str | None, session_factory) -> bool:
    """Endpoint gate: open until the first token is minted, then required."""
    if not await any_tokens_exist(session_factory):
        return True                       # bootstrap — nothing to check yet
    return await validate_token(presented or "", session_factory)

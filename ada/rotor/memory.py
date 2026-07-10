"""
Layered rotor memory + the cascade.

    prompts, one after the other, stored AS A WHOLE   → short term
    short term SUMMARIZED INTO EVENTS                 → mid term
    events DISTILLED INTO FACTS (the lattice)         → long term

Short term  — rotor_turns: verbatim turns per conversation + vectors.
Mid term    — rotor_events: local-model summaries of turn spans.
Long term   — the Ada lattice via absorb() (dedup + 7×33 enrichment).

The cascade never blocks and never loses: with no local model, turns
simply pool unsummarized (and events undistilled) until one shows up.
"""

from __future__ import annotations

import hashlib
import logging
import time

from sqlalchemy import select, update

from ada.rotor import embed as _embed
from ada.rotor import llm as _llm
from ada.rotor.primes import decompose, validate
from domains.models.db_models import RotorEvent, RotorTurn

logger = logging.getLogger(__name__)

# Cascade tuning — spans are small so summaries stay concrete.
EVENT_SPAN = 8          # turns folded into one event
EVENT_TRIGGER = 12      # unsummarized turns before folding begins
RECALL_K = 4            # semantic hits pulled per layer
MAX_TURN_CHARS = 16000  # cap a stored turn — bounds DB growth / scan cost


def conversation_id(messages: list[dict]) -> str | None:
    """Identity for 'prompts happening one after the other': the hash of
    the first user turn. Claude resends the whole history every call, so
    consecutive requests in one chat share it; a new chat gets a new one."""
    for m in messages:
        if m.get("role") == "user":
            text = _content_text(m.get("content"))
            if text:
                return hashlib.sha256(text.encode()).hexdigest()[:32]
    return None


def _has_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _content_text(content) -> str:
    """Flatten an Anthropic content field (str | block list) to text.
    Tool blocks and harness-injected <system-reminder> text are ignored —
    memory stores what the human actually said, not plumbing."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if not (isinstance(b, dict) and b.get("type") == "text"):
                continue
            t = (b.get("text") or "").strip()
            if t and not t.startswith("<system-reminder>"):
                parts.append(t)
        return "\n".join(parts).strip()
    return ""


def last_user_text(messages: list[dict]) -> str:
    """The latest genuine user prompt. Skips tool-loop continuations — a
    user message carrying tool_result blocks is the agent feeding results
    back, not a new human turn — so recall keys off what the human asked."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        if _has_tool_result(m.get("content")):
            continue
        t = _content_text(m.get("content"))
        if t:
            return t
    return ""


# ── WRITE — short term ─────────────────────────────────────────────────

async def write_turn(session_factory, conv_id: str, role: str,
                     text: str) -> int | None:
    """Store one turn as a whole. Dedup on (conv, sha) — Claude resends
    the full history every request, so most turns are already there.
    Returns the row id for a NEW turn, None for a duplicate."""
    text = text.strip()[:MAX_TURN_CHARS]
    if not text:
        return None
    sha = hashlib.sha256(text.encode()).hexdigest()
    async with session_factory() as s:
        existing = (await s.execute(
            select(RotorTurn.id).where(RotorTurn.conv_id == conv_id,
                                       RotorTurn.sha == sha))).scalar()
        if existing is not None:
            return None
        vec = await _embed.embed_text(text, kind="document")
        turn = RotorTurn(
            conv_id=conv_id, role=role, content=text, sha=sha,
            primes=decompose(text),
            vec=_embed.pack(vec) if vec else None,
            created_at=time.time(),
        )
        s.add(turn)
        await s.commit()
        await s.refresh(turn)
        return turn.id


# ── RECALL — the semantic layer ────────────────────────────────────────

async def semantic_recall(session_factory, query_vec: list[float],
                          exclude_conv: str | None, k: int = RECALL_K,
                          ) -> dict[str, list[dict]]:
    """Embedding search over turns and events (brute-force dot over
    unit vectors — the store is local and small). The live conversation
    is excluded: Claude already has it in-context."""
    hits: dict[str, list[dict]] = {"turns": [], "events": []}
    if not query_vec:
        return hits
    async with session_factory() as s:
        turns = (await s.execute(
            select(RotorTurn.content, RotorTurn.vec, RotorTurn.primes,
                   RotorTurn.created_at, RotorTurn.conv_id)
            .where(RotorTurn.vec.isnot(None))
            .order_by(RotorTurn.id.desc()).limit(2000))).all()
        events = (await s.execute(
            select(RotorEvent.summary, RotorEvent.vec, RotorEvent.primes,
                   RotorEvent.created_at, RotorEvent.conv_id)
            .where(RotorEvent.vec.isnot(None))
            .order_by(RotorEvent.id.desc()).limit(2000))).all()

    def rank(rows, text_ix: int) -> list[dict]:
        scored = []
        for row in rows:
            if exclude_conv and row.conv_id == exclude_conv:
                continue
            score = _embed.dot(query_vec, _embed.unpack(row.vec))
            scored.append({"text": row[text_ix], "score": score,
                           "primes": row.primes or [],
                           "at": row.created_at})
        scored.sort(key=lambda h: h["score"], reverse=True)
        return [h for h in scored[:k] if h["score"] > 0.35]

    hits["turns"] = rank(turns, 0)
    hits["events"] = rank(events, 0)
    return hits


# ── THE CASCADE — short → mid → long ───────────────────────────────────

async def stats(session_factory) -> dict:
    """Layered-memory vital signs for the REPL's `rotor` verb."""
    from sqlalchemy import func
    async with session_factory() as s:
        turns = (await s.execute(
            select(func.count()).select_from(RotorTurn))).scalar() or 0
        pooled = (await s.execute(
            select(func.count()).select_from(RotorTurn)
            .where(RotorTurn.summarized == 0))).scalar() or 0
        convs = (await s.execute(
            select(func.count(func.distinct(RotorTurn.conv_id))))).scalar() or 0
        events = (await s.execute(
            select(func.count()).select_from(RotorEvent))).scalar() or 0
        pending = (await s.execute(
            select(func.count()).select_from(RotorEvent)
            .where(RotorEvent.distilled == 0))).scalar() or 0
    return {"conversations": convs, "turns": turns, "turns_pooled": pooled,
            "events": events, "events_pending_distill": pending}


async def cascade(session_factory, brain) -> dict:
    """One background pass: fold pooled turns into events, distill
    pending events into lattice facts. Every step that needs the local
    model skips silently when it is down — nothing is lost, work just
    waits."""
    report = {"events": 0, "facts": 0}
    report["events"] = await _summarize_turns(session_factory)
    report["facts"] = await _distill_events(session_factory, brain)
    return report


async def _summarize_turns(session_factory) -> int:
    """Short → mid: conversations with enough pooled turns get their
    oldest EVENT_SPAN folded into one event."""
    made = 0
    async with session_factory() as s:
        convs = (await s.execute(
            select(RotorTurn.conv_id).where(RotorTurn.summarized == 0)
            .group_by(RotorTurn.conv_id))).scalars().all()
    for conv in convs:
        async with session_factory() as s:
            rows = (await s.execute(
                select(RotorTurn).where(RotorTurn.conv_id == conv,
                                        RotorTurn.summarized == 0)
                .order_by(RotorTurn.id).limit(EVENT_TRIGGER))).scalars().all()
            if len(rows) < EVENT_TRIGGER:
                continue
            span = rows[:EVENT_SPAN]
            transcript = "\n".join(f"{t.role}: {t.content[:400]}" for t in span)
            summary = await _llm.complete(
                "Summarize this conversation excerpt as ONE event: what "
                "happened, who/what was involved, any decisions or facts "
                "stated. 2-3 plain sentences, past tense, no preamble.\n\n"
                + transcript, max_tokens=200)
            if not summary:
                return made          # model down — the pool waits
            vec = await _embed.embed_text(summary, kind="document")
            primes: list[str] = []
            for t in span:
                primes.extend(t.primes or [])
            s.add(RotorEvent(
                conv_id=conv, summary=summary,
                primes=validate(primes + decompose(summary)),
                vec=_embed.pack(vec) if vec else None,
                turn_ids=[t.id for t in span],
                created_at=time.time(),
            ))
            await s.execute(
                update(RotorTurn)
                .where(RotorTurn.id.in_([t.id for t in span]))
                .values(summarized=1))
            await s.commit()
            made += 1
    return made


async def _distill_events(session_factory, brain) -> int:
    """Mid → long: extract durable, declarative facts from an event and
    absorb them into the lattice (absorb dedups + enriches into the
    7×33 schema — the existing write path, unchanged)."""
    made = 0
    async with session_factory() as s:
        events = (await s.execute(
            select(RotorEvent).where(RotorEvent.distilled == 0)
            .order_by(RotorEvent.id).limit(4))).scalars().all()
    for ev in events:
        raw = await _llm.complete(
            "Extract durable facts from this event — things worth "
            "remembering months from now (identities, relationships, "
            "preferences, decisions, dates). One short declarative "
            "sentence per line. If nothing is durable, reply NONE.\n\n"
            + ev.summary, max_tokens=250)
        if raw is None:
            return made              # model down — events wait
        facts = [ln.strip("-• \t") for ln in raw.splitlines()
                 if ln.strip() and ln.strip().upper() != "NONE"]
        store = brain.space("main")
        for fact in facts[:6]:
            try:
                stored = store.absorb(fact, speaker="rotor")
                if hasattr(stored, "__await__"):
                    stored = await stored
                if stored is not None:
                    made += 1
            except Exception:
                logger.warning("distill absorb failed", exc_info=True)
        async with session_factory() as s:
            await s.execute(update(RotorEvent)
                            .where(RotorEvent.id == ev.id)
                            .values(distilled=1))
            await s.commit()
    return made

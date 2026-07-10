"""
The Anthropic-compatible proxy — the rotor's delivery vehicle.

Point Claude Desktop (Developer → Configure Third-Party Inference →
Gateway) or any ANTHROPIC_BASE_URL client at this server and every
/v1/messages call spins the rotor on its way upstream:

    Claude ──▶ POST /v1/messages ──▶ rotor (recall · reason)
                                      └─▶ api.anthropic.com (caller's key,
                                          forwarded verbatim, never stored)
                                      ◀── response streamed back byte-for-byte

Everything else under /v1/* (count_tokens, models, …) passes through
untouched. A rotor failure downgrades to pure passthrough — the proxy
must never be the reason Claude breaks. Upstream response headers
(retry-after, request-id, anthropic-ratelimit-*, x-should-retry) are
preserved so client-side retry/backoff still works; content-encoding and
-length are dropped because httpx has already decoded the body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from ada.rotor import rotor as _rotor
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Request headers that ride through to Anthropic. The caller's credential
# is forwarded verbatim and NEVER logged or persisted. Duplicates (e.g.
# multiple anthropic-beta lines) are preserved via the raw header list.
_FORWARD = {b"x-api-key", b"authorization", b"anthropic-version",
            b"anthropic-beta", b"content-type", b"accept"}
# Response headers httpx has already acted on — forwarding them would make
# the client double-decode or see a wrong length.
_STRIP_RESP = {"content-encoding", "content-length", "transfer-encoding",
               "connection"}
_SSE_TEXT = re.compile(r'"text_delta","text":"((?:[^"\\]|\\.)*)"')
_STREAM_DONE = re.compile(rb'"type"\s*:\s*"message_stop"')


def _upstream_headers(request: Request) -> list[tuple[bytes, bytes]]:
    """Raw header list (preserves duplicate header lines)."""
    return [(k, v) for k, v in request.headers.raw if k.lower() in _FORWARD]


def _response_headers(resp: httpx.Response) -> list[tuple[str, str]]:
    return [(k, v) for k, v in resp.headers.items()
            if k.lower() not in _STRIP_RESP]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=get_settings().anthropic_upstream,
                             timeout=httpx.Timeout(600.0, connect=15.0))


def _reply_text_from_sse(raw: bytes) -> str:
    """Recover the assistant's text from a captured SSE stream — best
    effort, for the post-response WRITE only."""
    out: list[str] = []
    for part in _SSE_TEXT.findall(raw.decode("utf-8", "ignore")):
        try:
            out.append(json.loads(f'"{part}"'))
        except Exception:
            continue
    return "".join(out)


def _reply_text_from_json(payload: dict) -> str:
    return "\n".join(b.get("text", "") for b in payload.get("content", [])
                     if isinstance(b, dict) and b.get("type") == "text")


@router.post("/v1/messages")
async def messages(request: Request):
    raw_body = await request.body()
    settings = get_settings()
    app = request.app

    # Parse for the rotor; on any parse failure fall straight through with
    # the original bytes so upstream returns its own structured error.
    body: dict | None = None
    if settings.rotor_enabled:
        try:
            body = json.loads(raw_body)
        except Exception:
            body = None

    # ── the rotor: recall · reason (WRITE is backgrounded below) ──
    if body is not None:
        try:
            injection = await _rotor.spin(app.state.brain,
                                          _session_factory(), body)
            if injection:
                body = _rotor.inject(body, injection)
            _background(app, _rotor.write_prompt(_session_factory(), body))
        except Exception:
            logger.warning("rotor failed — passing through", exc_info=True)

    out_bytes = raw_body if body is None else json.dumps(body).encode()
    headers = _upstream_headers(request)
    params = dict(request.query_params)
    client = _client()

    if body is not None and body.get("stream"):
        return await _stream(client, body, out_bytes, headers, params, app)

    try:
        resp = await client.post("/v1/messages", content=out_bytes,
                                 headers=headers, params=params)
    except Exception:
        await client.aclose()
        raise
    await client.aclose()
    if body is not None and resp.status_code == 200:
        try:
            reply = _reply_text_from_json(resp.json())
            _background(app, _rotor.absorb_reply(_session_factory(), body, reply))
        except Exception:
            pass
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=dict(_response_headers(resp)),
                    media_type=resp.headers.get("content-type"))


async def _stream(client: httpx.AsyncClient, body: dict, out_bytes: bytes,
                  headers, params, app) -> StreamingResponse:
    req = client.build_request("POST", "/v1/messages", content=out_bytes,
                               headers=headers, params=params)
    try:
        upstream = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    captured = bytearray()

    async def pump():
        try:
            async for chunk in upstream.aiter_bytes():
                if len(captured) < 2_000_000:
                    captured.extend(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            # Only absorb a reply the stream actually FINISHED delivering —
            # a Stop/disconnect/error mid-stream must not persist a
            # truncated turn as if it were complete.
            if (upstream.status_code == 200
                    and _STREAM_DONE.search(bytes(captured))):
                reply = _reply_text_from_sse(bytes(captured))
                if reply:
                    _background(app, _rotor.absorb_reply(
                        _session_factory(), body, reply))

    return StreamingResponse(
        pump(), status_code=upstream.status_code,
        headers=dict(_response_headers(upstream)),
        media_type=upstream.headers.get("content-type", "text/event-stream"))


@router.api_route("/v1/{path:path}",
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str):
    """Everything else under /v1 rides through untouched — count_tokens,
    models discovery, whatever Anthropic adds next."""
    client = _client()
    try:
        resp = await client.request(
            request.method, f"/v1/{path}",
            content=await request.body(),
            headers=_upstream_headers(request),
            params=dict(request.query_params),
        )
    finally:
        await client.aclose()
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=dict(_response_headers(resp)),
                    media_type=resp.headers.get("content-type"))


def _session_factory():
    from infrastructure.database import async_session_maker
    return async_session_maker


def _background(app, coro) -> None:
    """Fire-and-forget on the running loop; tracked so tests can await."""
    task = asyncio.get_running_loop().create_task(coro)
    tasks = getattr(app.state, "rotor_tasks", None)
    if tasks is None:
        tasks = app.state.rotor_tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)

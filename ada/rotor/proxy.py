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
# Non-credential headers forwarded verbatim. The credential is NOT
# forwarded blindly — Ada is a router and attaches its own stored
# provider key (see _upstream_headers).
_FORWARD = {b"anthropic-version", b"anthropic-beta", b"content-type", b"accept"}
# Response headers httpx has already acted on — forwarding them would make
# the client double-decode or see a wrong length.
_STRIP_RESP = {"content-encoding", "content-length", "transfer-encoding",
               "connection"}
_SSE_TEXT = re.compile(r'"text_delta","text":"((?:[^"\\]|\\.)*)"')
_STREAM_DONE = re.compile(rb'"type"\s*:\s*"message_stop"')


def _caller_credential(request: Request) -> bytes | None:
    """The credential the caller presented (x-api-key or Bearer). This is
    the Ada token — validated by the gate, and used here only as a
    fallback provider credential when Ada holds no provider key."""
    xkey = bearer = None
    for k, v in request.headers.raw:
        lk = k.lower()
        if lk == b"x-api-key":
            xkey = v
        elif lk == b"authorization" and v[:7].lower() == b"bearer ":
            bearer = v[7:].strip()
    return xkey or bearer


def _upstream_headers(request: Request) -> list[tuple[bytes, bytes]]:
    """Ada is a router. The caller authenticates to Ada with an Ada token
    (checked by the gate, not here); this attaches the USER's own stored
    PROVIDER key for the upstream — Anthropic wants `x-api-key`. If Ada
    holds no provider key, it falls back to forwarding the caller's own
    credential (so a user who put their provider key directly in the
    client still works). Non-credential headers pass through verbatim;
    no credential is ever logged."""
    s = get_settings()
    out = [(k, v) for k, v in request.headers.raw if k.lower() in _FORWARD]
    to_anthropic = "anthropic.com" in s.anthropic_upstream
    if to_anthropic and s.anthropic_key:
        out.append((b"x-api-key", s.anthropic_key.encode()))     # router: Ada's stored key
        return out
    cred = _caller_credential(request)                            # fallback: caller's own
    if cred:
        out.append((b"x-api-key", cred) if to_anthropic
                   else (b"authorization", b"Bearer " + cred))
    return out


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


async def _gate(request: Request) -> Response | None:
    """The token gate: the caller's Ada token must be live (or no tokens
    exist yet — bootstrap). Returns a 401 Response to short-circuit, or
    None to proceed. Ada's OWN boundary, distinct from the provider key
    she attaches upstream."""
    from ada.rotor.auth import gate_ok
    cred = _caller_credential(request)
    try:
        ok = await gate_ok(cred.decode("latin-1") if cred else None,
                           _session_factory())
    except Exception:                     # can't verify → fail CLOSED
        logger.warning("token gate errored — denying", exc_info=True)
        ok = False
    if ok:
        return None
    return Response(
        content=json.dumps({"type": "error", "error": {
            "type": "authentication_error",
            "message": "Ada: invalid or missing token. Mint one with "
                       "`ada token create` and set it as the gateway key."}}).encode(),
        status_code=401, media_type="application/json",
        headers={"www-authenticate": "Bearer"})


@router.post("/v1/messages")
async def messages(request: Request):
    denied = await _gate(request)
    if denied is not None:
        return denied
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
    denied = await _gate(request)
    if denied is not None:
        return denied
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

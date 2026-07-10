"""
Ada REPL — the single interface: a plain command line over the substrate's
/mcp door. No mouse, no full-screen panels, no login — commands in, answers
out, same as any REPL you already know.

No auth: machine access to this process is the trust boundary (the same
posture as Ollama, a local Postgres, any other local-first tool). If you
want an attributable credential anyway — for a non-local deployment, or
just to track who minted what — `connect`/`token` still mint real tokens
via `ada token create`; nothing on /mcp checks one, but nothing stops you
from wiring your own reverse proxy to require it.
"""

from __future__ import annotations

import json
import os
import re

from rich.box import SQUARE
from rich.console import Console
from rich.table import Table

# ── palette — one purple, neutral grays, white. 256-safe (see
#    tests/test_palette_law.py): grays are r=g=b so they can't shift hue
#    under 256-color quantization; #af5fff is an exact xterm-256 entry. ──
TEXT = "#ffffff"
TEXT_SOFT = "#dadada"
TEXT_MUTED = "#a8a8a8"
TEXT_FAINT = "#6c6c6c"
SIGNAL = "#af5fff"
PRIMARY = "#af5fff"
ACCENT = "#af5fff"
CALM = "#dadada"
GREEN = "#af5fff"

console = Console(highlight=False)


class ServerBackend:
    """MCP client against a running Ada server — the only principal now.
    Every call is a tools/call against /mcp; the REPL, Claude, and every
    other MCP client share ONE memory, no credential required."""

    label = "server"

    def __init__(self, url: str):
        import httpx
        self.url = url
        self._http = httpx.Client(timeout=60.0)
        self._headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        r = self._rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "ada-repl", "version": "0.8"},
        }, rpc_id=1)
        sid = r.headers.get("mcp-session-id")
        if sid:
            self._headers["mcp-session-id"] = sid
        self._notify("notifications/initialized")

    def _rpc(self, method: str, params: dict, rpc_id: int = 2):
        body = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
        return self._http.post(self.url, json=body, headers=self._headers)

    def _notify(self, method: str) -> None:
        self._http.post(self.url, json={"jsonrpc": "2.0", "method": method},
                        headers=self._headers)

    def call(self, tool: str, args: dict) -> dict:
        resp = self._rpc("tools/call", {"name": tool, "arguments": args})
        text = resp.text
        m = re.search(r"^data: (.*)$", text, re.M)
        payload = json.loads(m.group(1)) if m else json.loads(text)
        if "error" in payload:
            return {"error": payload["error"].get("message", str(payload["error"]))}
        content = payload["result"]["content"][0]["text"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"answer": content}


def _parse_conditions(body: str) -> dict | None:
    """'l.r=v l.r=v2' → conditions dict; repeated slots become lists."""
    conditions: dict = {}
    for part in body.split():
        if "=" not in part or "." not in part.split("=", 1)[0]:
            return None
        lr, value = part.split("=", 1)
        if lr in conditions:
            prev = conditions[lr]
            conditions[lr] = (prev if isinstance(prev, list) else [prev]) + [value]
        else:
            conditions[lr] = value
    return conditions or None


_HELP = """[b]memory[/b]
  tell <text>            absorb a fact (durable, shared)
  tell key=K <text>      absorb as the next version of K
  ask <question>         targeted retrieval (refuses 'I don't know')
  think <input>          broad associative recall
  recall <query>         lexical search, top matches + scores
[b]query[/b]
  count <l.r>=<v> ...    count entities matching ALL conditions
  find <l.r>=<v> ...     entities matching ALL conditions
  top <layer.role> [pred]  distribution
  list [filter]          keyed facts (current belief)
  fact <key>             anatomy of a fact: slots, chain, provenance
  history <key>          version chain
  similar <entity> [k]   nearest profiles by exact jaccard
  drift <entity> [days]  what moved in the window
[b]curate[/b]
  merges                 deterministic alias proposals
  merge <a> => <b>       merge a into b (run again with `confirm`)
  amend <key> <text>     fix a fact in place (no new version)
  consolidate [dry]      re-enrich · resolve identity · dedup
  archive <l.r>=<v> …    soft-remove a cohort, reversible (+ `confirm`)
  forget <key>           erase a key's chain (+ `confirm`)
  forget entity <name>   erase every fact of an entity (+ `confirm`)
  forget all <space>     wipe the whole space (type the space name)
[b]session[/b]
  space [id]             show / switch space
  me [name]              show / set identity (resolves 'I/my')
  connect [claude]       mint a token + print MCP configs
  token                  list tokens
  token revoke|delete <id>   manage a token
  stats                  substrate vital signs
  rotor                  layered-memory vital signs (turns · events · facts)
  clear                  clear the scrollback
  help                   this help
  exit                   leave the REPL

Bare text with no verb is treated as [b]ask[/b]."""


class Session:
    """Live REPL state: space, identity, the connected backend."""

    def __init__(self, url: str):
        self.url = url
        self.backend: ServerBackend | None = None
        self.space = "main"
        self.me: str | None = None

    def call(self, tool: str, args: dict) -> dict:
        if not self.backend:
            return {"error": "not connected to a server"}
        a = dict(args)
        if self.space != "main" and "space" not in a:
            a["space"] = self.space
        if tool == "tell" and self.me and "speaker" not in a:
            a["speaker"] = self.me
        try:
            return self.backend.call(tool, a)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


def run_repl(url: str) -> None:
    """The `ada` experience: connect, drop into a plain REPL over the
    substrate's /mcp door. No login — machine access is the trust
    boundary. Same verbs as always."""
    sess = Session(url)
    try:
        sess.backend = ServerBackend(url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [{SIGNAL}]could not connect: {exc}[/]")
        return
    console.print(f"  [{PRIMARY}]ada[/] [{TEXT_FAINT}]· one terminal for "
                  f"the whole substrate. type 'help'.[/]")
    while True:
        try:
            raw = input("  ada› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            break
        _dispatch(sess, raw)


def _grid(columns: list[str], rows: list[tuple], justify: list[str] | None = None,
          widths: list[int | None] | None = None) -> Table:
    t = Table(box=SQUARE, show_lines=False, expand=False, pad_edge=False,
              border_style=TEXT_FAINT, header_style=f"bold {TEXT}")
    for i, c in enumerate(columns):
        t.add_column(c, justify=(justify[i] if justify else "left"),
                     width=(widths[i] if widths else None), overflow="fold")
    for r in rows:
        t.add_row(*[str(c) for c in r])
    return t


def _dispatch(sess: Session, raw: str) -> None:
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()

    def out(line: str) -> None:
        console.print(f"  {line}")

    def err(r: dict) -> bool:
        if "error" in r:
            out(f"[{SIGNAL}]{r['error']}[/]")
            return True
        return False

    def do_ask(question: str) -> None:
        r = sess.call("ask", {"question": question})
        if err(r):
            return
        if r.get("refused"):
            out(f"[{TEXT_FAINT}]{r.get('answer') or 'I dont know.'}[/]")
            return
        conf = r.get("confidence")
        tail = f"  [{TEXT_FAINT}]{conf}[/]" if conf is not None else ""
        out(f"[{CALM}]{r.get('answer', '')}[/]{tail}")

    if verb in ("help", "?"):
        console.print(_HELP)
    elif verb == "clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif verb == "space":
        if rest:
            sess.space = rest
            out(f"[{TEXT_FAINT}]switched to space:[/] {rest}")
        else:
            out(f"[{TEXT_FAINT}]current space:[/] {sess.space}")
    elif verb == "me":
        if rest:
            sess.me = rest.lower()
            out(f"[{TEXT_FAINT}]you are:[/] {rest.lower()}")
        else:
            out(f"[{TEXT_FAINT}]identity:[/] {sess.me or '(unset)'}")
    elif verb == "stats":
        r = sess.call("stats", {})
        if not err(r):
            for k, v in r.items():
                out(f"[{TEXT_FAINT}]{k:<18}[/] {v}")
    elif verb == "rotor":
        r = sess.call("rotor_stats", {})
        if not err(r):
            for k, v in r.items():
                out(f"[{TEXT_FAINT}]{k:<26}[/] {v}")
    elif verb == "tell":
        args = {"text": rest}
        if rest.startswith("key="):
            key, _, body = rest[4:].partition(" ")
            args = {"text": body.strip(), "key": key}
        r = sess.call("tell", args)
        if not err(r):
            if r.get("told"):
                v = r.get("version")
                out(f"[{GREEN}]absorbed[/]" + (f" [{TEXT_FAINT}]v{v}[/]" if v else ""))
            else:
                out(f"[{TEXT_FAINT}]not stored ({r.get('reason', 'duplicate')})[/]")
    elif verb == "ask":
        do_ask(rest)
    elif verb == "think":
        r = sess.call("think", {"input": rest, "top_k": 8})
        if not err(r):
            acts = r.get("activated", [])
            if not acts:
                out(f"[{TEXT_FAINT}](nothing surfaced)[/]")
            for a in acts:
                out(f"[{TEXT_FAINT}][{a.get('similarity', 0):+.2f}][/] {a['content']}")
    elif verb == "recall":
        r = sess.call("recall", {"query": rest, "top_k": 8})
        if not err(r):
            res = r.get("results", [])
            if not res:
                out(f"[{TEXT_FAINT}](no matches)[/]")
            for x in res:
                tag = f"  [{TEXT_FAINT}][{x['key']}][/]" if x.get("key") else ""
                out(f"[{TEXT_FAINT}][{x['similarity']:+.2f}][/] {x['content']}{tag}")
    elif verb in ("count", "find"):
        conditions = _parse_conditions(rest)
        if not conditions:
            out(f"[{TEXT_FAINT}]usage: {verb} layer.role=value …[/]")
        else:
            op = "count" if verb == "count" else "who"
            r = sess.call("query", {"op": op, "conditions": conditions})
            if not err(r):
                out(f"[{CALM}]{r.get('answer')}[/]")
    elif verb == "top":
        parts = rest.split()
        if not parts or "." not in parts[0]:
            out(f"[{TEXT_FAINT}]usage: top layer.role [predicate][/]")
        else:
            q = {"op": "top", "slot": parts[0]}
            if len(parts) > 1:
                q["predicate_contains"] = parts[1]
            r = sess.call("query", q)
            if not err(r):
                out(f"[{CALM}]{r.get('answer')}[/]")
    elif verb in ("list", "keys"):
        r = sess.call("browse", {"limit": 200, **({"q": rest} if rest else {})})
        if not err(r):
            facts = r.get("facts", [])
            if not facts:
                out(f"[{TEXT_FAINT}](no thoughts)[/]")
            else:
                rows = [(str(f.get("thought_id", ""))[:12],
                         str(f.get("key")) if f.get("key") else "—",
                         f"v{f['version']}", str(f["content"])[:70])
                        for f in facts]
                console.print(_grid(["id", "key", "ver", "thought"], rows,
                                    justify=["left", "left", "center", "left"],
                                    widths=[14, 22, 5, None]))
    elif verb == "fact":
        r = sess.call("inspect", {"key": rest})
        if not err(r):
            out(f"[{CALM}]{r.get('key') or r.get('thought_id')}[/]  {r.get('content')}")
            for sl in r.get("slots", []):
                out(f"  [{TEXT_FAINT}]{sl['layer']}.{sl['role']:<14}[/] {sl['value']}")
    elif verb == "history":
        r = sess.call("history", {"key": rest})
        if not err(r):
            if not r.get("versions"):
                out(f"[{TEXT_FAINT}]no history for {rest!r}[/]")
            for item in r.get("chain", []):
                out(f"[{TEXT_FAINT}]v{item['version']}:[/] {item['content']}")
    elif verb == "similar":
        parts = rest.split()
        k = 5
        if len(parts) > 1 and parts[-1].isdigit():
            k = int(parts[-1])
            parts = parts[:-1]
        r = sess.call("similar", {"entity": " ".join(parts), "k": k})
        if not err(r):
            sim = r.get("similar", [])
            if not sim:
                out(f"[{TEXT_FAINT}]{' '.join(parts)} shares no slot value[/]")
            for e in sim:
                out(f"[{CALM}]{e['name']}[/]  [{TEXT_FAINT}]jaccard {e['similarity']} · "
                    f"{' · '.join(e['shared'][:4])}[/]")
    elif verb == "drift":
        parts = rest.split()
        days = 30
        if len(parts) > 1 and parts[-1].isdigit():
            days = int(parts[-1])
            parts = parts[:-1]
        r = sess.call("drift", {"entity": " ".join(parts), "window_days": days})
        if not err(r):
            out(f"[{CALM}]drift {r.get('drift')}[/]  [{TEXT_FAINT}]"
                f"+{len(r.get('added', []))} −{len(r.get('dropped', []))} · "
                f"{len(r.get('churned_keys', []))} keys churned[/]")
            for d in r.get("added", [])[:6]:
                out(f"[{CALM}]+ {d}[/]")
            for d in r.get("dropped", [])[:6]:
                out(f"[{SIGNAL}]− {d}[/]")
    elif verb == "merges":
        r = sess.call("merge_candidates", {})
        if not err(r):
            cands = r.get("candidates", [])
            if not cands:
                out(f"[{TEXT_FAINT}]no alias proposals[/]")
            for c in cands:
                out(f"[{CALM}]{c['a']}[/] ≈ [{CALM}]{c['b']}[/]  "
                    f"[{ACCENT}]overlap {c['score']}[/]")
                if c["shared"]:
                    out(f"  [{TEXT_FAINT}]shared: {' · '.join(c['shared'][:4])}[/]")
                if c["conflicts"]:
                    out(f"  [{SIGNAL}]conflicts: {' · '.join(c['conflicts'])}[/]")
                out(f"  [{TEXT_FAINT}]merge {c['a']} => {c['b']}[/]")
    elif verb == "merge":
        body = rest
        confirm = body.endswith(" confirm")
        if confirm:
            body = body[:-len(" confirm")].strip()
        if "=>" in body:
            src, tgt = [x.strip().lower() for x in body.split("=>", 1)]
        else:
            p = body.split()
            src, tgt = (p[0].lower(), p[1].lower()) if len(p) == 2 else ("", "")
        if not src or not tgt:
            out(f"[{TEXT_FAINT}]usage: merge <source> => <target> [confirm][/]")
        else:
            r = sess.call("merge", {"source": src, "target": tgt, "dry_run": not confirm})
            if not err(r):
                if confirm:
                    out(f"[{TEXT_FAINT}]merged {src} → {tgt} · "
                        f"{r.get('facts')} facts re-pointed[/]")
                else:
                    out(f"[{SIGNAL}]would merge {src} → {tgt} · "
                        f"{r.get('facts')} facts[/]  [{TEXT_FAINT}]"
                        f"run: merge {src} => {tgt} confirm[/]")
    elif verb == "consolidate":
        a = {"dry_run": rest.strip() == "dry"}
        if sess.me:
            a["me"] = sess.me
        r = sess.call("consolidate", a)
        if not err(r):
            out(f"[{TEXT_FAINT}]{'preview' if a['dry_run'] else 'consolidated'} · "
                f"{r.get('scanned')} scanned · re-enriched {r.get('re_enriched')} · "
                f"identity {r.get('identity_resolved')} · "
                f"dups {len(r.get('duplicates_archived', []))}[/]")
            for d in r.get("duplicates_archived", []):
                out(f"  [{TEXT_FAINT}]archived “{d['archived']}” ⇒ kept “{d['kept']}”[/]")
    elif verb == "amend":
        parts = rest.split(None, 1)
        if parts and parts[0] == "id" and len(parts) == 2:
            idtext = parts[1].split(None, 1)
            req = ({"thought_id": idtext[0], "text": idtext[1]}
                   if len(idtext) == 2 else None)
        elif len(parts) == 2:
            req = {"key": parts[0].lower(), "text": parts[1]}
        else:
            req = None
        if not req:
            out(f"[{TEXT_FAINT}]usage: amend <key> <text> · amend id <tid> <text>[/]")
        else:
            r = sess.call("amend", req)
            if not err(r):
                out(f"[{TEXT_FAINT}]amended in place · v{r.get('version')} "
                    f"(no new version)[/]")
    elif verb == "archive":
        body = rest
        confirm = body.endswith(" confirm")
        if confirm:
            body = body[:-len(" confirm")].strip()
        conditions = _parse_conditions(body)
        if not conditions:
            out(f"[{TEXT_FAINT}]usage: archive l.r=v [l.r=v ...] [confirm][/]")
        else:
            r = sess.call("archive", {"conditions": conditions, "dry_run": not confirm})
            if not err(r):
                word = "archived" if confirm else "would archive"
                out(f"[{SIGNAL if not confirm else TEXT_FAINT}]{word} "
                    f"{r.get('facts')} facts of {r.get('entities_matched')} "
                    f"entities[/]" + ("" if confirm or not r.get('facts')
                    else f"  [{TEXT_FAINT}]run: archive {body} confirm[/]"))
    elif verb == "forget":
        _forget(sess, rest, out, err)
    elif verb == "connect":
        _connect_cmd(sess, rest, out, err)
    elif verb == "token":
        sub, _, arg = rest.partition(" ")
        if sub == "revoke" and arg:
            r = sess.call("token_revoke", {"token": arg.strip()})
            if not err(r):
                out(f"[{TEXT_FAINT}]revoked: {', '.join(r.get('revoked', []))}[/]")
        elif sub == "delete" and arg:
            r = sess.call("token_delete", {"token": arg.strip()})
            if not err(r):
                out(f"[{TEXT_FAINT}]deleted: {', '.join(r.get('deleted', []))}[/]")
        else:
            r = sess.call("token_list", {})
            if not err(r):
                toks = r.get("tokens", [])
                if not toks:
                    out(f"[{TEXT_FAINT}]no tokens — mint one with 'connect'[/]")
                else:
                    rows = [(t["prefix"], t["name"], ",".join(t.get("permissions", [])),
                             t["status"], t.get("expires_at") or "never")
                            for t in toks]
                    console.print(_grid(["prefix", "name", "perms", "status", "expires"], rows))
    else:
        # Bare text or an unrecognised verb → targeted ask over the line.
        do_ask(raw)


def _forget(sess: Session, rest: str, out, err) -> None:
    """forget <key> | entity <name> | id <tid> | all [space], two-step."""
    body = rest.strip()
    confirm = body == "confirm" or body.endswith(" confirm")
    if confirm:
        body = "" if body == "confirm" else body[:-len(" confirm")].rstrip()
    if body == "all" or body.startswith("all"):
        arg = body[len("all"):].strip().lower()
        a = {"dry_run": not arg}
        if arg:
            a["confirm"] = arg
        r = sess.call("forget_all", a)
        if err(r):
            return
        sp = r.get("space", sess.space)
        if r.get("dry_run"):
            if not r.get("facts"):
                out(f"[{TEXT_FAINT}]space {sp} is already empty[/]")
            else:
                out(f"[{SIGNAL}]would erase ALL {r['facts']} facts in {sp}[/]"
                    f"  [{TEXT_FAINT}]run: forget all {sp}[/]")
        else:
            out(f"[{SIGNAL}]erased {r['facts']} facts — {sp} is empty[/]")
        return
    parts = body.split(None, 1)
    if len(parts) == 2 and parts[0] in ("entity", "id"):
        scope = ({"entity": parts[1].strip().lower()} if parts[0] == "entity"
                 else {"thought_id": parts[1].strip()})
        label = f"{parts[0]} {parts[1].strip()}"
    elif body:
        scope = {"key": body.lower()}
        label = f"key {body.lower()}"
    else:
        out(f"[{TEXT_FAINT}]usage: forget <key> | entity <name> | id <tid> | all   [confirm][/]")
        return
    r = sess.call("forget", {**scope, "dry_run": not confirm})
    if err(r):
        return
    n = r.get("facts", 0)
    s = "" if n == 1 else "s"
    if confirm:
        out(f"[{SIGNAL}]erased {n} fact{s} — {label}[/]")
    elif not n:
        out(f"[{TEXT_FAINT}]nothing matches {label}[/]")
    else:
        out(f"[{SIGNAL}]would erase {n} fact{s} — {label}[/]"
            f"  [{TEXT_FAINT}]run: forget {body} confirm[/]")


def _connect_cmd(sess: Session, rest: str, out, err) -> None:
    """Mint a token (optional, unenforced) + print MCP configs."""
    target = (rest.split() or ["show"])[0]
    name = "claude-code" if target == "claude" else "mcp-client"
    tok = sess.call("create_token", {"name": name, "permissions": ["read", "write"]})
    if err(tok):
        return
    t = tok.get("token", "")
    url = sess.url
    out(f"[{GREEN}]token minted — shown once:[/] [{CALM}]{t}[/]")
    out(f"[{TEXT_FAINT}]Claude Code:[/] claude mcp add ada --transport http {url}")
    out(f"[{TEXT_FAINT}]Claude Desktop / Cursor:[/] "
        f'{{"mcpServers":{{"ada":{{"transport":"http","url":"{url}"}}}}}}')
    out(f"[{TEXT_FAINT}]ChatGPT · Gemini · any MCP client →[/] endpoint {url}")
    out(f"[{TEXT_FAINT}]no credential is required to connect — the token above is "
        f"for your own bookkeeping / a future reverse proxy[/]")
    out(f"[{TEXT_FAINT}]revoke: token revoke {t[:12]}[/]")

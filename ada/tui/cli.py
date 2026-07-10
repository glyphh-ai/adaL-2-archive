"""
Ada CLI — interactive shell + headless server.

Usage:
    ada              Launch interactive shell
    ada serve        Start headless (MCP endpoint only)
    ada serve -p 9000  Custom port
    ada setup        Show how to connect to Claude/ChatGPT
    ada setup claude Auto-configure Claude Code
"""

import sys
import threading
import time

# ANSI colors
# White + the web's purple (glyphh.ai --accent ramp), xterm-256 quantized.
P = "\033[38;5;135m"   # the one purple (#af5fff, 256-exact)
B = "\033[38;5;231m"   # white — values read white; purple is for status
G = "\033[38;5;135m"   # status accents — the one purple
D = "\033[38;5;244m"   # dim
W = "\033[38;5;231m"   # true white (231, not the theme-tinted 37)
R = "\033[0m"          # reset
BOLD = "\033[1m"
PINK = "\033[38;5;135m"  # errors — the one purple

# Braille rendering of the full Glyphh logo (mascot + wordmark), from
# glyphh-logo-white.png — white outline art quantized to 2x4 braille cells.
_LOGO = (
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⢸⡏⠳⣄⡀⠀⣀⣠⠴⠚⠉⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠈⢧⣀⡤⠟⠛⠉⠀⠀⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⢛⣛⢳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⢛⣛⢳⠀⠀⠀⠀⠀⢰⠛⣛⡛⡆⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⢠⠞⠁⢀⣤⠴⠖⠒⠲⣄⠈⢳⡀⠀⠀⠀⣀⡤⠤⠶⠶⠦⠤⣼⢸⣿⢸⠤⠤⣤⠀⣠⠤⠤⣤⠤⠤⠶⠶⠤⢤⣀⡇⢸⣿⠸⠤⠶⠦⠤⣄⢸⠀⣿⡇⠧⠴⠦⠤⣄⠀⠀⠀\n"
    "⠀⠀⡏⠀⣰⠋⠀⠀⢠⣶⡆⢸⡆⠀⣧⠀⢀⡞⣡⣶⠿⣛⣛⢿⣷⢸⢸⣿⢸⢸⣷⢸⠀⣿⢸⣿⠀⢸⣿⢛⣛⠻⢷⣎⠃⣸⣿⠾⢛⣛⠻⣷⣮⢻⠀⣿⡷⠟⣛⠻⣷⣮⢳⡀⠀\n"
    "⠀⠀⣇⠀⠘⠲⢤⣄⣀⣈⣠⠞⠁⢀⡟⠀⢸⠱⣿⡇⣏⠉⣹⢸⣿⢸⢸⣿⢸⢸⣿⢸⠀⣿⢸⣿⠀⢸⣿⢸⡉⣹⠎⣿⡆⣿⣿⢸⠉⠉⢻⢸⣿⢸⠀⣿⡇⡏⠉⢻⢸⣿⢸⡇⠀\n"
    "⠀⠀⠈⢹⡆⠀⣀⡀⠉⠉⠀⢤⠖⠋⠀⠀⠈⢧⡻⢿⣮⣭⣥⢾⣿⢸⢸⣿⢸⡘⢿⣮⣭⣥⢾⣿⠀⢸⣿⣬⣭⣥⣾⠟⡄⢻⣿⢸⠀⠀⢸⢸⣿⢸⠀⣿⡇⡇⠀⢸⢸⣿⢸⡇⠀\n"
    "⠀⠀⠀⡾⠀⣸⣿⣿⣿⣶⡀⠈⠷⢚⣻⠀⠀⣼⢛⣓⠶⠖⣂⣼⡟⣸⠶⠶⠚⡟⣓⡲⠖⣂⣼⡟⡀⢸⣿⢰⠤⠶⠖⠋⠓⠶⠶⠚⠀⠀⠘⠒⠶⠚⠓⠶⠶⠃⠀⠘⠶⠶⠞⠃⠀\n"
    "⠀⠀⠰⣧⣀⣉⣩⣭⣭⣭⣧⣤⣼⠏⠀⠀⠀⠳⢬⣝⣛⣛⣫⡭⠞⠁⠀⠀⠘⠧⣭⣛⣛⣫⡭⠞⠷⠬⠭⠼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
)


def _print_banner(version: str = "0.8.0") -> None:
    """The full logo outline in white braille, with \"Ada <version>\"
    right-aligned under it."""
    white = f"{BOLD}\033[38;5;231m"
    lines = _LOGO.split("\n")
    w = max(len(ln) for ln in lines)
    for ln in lines:
        print(f"  {white}{ln}{R}")
    ver = f"Ada v{version}"
    print(f"  {D}{ver.rjust(w)}{R}")
    print()


def _print_config(host: str, port: int) -> None:
    from ada.encoder.local_enricher import (local_available, local_base_url,
                                            local_model_name)
    display_host = "localhost" if host == "0.0.0.0" else host

    print(f"  {D}MCP{R}    http://{display_host}:{port}/mcp")
    print(f"  {D}Tools{R}  think · ask · tell · query · history · stats")
    print(f"  {D}Local{R}  {local_model_name()} {D}@{R} {local_base_url()}")
    if local_available():
        print(f"  {D}Model{R}  {G}●{R} endpoint up")
    else:
        print(f"  {D}Model{R}  {PINK}○{R} endpoint down {D}(extraction falls back to heuristics){R}")
    print()


def _local_model_notice() -> None:
    """Warn (never block) when no local model endpoint answers."""
    from ada.encoder.local_enricher import local_available, local_base_url
    if local_available():
        return
    print(f"\n  {PINK}No local model at {local_base_url()}.{R}")
    print(f"  {D}Extraction falls back to regex heuristics. For full extraction:{R}")
    print(f"  {W}ollama pull qwen3:4b{R} {D}(Ollama serves on :11434 automatically){R}")
    print(f"  {D}…or any llama-server / OpenAI-compatible endpoint via{R} ADA_LOCAL_URL\n")


def _token_cmd(args: list[str]) -> None:
    """Local, bootstrap token management (mint/list/revoke).

    This is the one MCP-can't-do-itself piece: the first token has to be minted
    locally with direct DB access, because minting over MCP needs a token.
    Raw tokens are printed once — store them now.

    A freshly minted token is printed once, then wiped from the terminal after
    a countdown (default 30s) so the secret doesn't linger on an admin's screen.
    The wipe is TTY-only — piped/redirected output stays scriptable.

    Usage:
        ada token create [--name N] [--expires-days D] [--permissions read,write]
                         [--model-id M] [--clear-after SECONDS | --no-clear]
        ada token list
        ada token revoke <id-or-prefix>
    """
    import asyncio

    sub = args[0] if args else "list"

    def _opt(flag: str, default=None):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
        return default

    async def _run() -> None:
        import hashlib
        import secrets
        from datetime import datetime, timedelta

        from infrastructure.database import async_session_maker, init_db
        from domains.models.db_models import Token
        from sqlalchemy import select

        await init_db()  # ensure schema exists (create_all on SQLite / migrate on PG)
        org_id = "ada"

        if sub == "create":
            name = _opt("--name", "cli-token")
            model_id = _opt("--model-id")
            perms_raw = _opt("--permissions", "read,write")
            permissions = [p.strip() for p in perms_raw.split(",") if p.strip()]
            days = _opt("--expires-days")
            expires_at = (
                datetime.utcnow() + timedelta(days=int(days)) if days else None
            )
            raw = f"ada_{secrets.token_urlsafe(32)}"
            token_hash = hashlib.sha256(raw.encode()).hexdigest()
            async with async_session_maker() as session:
                tok = Token(
                    name=name,
                    token_hash=token_hash,
                    token_prefix=raw[:12],
                    org_id=org_id,
                    model_id=model_id,
                    permissions=permissions,
                    expires_at=expires_at,
                )
                session.add(tok)
                await session.commit()
                await session.refresh(tok)
                tok_id = str(tok.id)
            clear_after = (
                0 if "--no-clear" in args else int(_opt("--clear-after", "30") or 30)
            )
            block = [
                "",
                f"  {G}Token created.{R} {D}Store it now — it is not shown again.{R}",
                "",
                f"  {BOLD}{raw}{R}",
                "",
                f"  {D}id{R}          {tok_id}",
                f"  {D}name{R}        {name}",
                f"  {D}permissions{R} {', '.join(permissions)}",
                f"  {D}expires{R}     {expires_at.isoformat() if expires_at else 'never'}",
                "",
                f"  {D}Use it:{R} Authorization: Bearer {raw[:12]}…",
                "",
            ]
            for ln in block:
                print(ln)

            # Wipe the secret from the terminal after a countdown (TTY only;
            # piped/redirected output just prints and exits, so it stays
            # scriptable). Ctrl-C keeps it on screen.
            if clear_after > 0 and sys.stdout.isatty():
                try:
                    for rem in range(clear_after, 0, -1):
                        sys.stdout.write(
                            f"\r  {D}clearing in {rem:2d}s — copy it now "
                            f"(Ctrl-C to keep on screen){R}"
                        )
                        sys.stdout.flush()
                        time.sleep(1)
                    sys.stdout.write("\r\033[2K")  # erase countdown line
                    for _ in range(len(block)):
                        sys.stdout.write("\033[1A\033[2K")  # up + erase each block line
                    sys.stdout.write(f"  {D}(token cleared from screen){R}\n")
                    sys.stdout.flush()
                except KeyboardInterrupt:
                    sys.stdout.write("\r\033[2K")
                    sys.stdout.write(f"  {D}(kept on screen){R}\n")
                    sys.stdout.flush()

        elif sub == "revoke":
            ident = args[1] if len(args) > 1 else None
            if not ident:
                print(f"  {PINK}Usage: ada token revoke <id-or-prefix>{R}")
                return
            async with async_session_maker() as session:
                result = await session.execute(
                    select(Token).where(
                        (Token.token_prefix == ident) | (Token.token_prefix == ident[:12])
                    )
                )
                tok = result.scalar_one_or_none()
                if tok is None:
                    # try by id
                    try:
                        result = await session.execute(select(Token).where(Token.id == ident))
                        tok = result.scalar_one_or_none()
                    except Exception:
                        tok = None
                if tok is None:
                    print(f"  {PINK}No token matching {ident!r}.{R}")
                    return
                tok.status = "revoked"
                tok.revoked_at = datetime.utcnow()
                await session.commit()
                print(f"  {G}Revoked{R} {tok.name} ({tok.token_prefix}…)")

        else:  # list
            async with async_session_maker() as session:
                result = await session.execute(
                    select(Token).order_by(Token.created_at.desc())
                )
                tokens = result.scalars().all()
            if not tokens:
                print(f"  {D}No tokens. Mint one: ada token create{R}")
                return
            print(f"\n  {BOLD}Tokens{R}\n")
            for t in tokens:
                exp = t.expires_at.date().isoformat() if t.expires_at else "never"
                print(f"  {t.token_prefix}…  {t.name:<18} {t.status:<8} "
                      f"{','.join(t.permissions or [])}  exp:{exp}")
            print()

    try:
        asyncio.run(_run())
    except Exception as e:
        print(f"  {PINK}Token command failed: {e}{R}")


def _print_help() -> None:
    print(f"""
  {D}Just type anything to talk to Ada.{R}

  {BOLD}Commands{R}

  {B}help{R}           Show this message
  {B}setup{R}          How to connect Ada to an LLM
  {B}setup claude{R}   Auto-configure Claude Code
  {B}setup model{R}    Choose the local model (Ollama / llama-server)
  {B}token create{R}   Mint an MCP access token (shown once)
  {B}token list{R}     List tokens
  {B}token revoke{R}   Revoke a token by id/prefix
  {B}status{R}         Show brain status
  {B}config{R}         Show current configuration
  {B}exit{R}           Quit
""")


def _print_setup(port: int) -> None:
    url = f"http://localhost:{port}/mcp"
    print(f"""
  {P}{BOLD}Connect Ada to your LLM{R}

  {B}{BOLD}Claude Code{R} {G}(one command){R}
  {W}ada setup claude{R}

  {B}{BOLD}Claude Desktop{R}
  {D}Add to ~/Library/Application Support/Claude/claude_desktop_config.json:{R}
  {W}{{"mcpServers": {{"ada": {{"transport": "http", "url": "{url}"}}}}}}{R}

  {B}{BOLD}Cursor / VS Code{R}
  {D}Add to MCP settings:{R}
  {W}{{"ada": {{"transport": "http", "url": "{url}"}}}}{R}

  {B}{BOLD}Any MCP Client{R}
  {W}Endpoint: {url}{R}
  {W}Tool: think(input: string){R}
""")


def _get_ada_home() -> str:
    """~/.ada/ — Ada's config directory."""
    from pathlib import Path
    home = Path.home() / ".ada"
    home.mkdir(exist_ok=True)
    return str(home)


def _get_env_path() -> str:
    """Path to Ada's home env file (~/.ada/env)."""
    return f"{_get_ada_home()}/env"


def _project_env_paths() -> list[str]:
    """Candidate project .env files — the SAME file the runtime server reads.

    Checked so the REPL resolves storage/DB options (DATABASE_URL, etc.)
    identically to the server regardless of where it's launched from.
    """
    from pathlib import Path

    return [
        str(Path.cwd() / ".env"),
        str(Path(__file__).resolve().parents[2] / ".env"),  # repo root
    ]


def _load_ada_env() -> None:
    """Load Ada's env file(s) into os.environ."""
    import os

    # Load env files (lower priority than vault/explicit env). Includes
    # the project .env — the same file the runtime server reads — so the REPL
    # and server agree on storage/DB options like DATABASE_URL.
    seen: set[str] = set()
    for env_path in (_get_env_path(), *_project_env_paths()):
        if env_path in seen or not os.path.exists(env_path):
            continue
        seen.add(env_path)
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    # Override absent OR blank env vars so a stale/empty shell
                    # export can't shadow a real value from .env.
                    if value and not os.environ.get(key):
                        os.environ[key] = value


def _setup_model() -> None:
    """Choose which local model Ada uses (written to ~/.ada/env)."""
    import os

    from ada.encoder.local_enricher import local_model_name

    env_path = _get_env_path()
    current = local_model_name()

    models = [
        ("qwen3:4b",   "Qwen3 4B",   f"{G}validated — recommended (100% write / 85% read){R}"),
        ("qwen3:1.7b", "Qwen3 1.7B", f"{B}light — write-path perfect, weaker read{R}"),
    ]

    print(f"\n  {BOLD}Ada's local model{R} {D}(served by Ollama or llama-server){R}")
    print(f"  {D}Current:{R} {current}\n")

    for i, (model_id, name, desc) in enumerate(models, 1):
        marker = f" {G}←{R}" if model_id == current else ""
        print(f"  {B}{i}{R}  {W}{name}{R} {D}({model_id}){R} {desc}{marker}")

    print(f"\n  {D}Enter number or any model tag your endpoint serves (Enter to keep):{R}")

    try:
        choice = input(f"  {P}model{D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not choice:
        print(f"  {D}Keeping {current}{R}\n")
        return

    if choice.isdigit() and 1 <= int(choice) <= len(models):
        model_id = models[int(choice) - 1][0]
    else:
        model_id = choice

    # Write to ~/.ada/env
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = [ln for ln in f.readlines() if not ln.strip().startswith("ADA_LOCAL_MODEL=")]

    lines.append(f"ADA_LOCAL_MODEL={model_id}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ["ADA_LOCAL_MODEL"] = model_id
    print(f"  {G}Model set to {model_id}{R}")
    print(f"  {D}Make sure your endpoint serves it (e.g. ollama pull {model_id}).{R}\n")


def _setup_claude_code(port: int) -> None:
    import subprocess
    import shutil

    url = f"http://localhost:{port}/mcp"

    if not shutil.which("claude"):
        print(f"\n  {D}claude CLI not found. Install it first:{R}")
        print(f"  {W}npm install -g @anthropic-ai/claude-code{R}\n")
        return

    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "ada", "--transport", "http", url],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"\n  {G}{BOLD}Done.{R} Ada is connected to Claude Code.\n")
        elif "already exists" in (result.stderr or "").lower():
            print(f"\n  {G}Already configured.{R}\n")
        else:
            print(f"\n  {D}Try manually: claude mcp add ada --transport http {url}{R}\n")
    except Exception as e:
        print(f"\n  {D}Error: {e}{R}")
        print(f"  {D}Try manually: claude mcp add ada --transport http {url}{R}\n")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_error: str | None = None


def _start_server_background(host: str, port: int) -> threading.Thread:
    """Start Ada's server in a background thread."""
    import uvicorn

    def _run():
        global _server_error
        try:
            uvicorn.run(
                "ada.server:app",
                host=host,
                port=port,
                log_level="error",
            )
        except Exception as e:
            _server_error = str(e)

    t = threading.Thread(target=_run, daemon=True, name="ada-server")
    t.start()
    return t


def _wait_for_server(host: str, port: int, timeout: float = 300.0) -> bool:
    """Wait for the server to be ready. First boot can take minutes (encoding exemplars)."""
    import socket

    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    start = time.time()
    dots = 0
    while time.time() - start < timeout:
        try:
            with socket.create_connection((check_host, port), timeout=1):
                pass
            import httpx
            r = httpx.get(f"http://{check_host}:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        # Show progress dots so user knows it's working
        elapsed = int(time.time() - start)
        if elapsed > 5 and elapsed % 5 == 0 and elapsed // 5 > dots:
            dots = elapsed // 5
            sys.stdout.write(".")
            sys.stdout.flush()

        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _serve(args: list[str]) -> None:
    """Run Ada headless (no shell, MCP endpoint only)."""
    import uvicorn
    from infrastructure.config import get_settings

    settings = get_settings()

    port = settings.port
    host = settings.host
    for i, arg in enumerate(args):
        if arg in ("-p", "--port") and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg in ("-h", "--host") and i + 1 < len(args):
            host = args[i + 1]

    _print_banner()

    _local_model_notice()
    _print_config(host, port)

    uvicorn.run(
        "ada.server:app",
        host=host,
        port=port,
        log_level="info",
    )


def main() -> None:
    # Storage defaults to "sql" (durable, consistent — see
    # infrastructure/config.py). Set ADA_STORAGE=memory for an ephemeral
    # in-process space.
    from infrastructure.config import get_settings
    settings = get_settings()

    args = sys.argv[1:]

    # Parse global flags
    host = settings.host
    port = settings.port
    for i, arg in enumerate(args):
        if arg in ("-p", "--port") and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg in ("-h", "--host") and i + 1 < len(args):
            host = args[i + 1]

    # Load stored env vars (API key, model, etc.)
    _load_ada_env()

    if not args:
        _admin_repl(host, port)
    elif args[0] == "serve":
        _load_ada_env()  # also for headless
        _serve(args[1:])
    elif args[0] == "setup":
        if len(args) > 1 and args[1] in ("claude", "claude-code"):
            _setup_claude_code(port)
        elif len(args) > 1 and args[1] == "model":
            _setup_model()
        else:
            _print_setup(port)
    elif args[0] == "token":
        _token_cmd(args[1:])
    elif args[0] == "help":
        _print_banner()
        _print_help()
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: ada [serve|setup|token|help]")
        sys.exit(1)


def _ensure_server(host: str, port: int) -> str | None:
    """Banner + ensure a server is reachable, auto-starting one in-process
    when none is. Returns the /mcp url, or None if the server failed to
    start. Boot look and feel match the original shell."""
    import httpx
    from ada import __version__

    _print_banner(__version__)

    probe_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{probe_host}:{port}/mcp"
    health = f"http://{probe_host}:{port}/health"
    up = False
    try:
        up = httpx.get(health, timeout=2.0).status_code == 200
    except Exception:
        pass

    if up:
        print(f"  {G}● Online{R}   ")
    else:
        import logging
        for name in ["ada", "domains", "infrastructure", "shared", "api", "mcp"]:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.addHandler(logging.NullHandler())

        print(f"  {D}Starting...{R}", end="", flush=True)
        _start_server_background(host, port)
        if _wait_for_server(probe_host, port):
            print(f"\r  {G}● Online{R}   ")
        else:
            print(f"\r  {PINK}● Failed to start{R}   ")
            if _server_error:
                print(f"  {PINK}{_server_error}{R}")
            print(f"  {D}Try: ada serve (to see full error output){R}")
            return None

    _print_config(host, port)
    return url


def _admin_repl(host: str, port: int) -> None:
    """The `ada` experience: a plain REPL — the single interface (think ·
    ask · tell · query · history · stats · …) over the server's /mcp
    door. No mouse, no full-screen panels, no login."""
    url = _ensure_server(host, port)
    if url is None:
        return

    from ada.tui.repl import run_repl
    run_repl(url)


if __name__ == "__main__":
    main()

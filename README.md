# Ada

**Schema-on-write memory for LLMs**, by [Glyphh AI](https://glyphh.ai).
Facts in prose, answers exact, beliefs versioned, refusals honest.

```bash
pip install glyphh-ada        # imports as `ada`  (PyPI publish pending —
                              #  until then: pip install git+https://github.com/glyphh-ai/glyphh-adaL)
```

Ada is a universal fact table, deliberately not a graph: an LLM maps
each fact into one fixed schema at write time, and reads are exact,
deterministic scans — aggregation, distribution, multi-condition
intersection, versioned history — with full provenance, on corpora far
larger than any prompt can hold. It runs on-device (SQLite, zero setup)
or against Postgres for scale, and exposes itself to any LLM over MCP.

Measured (pre-registered protocol, raw logs in
[`ada-benchmarks`](https://github.com/glyphh-ai/ada-benchmarks)): matches
or beats a hand-schema'd graph database on NL query accuracy from 1K to
1M facts with **zero per-domain schema design** (100.0% at 1M facts,
no degradation with scale), beats text-to-SQL over a generic EAV store
by ~20 points, and beats embedding RAG by 30+ (RAG decays to ~24% at 1M
facts; Ada doesn't). 1M facts ingest in ~9 seconds. Every number traces
to the protocols and phase reports in
[`ada-benchmarks`](https://github.com/glyphh-ai/ada-benchmarks) —
including the ones we lost.

---

## How it works

Three pieces, each independently inspectable:

1. **Universal schema** — a fixed 7-layer × 33-role lattice every fact maps
   into (entity, perceptual, spatial, temporal, relational, quantitative,
   epistemic). Counting "facts where color = blue" is an exact scan; "top
   5 cities" is `Counter().most_common(5)`. An empty slot is a structural
   ∅ — Ada refuses instead of confabulating. `ada/cognitive/universal.py`.

2. **Versioned memory** — write a fact under a `key` and it becomes v1, v2,
   v3… with the full chain queryable via `history`. `ada/memory/thought_space.py`.

3. **Storage** — SQLite by default (`~/.ada/ada.db`, zero setup). Point
   `DATABASE_URL` at Postgres and the runtime switches backends. Two
   engine modes: **sql** (default — the eight closed ops compiled to fixed
   SQL templates over an indexed `fact_slots` table; the store IS the DB,
   so writes/forgets/amends are durable and consistent; O(1) boot, O(1)
   RAM, for 10M+-fact corpora) and **memory** (`ADA_STORAGE=memory` —
   in-process indexes, 0–sub-ms ops, ephemeral; for benchmarks/throwaway). The
   LLM only ever picks an op and fills values — no model-generated SQL
   exists, so the runaway-query class is excluded by construction.

Facts arrive two ways: **`tell_raw`** takes pre-structured slot fills with
no model in the path; **`tell`** takes natural language — a LOCAL model
(Ollama or llama.cpp's llama-server at `ADA_LOCAL_URL`; measured at 100%
extraction accuracy from Qwen3-1.7B up) maps the text into the schema at
write time, one cached call per fact, zero egress. **No cloud API key
exists anywhere in Ada.** Without a local endpoint, a regex heuristic
covers common fact shapes. The read path never calls a model; structured
queries and lexical recall are exact, deterministic scans.

---

## Install & start

### Quickest: `make`

```bash
make install        # .venv + editable install (+ dev extras), seeds .env
# local model (optional — substrate works without it): ollama pull qwen3:4b
make dev            # server with hot-reload on http://localhost:8002
```

Other targets: `make repl` (interactive shell), `make serve` (no reload),
`make test`, `make fmt`, `make clean`. Run `make` alone to list them.

### The terminal app — the interface

Just run `ada`. It opens a **full-screen terminal app** — one interface
for the whole substrate:

- **Dashboard** — live vital signs as terminal graphs: stat cards, the
  per-layer slot-fill bar chart, and a session sparkline, refreshed live.
- **Terminal** — the full command surface (`tell` · `ask` · `think` ·
  `count` · `top` · `find` · `history` · `recall` · …). Bare text with no
  verb is treated as `ask`.
- **Memory** — keyed facts at their current belief — filter, edit in
  place, delete per-row.
- **Constellation** — the entity point cloud, in the terminal.
- **API Keys** — mint / list / revoke the credentials that open `/mcp`.

Everything is an **MCP tool call** against the running server (the same
`/mcp` door Claude and every other client use), so the app, Claude, and
all clients share one SQL-backed memory. Writes are write-through
durable. Switch views with the rail (or keys `1`–`5`).

```bash
make dev            # start the server (or `ada` auto-starts one)
ada                 # the terminal app  ·  ADA_URL=http://host:8002/mcp ada
ada repl            # escape hatch: the classic line REPL
```

```
  ada› tell key=carol.location Carol lives in Austin.
  absorbed v1
  ada› tell key=carol.location Carol lives in Denver.
  absorbed v2
  ada› count spatial.location=denver
  3                                  # current belief — Carol's Austin era excluded
  ada› history carol.location
  v1: Carol lives in Austin.   v2: Carol lives in Denver.
  ada› top relational.object work    # distribution, predicate-filtered
  engineer (12), designer (9), ...
```

Type `help` in the REPL for the full command list. (The old browser
workbench and the full-screen TUI are both retired — the REPL is the
one interface.)

---

## Production: Docker

The image is storage-agnostic — `DATABASE_URL` decides the backend. Unset, Ada
runs on in-process SQLite. Point it at Postgres and the runtime runs the
alembic migrations on first boot.

### Recommended: compose (runtime + Postgres)

```bash
cp .env.example .env          # optional: ADA_LOCAL_URL for your model — DB is wired by compose
docker compose up -d --build
curl localhost:8002/health
```

The bundled `db` service is `postgres:16`; the runtime waits for it to be
healthy, connects via `DATABASE_URL=postgresql://ada:ada@db:5432/ada`, and
persists memory to the `pgdata` named volume across restarts and upgrades.

### Single container (SQLite, no Postgres)

```bash
docker build -t ada .
docker run -d -p 8002:8002 \
  -e ADA_LOCAL_URL=http://host.docker.internal:11434/v1 \
  -v ada-data:/root/.ada \
  ada
```

Leaving `DATABASE_URL` unset keeps SQLite at `~/.ada/ada.db`; mount a volume
there to persist the substrate.

---

## Using it from an LLM (MCP)

Ada exposes a single MCP endpoint at **`/mcp`**. Start the server, then point
any MCP client at `http://localhost:8002/mcp`.

### The tools

| Tool | What it does |
|---|---|
| `think(input, top_k)` | Broad recall — surfaces the thoughts that match. No answer, just context. |
| `ask(question)` | Targeted retrieval. Returns one fact + confidence, or "I don't know." |
| `tell(text, key?)` | Natural-language ingestion. With `key`, becomes a new version of that key. |
| `tell_raw(facts, key?)` | Programmatic ingestion of a universal-schema slot fill. **No LLM in the path.** |
| `recall(query, top_k)` | Deterministic lexical search. Top matches + similarity + versioned key. |
| `query(op, ...)` | One structured operation from the closed op set: count / who / top / lookup / prev / count_not / compare. Exact, no LLM. |
| `history(key)` | Full version chain for a key, chronological. |
| `stats()` | Substrate vital signs: thoughts, versioned keys, per-layer slot fill. |
| `create_token(...)` | Mint an API token for this MCP server (returned once). |

`domains/mcp/server.py`.

### Connecting clients

```bash
ada setup claude            # auto-configure Claude Code
ada setup                   # print config snippets for Claude Desktop / Cursor / any MCP client
```

### Spaces (multi-tenant scoping)

Every fact lives in a **space** (default `main`). Pass `space` to any
MCP tool to read/write an isolated space — facts, counts, and version
chains never cross. In the REPL, `space <name>` switches the session's
active space. A token can be **bound to one space** (its `model_id`
column): a space-bound token is rejected (403) on any other space, so
each team or engagement gets a token that can only touch its own
memory. This scoping lives in the open runtime — self-hosted
multi-team works without any hosted control plane.

### Identity (first-person facts)

Tell Ada who you are (`me chris` in the REPL, or `speaker` on the
`tell` tool) and first-person statements resolve to your entity at
write time: "I am married to Brandi" stores under `entity=chris`, so
"who is my wife?" becomes an exact entity join. Third-person facts are
untouched. Deterministic, no LLM.

### Authentication

There isn't any. Machine access to this process is the trust boundary —
the same posture as Ollama or a local Postgres. `ada token create` (or
`connect`/`token` in the REPL) still mints attributable tokens if you
want your own bookkeeping, or a reverse proxy that checks one for a
non-local deployment, but nothing on `/mcp` requires or checks a
credential.

```bash
ada                       # in the REPL: connect | token | token revoke <id>
```

### HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness probe |
| `POST /mcp` | MCP tool calls (Streamable HTTP) |

---

## Configuration

All values have defaults — the server boots with an empty `.env`. The `.env`
is read by both the server and the `ada` REPL.

| Variable | Default | Description |
|----------|---------|-------------|
| `ADA_LOCAL_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible local endpoint (Ollama default; llama-server works too) |
| `ADA_LOCAL_MODEL` | `qwen3:4b` | Local model tag for enrichment / rendering |
| `DATABASE_URL` | SQLite at `~/.ada/ada.db` | `postgresql://…` switches to the Postgres backend |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8002` | Server port |
| `ADA_STORAGE` | `sql` | `sql` (fact_slots-backed, durable, O(1) boot for 10M+ facts) or `memory` (in-RAM indexes, ephemeral) |
| `ENABLE_DOCS` | `false` | Enable `/docs` and `/redoc` |

---

## Architecture map

```
ada/
  cognitive/
    universal.py       7-layer × 33-role universal schema + LLM enricher
    surface.py         think / ask cognitive surfaces
    generate.py        optional LLM renderer (the surface form)
  memory/
    thought_space.py   the substrate: tell_raw / absorb / structured
                       queries (count, distribution, intersection) /
                       lexical recall / versioned keys
    thought_persistence.py   DB sync (SQLite / Postgres)
    ada_cognitive.py   memory + input routing + conversation
  encoder/
    llm_enricher.py    LLM-at-build-time text → schema extraction
  tui/                 the `ada` REPL — the single interface

domains/
  brain/               think() pipeline (perceive → recall → respond)
  mcp/server.py        MCP tools: think / ask / tell / tell_raw / recall /
                       history / stats / create_token

api/                   FastAPI HTTP routes (health)
infrastructure/        config + database (SQLite / Postgres)
shared/                exceptions, middleware
scripts/               the 77-fact hard test
```

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Copyright © 2026 Glyphh AI LLC.

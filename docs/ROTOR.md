# Rotors

A **rotor** is a set of steps that run in sequence against a prompt: work
through the lattice (and a semantic embedding layer) to find memories
related to the current activity, and inject a clear, bounded set of them
into the LLM prompt **just in time**.

The first rotor ships as an Anthropic-compatible proxy: point Claude
Desktop (Developer → Configure Third-Party Inference → Gateway) or any
`ANTHROPIC_BASE_URL` client at `http://localhost:8002` and every
`/v1/messages` call flows through the rotor on its way to Anthropic.
The caller's own API key rides the request headers and is forwarded
verbatim — Ada never stores it.

## The three stages

```
            ┌─────────────────────────────────────────────┐
 prompt ──▶ │ 1 WRITE   store the turn, verbatim          │
            │ 2 RECALL  primes → lattice + semantic layer │
            │           → the fact tree for this prompt   │
            │ 3 REASON  compose the JIT injection, route  │
            └─────────────────────────────────────────────┘ ──▶ Anthropic
```

1. **Write** — every prompt is stored as a whole, in sequence, into
   short-term memory (verbatim turns + an embedding vector).
2. **Recall** — the prompt is decomposed into **semantic primes**; the
   rotor walks the lattice (exact keyed facts, entity joins) and the
   semantic layer (embedding search over turns and events), then builds
   the **fact tree** for the prompt: recalled memories grouped under the
   primes they relate to.
3. **Reason** — the fact tree is composed into one clearly-marked,
   bounded system block (`<ada:memory>…</ada:memory>`) appended to the
   caller's system prompt, and the request is routed upstream. The
   response streams back verbatim; the assistant's reply is written back
   to short-term memory as it completes.

## Layered memory — the cascade

Prompts happening one after the other are stored **as a whole** →
short term. Short term is **summarized into events** → mid term. Events
are **distilled into facts** → long term.

| layer | representation | recall |
|---|---|---|
| short | `rotor_turns` — verbatim turns, per conversation, + vectors | embedding search |
| mid   | `rotor_events` — local-model summaries of turn spans, + vectors + primes | embedding search + prime overlap |
| long  | the Ada lattice — facts distilled from events through `absorb` (dedup + 7×33 enrichment) | deterministic: keys, entity joins, lexical recall |

The cascade runs in the background and **degrades silently**: no local
model → turns simply pool unsummarized until one appears; no embedder →
semantic recall is skipped and the lattice still answers. Nothing ever
blocks the proxied request.

## Semantic primes

Every idea is taken down to the primes it relates to (the ~65 NSM
universal primitives — KNOW, WANT, FEEL, DO, HAPPEN, WHERE, WHEN, GOOD,
BAD, …). Decomposition is heuristic-first (a deterministic keyword map,
always available) and sharpened by the local model when one is up.
Primes are stored on turns and events at write time; at recall time the
prompt's own primes select and group what comes back — that grouping
*is* the fact tree.

## Auth — Ada is a router, so there are two credentials

```
client ──[ Ada token: ada_… ]──▶ Ada ──[ the user's provider key ]──▶ Anthropic
        (Gateway API key field)    │                                └▶ OpenAI …
                                   validates the ada_ token (her Token
                                   table); attaches the stored provider key
```

- **client → Ada** authenticates with an **Ada token** (`ada token create`
  → `ada_…`, stored as a sha256 hash). That's the value you put in the
  client's "Gateway API key" field — *not* your Anthropic key. The gate is
  bootstrap-friendly: open until the first token is minted, then required;
  a bad token gets Ada's own 401, and it fails **closed** if the token
  store can't be read.
- **Ada → provider** attaches the **user's own provider key**, which Ada
  holds (`ADA_ANTHROPIC_KEY` / `ANTHROPIC_API_KEY`, `ADA_OPENAI_KEY` /
  `OPENAI_API_KEY`). For Anthropic that goes upstream as `x-api-key`
  (so the client's Bearer-vs-x-api-key choice no longer matters). If Ada
  holds no provider key, it falls back to forwarding the caller's own
  credential.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ADA_UPSTREAM` | `https://api.anthropic.com` | where proxied requests go |
| `ADA_ROTOR` | `true` | run the rotor on `/v1/messages` (off = pure passthrough) |
| `ADA_ANTHROPIC_KEY` | (`ANTHROPIC_API_KEY`) | the user's Anthropic key, attached upstream |
| `ADA_OPENAI_KEY` | (`OPENAI_API_KEY`) | the user's OpenAI key (for OpenAI routes) |
| `ADA_EMBED_URL` | (`ADA_LOCAL_URL`) | OpenAI-compatible `/embeddings` endpoint |
| `ADA_EMBED_MODEL` | `nomic-embed-text` | embedding model (`ollama pull nomic-embed-text`) |
| `ADA_LOCAL_MODEL` | `qwen3:4b` | local model for summarize/distill/decompose — install whatever you like |
| `ADA_INJECT_BUDGET` | `1500` | max characters of injected memory |

The local model is the user's choice end to end: anything served on an
OpenAI-compatible endpoint works, and every rotor feature that needs it
falls back to deterministic heuristics (or waits) when it is missing.

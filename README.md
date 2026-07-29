# rag-at-scale — what actually breaks at a few million vectors

The deep-dive series builds RAG from scratch and proves the ideas on a corpus of
four Markdown files. This project asks the question interviews (and production)
actually ask: **what happens at 1–5 million chunks?** Brute force stops being
free, index choice starts to matter, and every knob becomes a measured tradeoff
instead of a vibe.

The deliverable is not the code — it's the **numbers**: recall@k, p50/p95 query
latency, index build time, and memory, across brute force vs IVFFlat vs HNSW,
as the corpus grows. See [PLAN.md](PLAN.md) for the phases and the results tables.

**→ Read the write-up: [WRITEUP.md](WRITEUP.md)** ("RAG at 5M documents — what
actually breaks") — or the [visual version with a chart per phase](https://claude.ai/code/artifact/d0e5e9b4-97de-4f43-9805-7c1a1f1dc46c).
The hard-won gotchas live in [LESSONS.md](LESSONS.md). Running it yourself?
[WALKTHROUGH.md](WALKTHROUGH.md) is the end-to-end journey — every fork in the
road, what each stage costs, and where this is the wrong tool.

**Two embedding paths.** The database is always pgvector in Docker on this
machine. For embeddings you pick a provider with `EMBED_PROVIDER`:

- **`ollama`** — local `nomic-embed-text`, **$0**, no API key, but compute-bound
  at ~165 chunks/s (millions of chunks = hours).
- **`voyage`** — hosted Voyage API, many concurrent batched requests, dramatically
  faster. Needs a `VOYAGE_API_KEY`, but the full 5M-vector sweep (~140M tokens)
  fits inside Voyage's free token tier, so it stays ~$0 in practice.

pgvector doesn't care where the vectors come from, so the scaling results are the
same either way — Voyage just buys back the embedding hours.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # defaults work out of the box
docker compose up -d               # pgvector on localhost:5433
python check_setup.py              # verifies Docker, Postgres, and the embedding provider

# End-to-end smoke run at tiny scale (1k synthetic chunks):
python -m ragscale.bench --n 1000
```

The smoke run exercises the whole pipeline — generate chunks → embed via Ollama
→ load into pgvector → query — so scaling up is only a matter of `--n`.

## Layout

| Path | What it is |
|---|---|
| `ragscale/config.py` | Env-driven settings (DSN, Ollama URL, embed model/dim) |
| `ragscale/corpus.py` | Deterministic synthetic corpus generator (swap for Wikipedia later) |
| `ragscale/embed.py` | Batched embeddings against the local Ollama server |
| `ragscale/db.py` | Schema, bulk load, brute-force + indexed queries, index builders |
| `ragscale/bench.py` | The measurement harness (start here) |
| `docker-compose.yml` | pgvector (Postgres 17) with a persistent volume |
| `PLAN.md` | Phases, definitions of done, and the results tables |

## Honest scoping

- Synthetic chunks are fine for *retrieval mechanics* (recall/latency/memory are
  functions of vector geometry and count, not prose quality). For a write-up
  with real queries, phase 4 swaps in a Wikipedia slice.
- This measures the **retrieval** layer at scale. LLM serving throughput (vLLM,
  continuous batching) is a different project and needs an NVIDIA GPU.

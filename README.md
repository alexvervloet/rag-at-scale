# rag-at-scale — what actually breaks at a few million vectors

The deep-dive series builds RAG from scratch and proves the ideas on a corpus of
four Markdown files. This project asks the question interviews (and production)
actually ask: **what happens at 1–5 million chunks?** Brute force stops being
free, index choice starts to matter, and every knob becomes a measured tradeoff
instead of a vibe.

The deliverable is not the code — it's the **numbers**: recall@k, p50/p95 query
latency, index build time, and memory, across brute force vs IVFFlat vs HNSW,
as the corpus grows. See [PLAN.md](PLAN.md) for the phases and the results
tables to fill in.

**Cost: $0.** Embeddings come from a local model (`nomic-embed-text` via
Ollama), the database is pgvector in Docker, and everything runs on this
machine. No API key is used anywhere in this repo.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # defaults work out of the box
docker compose up -d               # pgvector on localhost:5433
python check_setup.py              # verifies Docker, Postgres, Ollama, model

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

# Plan — phases and definitions of done

Every phase ends with numbers in a table, not a feeling. Keep raw outputs in
`results/` (gitignored) and paste the summary tables here.

## Phase 0 — plumbing (done when the smoke run passes)

- [x] `python check_setup.py` all green
- [x] `python -m ragscale.bench --n 1000` completes: generate → embed → load → query

## Phase 1 — brute force baseline (done when the table is full)

Load N chunks, run 100 held-out queries with exact (sequential-scan) cosine
search. This is ground truth for every later recall number.

| N vectors | load time | table size | p50 query | p95 query |
|---|---|---|---|---|
| 100k | 604s | 413 MB | 91ms | 97ms |
| 500k | | | | |
| 1M | | | | |
| 5M | | | | |

Expected shape: query latency grows ~linearly with N. Find where it crosses
"too slow for a chat app" (~200ms?) — that's the motivation for phase 2.

## Phase 2 — IVFFlat (done when the recall/latency curve exists)

Build IVFFlat at 1M+ vectors. Sweep `lists` (index build knob) and `probes`
(query knob) and measure against phase 1's exact results.

| lists | probes | recall@10 | p50 | p95 | build time |
|---|---|---|---|---|---|

Questions to answer in writing: where is the knee of the recall/latency curve?
What does the `lists ≈ sqrt(N)` rule of thumb actually buy?

## Phase 3 — HNSW (done when the comparison table exists)

Same sweep with HNSW (`m`, `ef_construction` at build; `ef_search` at query).

| m | ef_search | recall@10 | p50 | p95 | build time | index size |
|---|---|---|---|---|---|---|

Then the headline table: brute force vs IVFFlat vs HNSW at 1M and 5M, at
matched recall (e.g. everything tuned to recall@10 ≈ 0.95).

## Phase 4 — make it a story (the write-up)

- [ ] Swap synthetic corpus for a Wikipedia slice; re-verify one table
- [ ] Concurrency: 10/50 simultaneous queries (locust or asyncio) — does p95 hold?
- [ ] Write "RAG at 5M documents — what actually breaks" with the tables and
      one graph per phase

## Notes / gotchas discovered along the way

(keep a running log here — this becomes the best part of the write-up)

- **Load time is embedding, not pgvector.** Ingest runs at ~165 chunks/s, fully
  bottlenecked by local Ollama (`nomic-embed-text`), so the "load time" column is
  really a corpus-prep cost, not a database ingest number. Practical consequence:
  5M rows is ~8.4h of embedding. The harness grows the table in place across N
  (100k → 500k → …) so each chunk is embedded exactly once over the whole sweep.
- **Extrapolating from tiny N lies.** Sub-10k latencies (~3–8ms) are dominated by
  fixed query overhead, not the scan. The first real anchor is 100k @ 91ms p50 →
  ~0.9ms per 1k rows, which puts the ~200ms "too slow for chat" crossing around
  ~220k. Confirm with the 500k point.

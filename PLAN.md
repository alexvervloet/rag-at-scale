# Plan — phases and definitions of done

Every phase ends with numbers in a table, not a feeling. Keep raw outputs in
`results/` (gitignored) and paste the summary tables here.

## Phase 0 — plumbing (done when the smoke run passes)

- [x] `python check_setup.py` all green
- [x] `python -m ragscale.bench --n 1000` completes: generate → embed → load → query

## Phase 1 — brute force baseline (done when the table is full)

Load N chunks, run 100 held-out queries with exact (sequential-scan) cosine
search. This is ground truth for every later recall number.

Embeddings: **Voyage `voyage-4`, 1024-d** (asymmetric: corpus=document,
queries=query). Load time is the *incremental* embed+COPY for that step (the
table is grown in place), not a from-scratch load — cumulative embed to 5M was
~47 min. Query latency is p50/p95 over the 100 held-out queries.

| N vectors | load time (step) | table size | p50 query | p95 query |
|---|---|---|---|---|
| 100k | 57s | 546 MB | 110ms | 119ms |
| 500k | 220s | 2.7 GB | 779ms | 828ms |
| 1M | 277s | 5.3 GB | 1482ms | 1554ms |
| 5M | ~2290s | 27 GB | 14308ms | 16586ms |

Result: latency crosses the ~200ms "too slow for a chat app" line around
**~180k** vectors — brute force is done as a primary retrieval path well before
1M. And it's worse than linear: 1M→5M is 5× the data but **9.7× the latency**,
because at 27 GB the table no longer fits in RAM/`shared_buffers` and every query
reads from disk. That memory cliff — not just O(n) scan cost — is the real
motivation for the ANN indexes in phase 2.

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

- **Load time is embedding, not pgvector.** Ingest is bottlenecked by the
  embedder, not the database, so the "load time" column is really a corpus-prep
  cost. Local Ollama (`nomic-embed-text`) did ~165 chunks/s → 5M would be ~8.4h;
  we switched to Voyage (`voyage-4`, concurrent batches) at ~1800 chunks/s → ~46
  min. The harness grows the table in place across N so each chunk is embedded
  exactly once over the whole sweep.
- **Extrapolating from tiny N lies.** Sub-10k latencies (~3–8ms) are dominated by
  fixed query overhead, not the scan. On 768-d nomic the 100k anchor (91ms) put
  the ~200ms crossing near ~220k; on the final 1024-d Voyage run the 100k anchor
  is 110ms (1024-d costs more per distance), moving the real crossing to **~180k**.
  Lesson holds: get one real-scale anchor before drawing the curve.
- **Brute force hits a memory cliff, not just an O(n) wall.** Latency is roughly
  linear while the table fits in RAM (100k→1M), then goes super-linear: 1M→5M is
  5× the rows but 9.7× the latency (1.48s → 14.3s), because at 27 GB the table
  exceeds `shared_buffers`/page cache and every query reads from disk (`IO` wait,
  scanned across parallel workers). The knee is where the working set stops
  fitting in memory — worth remembering when phase 2/3 index *sizes* start to
  matter as much as their algorithms.

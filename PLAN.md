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

IVFFlat at **5M vectors** (voyage-4, 1024-d), recall@10 vs phase-1 exact results.
`lists` brackets the two rules of thumb: √N ≈ 2236 and N/1000 = 5000. Index is 38 GB
regardless of lists (it re-stores every vector). Latency **forces the index**
(`enable_seqscan=off`) so we measure IVFFlat, not the planner fallback (see below);
lists=1000/5000 re-measured forced, lists=2236 from the initial sweep (verified
fallback-free).

| lists | probes | recall@10 | p50 | p95 | build time |
|---|---|---|---|---|---|
| 1000 | 1 | 0.809 | 11ms | 22ms | 637s |
| 1000 | 5 | 1.000 | 39ms | 67ms | 637s |
| 1000 | 10 | 1.000 | 110ms | 181ms | 637s |
| 1000 | 20 | 1.000 | 230ms | 359ms | 637s |
| 1000 | 50 | 1.000 | 603ms | 941ms | 637s |
| 1000 | 100 | 1.000 | 1640ms | 2448ms | 637s |
| 2236 | 1 | 0.523 | 7ms | 24ms | 1260s |
| 2236 | 5 | 0.982 | 23ms | 50ms | 1260s |
| 2236 | 10 | 1.000 | 47ms | 85ms | 1260s |
| 2236 | 20 | 1.000 | 105ms | 198ms | 1260s |
| 2236 | 50 | 1.000 | 289ms | 436ms | 1260s |
| 2236 | 100 | 1.000 | 602ms | 862ms | 1260s |
| 5000 | 1 | 0.418 | 6ms | 9ms | 2959s |
| 5000 | 5 | 0.896 | 13ms | 22ms | 2959s |
| 5000 | 10 | 0.984 | 19ms | 28ms | 2959s |
| 5000 | 20 | 1.000 | 37ms | 51ms | 2959s |
| 5000 | 50 | 1.000 | 135ms | 201ms | 2959s |
| 5000 | 100 | 1.000 | 269ms | 379ms | 2959s |

**IVFFlat vs brute force:** every config reaches recall@10 = 1.0 at ~40ms, versus
phase-1's **14,308ms** exact scan — a **~350× speedup at perfect recall**. This is
the payoff the phase-1 cliff was setting up.

**Where's the knee?** Recall climbs fast then saturates: for √N it's 0.52 → 0.98 →
1.0 across probes 1 → 5 → 10, and latency is linear in probes. The knee is ~probes
where recall first hits ~1.0 (probes=10 at √N, 47ms) — past it you pay latency for
zero recall gain. (Caveat: the synthetic corpus is very clusterable, so recall
saturates unrealistically early — real data needs a gentler curve; phase 4.)

**What does `lists ≈ √N` buy?** Less than the rule implies — for *query* speed,
almost nothing here. At matched recall (1.0) the latency is roughly flat across
lists — 39ms (lists=1000) ≈ 37ms (lists=5000) — because reaching a given recall
means scanning a roughly constant number of candidates (~20–25k vectors), and
`lists` just repartitions them: finer cells need proportionally more probes. What
`lists` *does* move, steeply, is **build time: 637s → 1260s → 2959s** (10.6 → 21 →
49 min). So on this data lists=1000 built 5× faster than lists=5000 for identical
query performance. `probes` is the recall/latency knob; `lists` is mostly a
build-cost decision, and √N is a safe middle rather than a query-speed win.

**Gotcha — the planner abandons the index at high probes.** With default settings,
pgvector's IVFFlat index-scan cost estimate grows with `probes` and, past a
threshold, the planner picks a parallel seq scan instead — silently giving you
brute-force latency (probes=100 at lists=5000: **14.3s** as a seq scan vs **269ms**
forced with `enable_seqscan=off`). A "using the index" assumption is not safe at
scale; check `EXPLAIN` or force the index. (This is what produced two bogus ~14s
rows in the first sweep — see LESSONS #9.)

## Phase 3 — HNSW (done when the comparison table exists)

**Corpus switched to real data here.** The synthetic corpus (dense near-duplicate
geometry) broke the ANN-vs-ANN comparison: it gave IVFFlat recall 1.0 trivially but
capped HNSW at 0.77 even at ef_search=1000, because HNSW's greedy graph walk gets
trapped among near-tied vectors — inverting the real-world ranking (LESSONS #11).
So phases 3's comparison runs on **real Wikipedia (wikitext-103), 500k chunks,
voyage-4 1024-d** (`CORPUS_SOURCE=wiki`), where neighborhoods are genuinely spread
out (top-10 distance spread ~0.06 vs synthetic's ~0.001) and every method's recall
climbs gradually — the honest tradeoff.

HNSW sweep (`m`, `ef_construction` at build; `ef_search` at query), index forced:

| m | ef_construction | ef_search | recall@10 | p50 | p95 | build | index |
|---|---|---|---|---|---|---|---|
| 16 | 64 | 10 | 0.779 | 0.8ms | 1.2ms | 144s | 3906 MB |
| 16 | 64 | 40 | 0.927 | 1.3ms | 1.9ms | 144s | 3906 MB |
| 16 | 64 | 100 | 0.975 | 1.9ms | 3.1ms | 144s | 3906 MB |
| 16 | 64 | 200 | 0.986 | 3.2ms | 4.9ms | 144s | 3906 MB |
| 16 | 200 | 40 | 0.949 | 1.5ms | 2.2ms | 381s | 3906 MB |
| 16 | 200 | 200 | 0.992 | 3.9ms | 6.1ms | 381s | 3906 MB |
| 32 | 64 | 40 | 0.965 | 1.7ms | 2.9ms | 252s | 3906 MB |
| 32 | 200 | 40 | 0.976 | 1.8ms | 2.9ms | 594s | 3906 MB |
| 32 | 200 | 200 | 0.995 | 5.3ms | 9.0ms | 594s | 3906 MB |

Higher `m`/`ef_construction` lift the recall ceiling and the recall-per-ef_search
(m=32,efc=200 reaches 0.95 by ef_search=40), at a steeper build cost (144s → 594s).
`ef_search` is the query knob, trading recall for latency along each curve.

### Headline — brute force vs IVFFlat vs HNSW at recall@10 ≈ 0.95 (500k real)

| method | config @ ~0.95 recall | recall@10 | p50 | vs brute | index build |
|---|---|---|---|---|---|
| brute force | (exact seq scan) | 1.000 | 889ms | 1× | — |
| IVFFlat | lists=707, probes=20 | 0.955 | 16.2ms | **55×** | 72s |
| HNSW | m=16, efc=200, ef_search=40 | 0.949 | 1.5ms | **~600×** | 381s |

**The result the synthetic data hid:** HNSW wins decisively — ~1.5ms at 0.95 recall,
~10× faster than IVFFlat and ~600× faster than brute force, for a comparable index
size (~3.9 GB) but a slower build. This is the textbook ordering (HNSW ≥ IVFFlat on
the recall/latency frontier) — and the *exact opposite* of what the synthetic corpus
"proved" (where HNSW couldn't clear 0.77). The tradeoff HNSW asks for is build time
and a fixed graph (no cheap rebuilds); IVFFlat builds far faster and is easier to
tune, which is its own kind of win.

(Scale note: the comparison ran at 500k real rather than 5M — validity needs real
geometry, not raw scale, and real embeddings are token-throughput-bound to embed.
Phases 1–2's 5M synthetic tables still stand for the brute-force cliff and the
IVFFlat mechanics; the 5M-vs-500k gap is a scale caveat, not a different conclusion.)

## Phase 4 — make it a story (the write-up)

- [x] Swap synthetic corpus for a Wikipedia slice; re-verify one table
      (done early in phase 3 — the synthetic geometry broke the ANN comparison,
      so the IVFFlat/HNSW head-to-head was rerun on real wikitext-103 @ 500k)
- [x] Concurrency: 10/50 simultaneous queries (asyncio/threads) — does p95 hold?
      Threads on their own connections hit the live HNSW index (500k wiki, ef_search=40):

      | concurrency | p50 | p95 | p99 | throughput |
      |---|---|---|---|---|
      | 1 | 1.3ms | 2.1ms | 2.2ms | 669 qps |
      | 10 | 1.9ms | 3.0ms | 4.8ms | 3850 qps |
      | 50 | 7.0ms | 22.8ms | 35.3ms | 4551 qps |

      p95 holds (~3ms) through concurrency=10 while throughput scales ~5.7×; at 50 it
      degrades *gracefully* (p95 23ms, no collapse) as the 12-vCPU VM saturates and
      throughput plateaus (~4550 qps). Classic queueing: throughput maxes near the
      core count, and past that you buy latency, not QPS. (Warm the index first — a
      cold 3.9GB index makes the first level measure disk, not concurrency.)
- [x] Write "RAG at 5M documents — what actually breaks" with the tables and
      one graph per phase — [WRITEUP.md](WRITEUP.md) + a
      [visual version](https://claude.ai/code/artifact/d0e5e9b4-97de-4f43-9805-7c1a1f1dc46c)
      (a chart per phase)

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

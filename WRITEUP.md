# RAG at 5M documents — what actually breaks

The tutorials prove RAG on four markdown files. This project asked the question
production asks: **what happens at 1–5 million chunks?** The answer isn't one wall,
it's a sequence of them — each a different subsystem tapping out — and a couple of
places where the *benchmark itself* was the thing that broke.

Everything below is measured, not asserted. pgvector in Docker, embeddings from
Voyage (`voyage-4`, 1024-d), one laptop. Raw numbers in `results/`, the full log of
gotchas in [LESSONS.md](LESSONS.md), the phase definitions in [PLAN.md](PLAN.md).

📊 **[Visual version — a chart per phase](https://claude.ai/code/artifact/d0e5e9b4-97de-4f43-9805-7c1a1f1dc46c)**

---

## The setup, and the first thing that broke: the plan to stay free

The design was "$0, all local" — embeddings from Ollama's `nomic-embed-text`. It
worked, and it was the wrong tool: **~165 chunks/s, entirely bottlenecked by the
local model.** 5M rows would have been ~8.4 hours of embedding. That single fact
reshaped everything — so the first real decision was to trade the "completely free"
badge for a Voyage API key. Concurrent batched requests (768/thread, 32 threads)
hit **~1800 chunks/s** and turned 8.4 hours into ~46 minutes, for ~$0 within
Voyage's free tier. Cost was never the issue; the local compute ceiling was.

*Lesson: profile to find the operation that owns your wall-clock, then ask whether
it's even the right tool before optimizing around it.*

## Phase 1 — brute force falls off a memory cliff

Exact (sequential-scan) cosine search is the ground truth every later recall number
is measured against. It's also the baseline that's supposed to get slow. It does —
but not linearly.

| N vectors | table size | p50 query | p95 query |
|---|---|---|---|
| 100k | 546 MB | 110ms | 119ms |
| 500k | 2.7 GB | 779ms | 828ms |
| 1M | 5.3 GB | 1,482ms | 1,554ms |
| 5M | 27 GB | **14,308ms** | 16,586ms |

The ~200ms "too slow for a chat app" line is crossed around **180k** vectors —
brute force is finished as a primary retrieval path well before 1M. But look at
the last row: 1M→5M is 5× the data and **9.7× the latency**. That extra factor
isn't scan arithmetic, it's memory. At 1M the 5.3 GB table still fits in RAM and
page cache; at 5M the 27 GB table doesn't, so every query reads it from disk. The
curve has a knee exactly where the working set stops fitting in memory.

*Lesson: "O(n)" is the optimistic region. The real cliff is the memory hierarchy —
when hot data spills from cache to RAM to disk, the constant factor jumps an order
of magnitude.*

## Phase 2 — IVFFlat, and the index that quietly turned itself off

IVFFlat clusters the vectors into `lists` cells; a query scans the `probes` nearest.
At 5M it's a ~350× speedup over brute force: **recall@10 = 1.0 at ~40ms vs 14.3s.**

Two findings worth more than the headline:

**`lists` is a build-time knob, not a query-speed knob.** At matched recall (1.0)
the query latency is flat across `lists` — 39ms at lists=1000, 37ms at lists=5000 —
because reaching a target recall means scanning a roughly constant number of
candidates (~20–25k), and `lists` just repartitions them. What `lists` *does* move,
steeply, is build time: **637s → 1260s → 2959s** as it goes 1000 → 2236 → 5000. The
famous `lists ≈ √N` is a safe default, not a query win; pushing past it is wasteful.

**The planner will abandon the index on you.** Two rows in the first sweep came back
at ~14 seconds — brute-force latency. I wrote them up as cache eviction. I was
wrong. `EXPLAIN` showed a **parallel sequential scan**: pgvector's IVFFlat cost
estimate grows with `probes`, and past a threshold the planner decides a full scan
is cheaper and silently takes it. Forced with `enable_seqscan = off`, the same
query is a 269ms index scan — 50× faster than the plan Postgres chose on its own.

*Lesson: when a number looks wrong, read the query plan before you invent a cause —
a plausible mechanism that fits half the evidence is how a wrong explanation gets
comfortable.*

## Phase 3 — HNSW, and the benchmark that lied

Here the fixture itself broke, and it's the most important result in the project.

The synthetic corpus — a seeded generator emitting a few near-identical templates
per topic — produced vectors sitting in tight clumps: a query's true top-10 all at
distances 0.338–0.339, with a crowd of near-ties just beyond. On that data:

- **IVFFlat looked perfect** — recall 1.0, trivially. Exhaustive cell scans don't
  care about ties.
- **HNSW looked broken** — recall capped at **0.77 even at ef_search=1000.** Its
  greedy graph walk gets trapped among the near-equal vectors and never reaches the
  specific ids the exact tie-break picked.

That ranking is backwards from reality. On real data HNSW *beats* IVFFlat. A naive
phase-3 table would have "proven" the exact opposite of the truth — because the data,
not the algorithm, was doing the talking. The fix isn't tuning HNSW for hours; it's
fixing the input. So the comparison was rerun on **real Wikipedia (wikitext-103,
500k chunks)**, where neighborhoods are genuinely spread out (top-10 distance spread
~0.06 vs synthetic's ~0.001) and every method's recall climbs gradually.

On honest data, the ranking flipped back:

### Headline — brute force vs IVFFlat vs HNSW at recall@10 ≈ 0.95 (500k real)

| method | config | recall@10 | p50 | vs brute force |
|---|---|---|---|---|
| brute force | exact scan | 1.000 | 889ms | 1× |
| IVFFlat | lists=707, probes=20 | 0.955 | 16ms | 55× |
| **HNSW** | m=16, ef_construction=200, ef_search=40 | 0.949 | **1.5ms** | **~600×** |

HNSW wins by ~10× over IVFFlat at matched recall — the textbook ordering, and the
exact inverse of the synthetic verdict. Same code, same knobs, honest data. It pays
for that speed with a slower build and a fixed graph (no cheap rebuilds); IVFFlat
builds far faster and tunes more easily, which is its own kind of win.

*Lesson: a benchmark's data can decide its winner. If a well-regarded method posts
absurd numbers, suspect the corpus before the method — and diagnose by distance, not
id, to tell a wrong answer from a different-but-equivalent one.*

## Phase 4 — does the tail hold under load?

Phases 1–3 measured one query at a time; a chat app serves many. N threads, each on
its own connection, hammering the live HNSW index (500k, ef_search=40):

| concurrency | p50 | p95 | p99 | throughput |
|---|---|---|---|---|
| 1 | 1.3ms | 2.1ms | 2.2ms | 669 qps |
| 10 | 1.9ms | 3.0ms | 4.8ms | 3,850 qps |
| 50 | 7.0ms | 22.8ms | 35.3ms | 4,551 qps |

**p95 holds** (~3ms) through concurrency=10 while throughput scales ~5.7×. At 50 it
degrades *gracefully* — p95 rises to 23ms, no collapse — as the 12-vCPU VM saturates
and throughput plateaus near 4,550 qps. Classic queueing: throughput maxes around
the core count; past that you buy latency, not QPS. (Warm the index first — a cold
3.9 GB index makes the first measurement clock the disk, not the concurrency.)

---

## So what actually breaks?

Nothing all at once. Different layers tap out at different scales:

1. **~180k vectors** — brute-force latency crosses "too slow for chat."
2. **~5M / 27 GB** — the table exceeds RAM; brute force goes super-linear (disk).
3. **High `probes`** — the query planner silently abandons the IVFFlat index for a
   seq scan unless you force it.
4. **Index memory** — a 38 GB IVFFlat/HNSW build blows past `maintenance_work_mem`
   and `/dev/shm`; and the index can be *larger than the table it indexes*.
5. **The benchmark** — synthetic data inverted the IVFFlat-vs-HNSW ranking. The
   most dangerous break, because it looks like a result.

The retrieval layer itself scales beautifully once you pick the right index: **HNSW
answers a 500k-vector query in ~1.5ms at 95% recall and holds thousands of QPS.**
The engineering is in knowing which wall is coming next — and in trusting a number
only after you've checked what actually produced it.

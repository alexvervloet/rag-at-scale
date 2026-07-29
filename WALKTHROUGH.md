# WALKTHROUGH — the end-to-end journey

The [README](README.md) tells you how to install and smoke-test this. This file is
the *journey*: the whole run from an empty Docker volume to a finished
brute-force-vs-IVFFlat-vs-HNSW comparison, in the order you actually hit it —
every fork in the road, what each stage costs in wall-clock and disk, what breaks
and why, and where this project is the right tool versus the wrong one.

Read it before you start a long run. Most of the expensive mistakes here are
decisions made in the first five minutes (wrong provider, wrong corpus, wrong
`EMBED_DIM`) that only become visible an hour into an embedding job.

*Companions: [PLAN.md](PLAN.md) has the phase definitions and full result tables ·
[WRITEUP.md](WRITEUP.md) has the narrative results · [TEXTBOOK.md](TEXTBOOK.md)
explains why the walls are where they are · [LESSONS.md](LESSONS.md) is the log of
every gotcha, with the reasoning that produced it.*

---

## The map

```
  .env  ──►  docker compose up -d  ──►  check_setup.py
                                             │
                                             ▼
                          ragscale.bench --n 1000        (smoke: does the pipe work?)
                                             │
                                             ▼
              ragscale.bench --grow ...   PHASE 1   exact scan
                                          ├─ writes phase1_<corpus>.csv
                                          └─ writes ground_truth_<corpus>_<n>.json ◄── the contract
                                             │                                          for everything below
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
              ragscale.phase2        ragscale.phase3        ragscale.phase4
                 IVFFlat                  HNSW                concurrency
              (lists/probes)      (m/efc/ef_search)      (threads vs live index)
```

Four things are worth internalising before the first command:

1. **The table is grown in place, never rebuilt.** `corpus.generate()` is
   prefix-consistent (row *i* is the same text regardless of the requested *n*), so
   a sweep 100k → 500k → 1M only ever embeds the delta. Embedding is the entire
   cost of this project; this is what makes the sweep affordable.
2. **Phase 1's ground truth is the contract.** Every recall number in phases 2–3 is
   scored against `results/ground_truth_<corpus>_<n>.json`, keyed to the corpus
   source and the exact row count. Change either and phase 2 will not find its
   reference (or, worse, would score against a stale one).
3. **There is one table and one index at a time.** `chunks` is shared by every
   phase, and `create_ivfflat`/`create_hnsw` each drop the other index first. The
   head-to-head is sequential, not simultaneous.
4. **Everything lands in `results/`, which is gitignored.** Raw logs and CSVs stay
   local; the summary tables get pasted into PLAN.md by hand.

---

## Fork 0 — pick your route before you spend anything

This is the highest-leverage decision in the project. Three routes, all supported
by the same code:

| Route | Settings | Wall clock | Disk | What it gets you |
|---|---|---|---|---|
| **Taste test** | `EMBED_PROVIDER=ollama`, synthetic, N=100k | ~15 min | ~1 GB | The pipeline end to end, a real phase-1 anchor, zero API keys |
| **Honest comparison** ★ | `EMBED_PROVIDER=voyage`, `CORPUS_SOURCE=wiki`, N=500k | ~1 hour | ~12 GB | The actual IVFFlat-vs-HNSW result. This is the route that produced the headline table |
| **Full sweep** | `EMBED_PROVIDER=voyage`, synthetic, N→5M | most of a day | ~70 GB+ | The brute-force memory cliff and the 5M IVFFlat mechanics |

★ **Start here if you only run one thing.** The 5M sweep is impressive and mostly
teaches you about brute force; the 500k *real* run is the one whose conclusions
generalise (see Stage 5 for why the corpus, not the scale, is what makes the ANN
comparison valid).

The three knobs that define a route, all in `.env`:

```bash
EMBED_PROVIDER=voyage    # ollama (local, $0, ~165 chunks/s) | voyage (hosted, ~1800 chunks/s)
EMBED_DIM=1024           # MUST match the model: nomic-embed-text=768, voyage-4=1024
CORPUS_SOURCE=wiki       # synthetic (fast, clusterable, dishonest for ANN) | wiki (real)
```

> **Gotcha — `EMBED_DIM` is baked into the table.** The schema is
> `embedding vector(EMBED_DIM)`, fixed at `CREATE TABLE`. Switching provider (or
> model) mid-project means the next `COPY` fails on a dimension mismatch. Changing
> dimension is a **`--fresh` table and a full re-embed**, and it invalidates every
> earlier number — 768-d and 1024-d results are not comparable (different scan
> cost, table size, index size). Decide the provider once, up front.

> **Gotcha — Voyage's free tier is a token budget, not a request budget.** The full
> 5M synthetic sweep (~140M tokens) fits inside it. The wiki corpus uses 800-char
> chunks (~4× the tokens per chunk of a synthetic one), which is why the same
> provider does ~1800 chunks/s on synthetic and **597 chunks/s** on wiki. Budget by
> tokens, not rows.

---

## Stage 1 — bring up the environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set your route from Fork 0
docker compose up -d          # pgvector (Postgres 17) on localhost:5433
python check_setup.py
```

`check_setup.py` is worth running every time you change `.env` — it catches the
two failures that otherwise surface an hour later: an unreachable embedding
provider, and an `EMBED_DIM` that disagrees with the model (it embeds the string
`"hello"` and compares `len(vec)`).

What the Docker settings are doing, because two of them are load-bearing at scale:

| Setting | Value | Why |
|---|---|---|
| port | `5433:5432` | Avoids clashing with a local Postgres |
| `shared_buffers` | 2 GB | Query-path memory. Not what index builds use |
| `maintenance_work_mem` | 1 GB (server default) | Overridden per-session by phases 2–3 |
| `shm_size` | 12 GB | Parallel HNSW builds allocate a `/dev/shm` segment sized to `maintenance_work_mem`. Too small → `CREATE INDEX` dies with "No space left on device" |

> **Gotcha — Docker Desktop's disk image is its own limit.** The 5M route needs a
> 27 GB table *plus* a 38 GB index *plus* build temp space. The default VM disk
> (60 GB here) is not enough; it was raised to 150 GB. Check this before, not
> during, an 80-minute index build.

> **Small inaccuracy to know about:** `check_setup.py`'s docstring says it makes no
> paid call. With `EMBED_PROVIDER=voyage` it does issue one real (one-token, free-tier)
> API request — that's how it verifies reachability and dimension.

**Full reset**, when you want to start clean: `docker compose down -v` (the `-v`
destroys the `pgdata` volume and therefore every loaded vector).

---

## Stage 2 — the smoke run

```bash
python -m ragscale.bench --n 1000
```

Seconds, not minutes. It exercises the whole pipe — generate → embed → `COPY` →
query — and finishes by printing top-3 hits for a human-readable sanity question.
On the synthetic corpus those hits should obviously be on-topic. On the wiki
corpus they will look *unrelated* to the sanity question, and that is expected:
the sanity query is about billing and the corpus is Wikipedia.

**Branch:** if all you wanted was to see the mechanism work, stop here. Everything
after this point is a measurement campaign.

---

## Stage 3 — phase 1, the exact-search baseline

```bash
python -m ragscale.bench --grow 100000,500000,1000000 --repeats 1
```

Each step embeds only the new rows, runs 100 held-out queries with an exact
sequential scan, and writes two artifacts: a row in `phase1_<corpus>.csv`, and the
frozen top-10 ids per query in `ground_truth_<corpus>_<n>.json`. Ground truth is
captured *during* the timed pass rather than recomputed later, because recomputing
it invites silent drift (LESSONS #4) — and because at 5M a second full scan of the
table would double the phase's wall time.

What to expect (voyage, 1024-d, synthetic):

| N | load (step) | table | p50 | p95 |
|---|---|---|---|---|
| 100k | 57s | 546 MB | 110ms | 119ms |
| 500k | 220s | 2.7 GB | 779ms | 828ms |
| 1M | 277s | 5.3 GB | 1,482ms | 1,554ms |
| 5M | ~2,290s | 27 GB | 14,308ms | 16,586ms |

The 5M row is not just "5× more rows": it's 9.7× the latency, because 27 GB no
longer fits in RAM and every query reads from disk. That knee is the whole reason
phases 2–3 exist.

> **Gotcha — the table only ever grows.** `load_corpus` returns early when
> `existing >= n`. Ask for `--n 1000` against a table holding 5M rows and it will
> happily benchmark all 5M and record the row as `n=5000000`. `--fresh` is the only
> way down, and it only applies to the *first* step of a `--grow` sweep.

> **Gotcha — running `bench` destroys your index.** `run_point` calls
> `drop_indexes()` before measuring, by design (exact search *is* the absence of an
> index). Re-run phase 1 "just to check something" after a 49-minute IVFFlat build
> and you have thrown the build away.

> **Gotcha — a re-measured point reports `load_s = 0.0`.** If the rows were already
> there, no load happened. That's the `0.0` in the 5M row of
> `results/phase1_synthetic.csv`; the real cumulative embed time (~47 min to 5M)
> lives in the logs, not the CSV.

> **Budget note.** At 5M, a single pass of 100 exact queries is ~24 minutes.
> `--repeats 3` is three times that. Keep repeats at 1 for the big brute-force
> points and spend the repeats on the fast indexed phases instead.

---

## Stage 4 — phase 2, IVFFlat

```bash
python -m ragscale.phase2 --lists 1000,2236,5000 --probes 1,5,10,20,50,100
```

For each `lists` it builds the index, then sweeps `probes`, scoring recall@10
against phase 1's frozen ids and recording p50/p95 plus the one-time build cost.
Two mechanisms in this script exist because of measurement bugs, and both are worth
understanding rather than deleting:

**It raises `maintenance_work_mem` (default `--maintenance-mem 8GB`).** Index
builds are a different resource regime from queries: IVFFlat's k-means needs memory
scaling with `lists × dim`, not with the data you already loaded fine. The first
build failed instantly with `memory required is 1438 MB, maintenance_work_mem is
1024 MB` (LESSONS #7). Measured need at 5M × 1024-d: ~1.4 GB at lists=2236, ~5.9 GB
at lists=5000.

**It forces the index with `SET enable_seqscan = off`.** pgvector's IVFFlat cost
estimate grows with `probes`; past a threshold the planner decides a parallel
sequential scan is cheaper and silently takes it, handing you brute-force latency
from an "indexed" query (14.3s vs 269ms for the same query forced — LESSONS #9).
Without this line the sweep measures the fallback, not the index.

> That fallback is not a benchmark artifact — it is a **production** pgvector gotcha.
> In a real deployment at high `probes` on a large table, the same thing happens and
> nothing warns you. `EXPLAIN` is the only ground truth for what actually ran.

Useful flags and behaviours:

- `--measure-only` — re-measure the *existing* index without paying for a rebuild.
  Pass the live index's `lists` as the label; `build_s` is recorded as 0.
- `probes > lists` combos are skipped automatically.
- The CSV is **appended**, not truncated. Re-running a sweep stacks new rows onto
  the old ones (that's why `results/` has `phase2_raw.csv` and `phase2_prev.csv`).
  Move the old file aside before a re-run you intend to treat as authoritative.
- Recall loads `ground_truth_<corpus>_<count(*)>.json`. No phase-1 run at exactly
  this row count and corpus → `FileNotFoundError`. That's the contract working.

What the numbers say: every config reaches recall 1.0 on synthetic data, at ~40ms
versus brute force's 14.3s (~350×). But at *matched recall*, latency is flat across
`lists` (39ms at 1000 vs 37ms at 5000) while build time goes 637s → 1260s → 2959s.
**`probes` is the recall/latency knob; `lists` is a build-cost decision.** `√N` is a
safe default, not a query-speed win.

---

## Stage 5 — phase 3, HNSW *and the corpus fork*

**This is the most important branch in the whole project, and it is not optional.**

If you run HNSW against the synthetic corpus, you will get a clean-looking table
that is *backwards*. The seeded generator emits a few near-identical templates per
topic, so a query's true top-10 sit at nearly identical distances (0.338–0.339) in
a crowd of near-ties. IVFFlat scans whole cells exhaustively and doesn't care about
ties, so it reads as perfect. HNSW's greedy graph walk gets trapped among the
near-equal vectors and **ceilings at 0.77 recall even at `ef_search=1000`**. A naive
write-up would conclude IVFFlat beats HNSW — the exact inverse of reality
(LESSONS #11).

The fix is not to tune HNSW for hours. It's to fix the fixture:

```bash
# 1. switch the corpus (and rebuild — the table is shared)
#    .env: CORPUS_SOURCE=wiki
python -m ragscale.bench --n 500000 --fresh     # re-embeds + new ground truth for wiki@500k

# 2. IVFFlat on real data (√500k ≈ 707)
python -m ragscale.phase2 --lists 707,2000 --probes 1,5,10,20,50,100

# 3. HNSW on the same rows
python -m ragscale.phase3 --m 16,32 --ef-construction 64,200 --ef-search 10,40,100,200
```

Step 1 is not skippable. Switching `CORPUS_SOURCE` does not touch the database, so
without `--fresh` you would silently mix wiki rows on top of synthetic ones, and
the ground truth for the new corpus wouldn't exist.

On real data the ranking flips back to the textbook ordering, at recall ≈ 0.95:

| method | config | recall@10 | p50 | vs brute | build |
|---|---|---|---|---|---|
| brute force | exact scan | 1.000 | 889ms | 1× | — |
| IVFFlat | lists=707, probes=20 | 0.955 | 16.2ms | 55× | 72s |
| **HNSW** | m=16, efc=200, ef_search=40 | 0.949 | **1.5ms** | **~600×** | 381s |

Practicalities of the wiki path:

- The first run downloads wikitext-103 via HF `datasets` (a few hundred MB, cached
  afterwards). Expect an unauthenticated-rate-limit warning; it's harmless, or set
  `HF_TOKEN`.
- Embedding is slower per chunk than synthetic — 597 chunks/s, ~14 minutes for
  500k — because the chunks are ~4× the tokens. The progress bar oscillates wildly
  (concurrent batches land in bursts); the final rate is the honest one.
- **Queries are passages, not questions.** Wiki held-out queries are the first 100
  chunks of the *validation* split — genuinely held out, but stylistically unlike
  user questions. Recall numbers are about vector geometry, not question answering.
- `ef_search < k` combos are skipped.
- Build cost is the real HNSW tax: 144s (m=16, efc=64) → 594s (m=32, efc=200) at
  500k, versus 72s for IVFFlat. Higher `m`/`ef_construction` raise the recall
  ceiling and the recall-per-`ef_search`; `ef_search` is the query knob.
- Both indexes are ~3.9 GB. An ANN index can be **larger than the table it
  indexes** (pgvector re-stores every vector), so capacity planning is
  table + index + build temp.

---

## Stage 6 — phase 4, concurrency

```bash
python -m ragscale.phase4 --concurrency 1,10,50 --ef-search 40 --per-thread 200
```

Runs against whatever index is currently live (HNSW, if you just finished stage 5).
It warms the index with three full passes first — a cold 3.9 GB index makes the
first level measure disk instead of concurrency — then fires N threads, each on its
own connection.

| concurrency | p50 | p95 | p99 | throughput |
|---|---|---|---|---|
| 1 | 1.3ms | 2.1ms | 2.2ms | 669 qps |
| 10 | 1.9ms | 3.0ms | 4.8ms | 3,850 qps |
| 50 | 7.0ms | 22.8ms | 35.3ms | 4,551 qps |

p95 holds ~3ms through 10 concurrent queries while throughput scales ~5.7×; at 50
it degrades *gracefully* rather than collapsing, as the 12-vCPU VM saturates.
Classic queueing — throughput maxes near the core count, and past that you buy
latency, not QPS.

- Threads, not asyncio: psycopg is blocking and releases the GIL on I/O, so threads
  are the honest simple choice here.
- Unlike phases 2–3, this CSV is **truncated** on each run.
- Caveat: the load generator shares the machine with the database. At concurrency
  50 the client is competing for the same 12 vCPUs, so treat the top row as
  "single-box saturation", not a pure server number.

---

## Every fork, in one table

| Fork | Options | How to choose |
|---|---|---|
| Embedding provider | `ollama` / `voyage` | Ollama for <100k or no API key. Voyage above that — 165 vs 1800 chunks/s is hours vs minutes |
| Dimension | 768 / 1024 | Dictated by the model. Never mix; changing it means a fresh table |
| Corpus | `synthetic` / `wiki` | Synthetic for latency, throughput, and memory mechanics. **Wiki for anything that ranks two ANN methods against each other** |
| Scale | 100k → 5M | 100k is a real anchor; sub-10k is fixed overhead in a costume. 500k is enough for index comparison; 5M is for the memory cliff |
| `lists` | √N vs N/1000 | Build-cost decision, not a query-speed one. √N is the safe middle |
| `probes` / `ef_search` | the query knob | Sweep it; the knee is where recall first saturates. Past that you pay latency for nothing |
| Force the index? | `enable_seqscan=off` | Force it when *characterising the index*. Leave it default when you want to know what production would actually do |
| Rebuild or reuse | `--measure-only` | Re-measuring a 49-minute build is free; do that before rebuilding |

### Plugging in your own corpus

The wiki path is the template. Add `ragscale/corpus_mine.py` exposing
`generate(n, seed) -> Iterator[(label, text)]` and `generate_queries(n, seed) -> list[str]`,
branch to it on `CORPUS_SOURCE` in `corpus.py` (both functions), then `--fresh` and
re-run phase 1 to mint ground truth. Two requirements, both load-bearing:
**prefix-consistency** (row *i* is stable across different *n*, or grow-in-place
re-embeds everything) and **genuinely held-out queries** (never a corpus row, or
recall is meaningless).

---

## Gotcha index

| Symptom | Cause | Fix |
|---|---|---|
| `expected 768 dimensions, not 1024` on COPY | Table created under the other provider | `--fresh` and re-embed; keep dimension fixed per campaign |
| `FileNotFoundError: ground_truth_<corpus>_<n>.json` | No phase-1 run at this corpus + row count | Run `bench` at exactly this N first |
| `ProgramLimitExceeded: memory required is …` | Build memory ≠ query memory | Raise `--maintenance-mem` (scales with `lists`, `dim`) |
| `CREATE INDEX` → "No space left on device" on /dev/shm | Parallel build's DSM segment > `shm_size` | Raise `shm_size` in docker-compose above `maintenance_work_mem` |
| Indexed query takes ~14s | Planner abandoned the index at high `probes` | `SET enable_seqscan = off`, and check `EXPLAIN (ANALYZE, BUFFERS)` |
| Benchmark ignored `--n`, used the whole table | `load_corpus` returns early when `existing >= n` | `--fresh`, or ask for a larger N |
| Index vanished between phases | `bench` and both `create_*` helpers call `drop_indexes()` | Don't re-run phase 1 mid-campaign; use `--measure-only` |
| Latencies spike on the first rows of a sweep | Cold pages in an index larger than RAM | Warm the *whole* query set, not one query (phases 2–4 do) |
| Duplicate/contradictory rows in a phase CSV | Phase 2/3 CSVs append | Move the old file aside before an authoritative re-run |
| HNSW recall ceilings around 0.77 | Synthetic near-duplicate geometry, not HNSW | Switch to `CORPUS_SOURCE=wiki`. Do not tune |
| Tiny-N latencies look great (3–8ms) | Fixed per-query overhead dominates the scan | Get one anchor at ≥100k before drawing any curve |

---

## Best and worst use-cases

### What this project is genuinely good for

- **Capacity planning for single-node pgvector.** Concrete answers to "at what N
  does brute force stop working" (~180k for a chat-latency budget), "how big is the
  index" (can exceed the table), and "how much RAM before the cliff" (the knee is
  where the working set leaves memory).
- **Choosing between IVFFlat and HNSW with numbers instead of folklore**, including
  the build-cost side of that choice, which most comparisons omit.
- **Building intuition for the knobs.** `lists` vs `probes`, `m`/`ef_construction`
  vs `ef_search` — measured at *matched recall*, which is the only comparison that
  exposes which knob is on which cost axis.
- **A worked example of benchmark hygiene**: frozen ground truth, one real-scale
  anchor, forcing the code path you claim to be measuring, and a fixture whose
  geometry can invert your conclusion.
- **Teaching and interview prep.** The failures are more instructive than the
  results, and they're all written down in LESSONS.md with the wrong reasoning
  included.

### Where each retrieval strategy actually belongs

| Use | Good fit | Bad fit |
|---|---|---|
| **Brute force** | <100k vectors; you need exact answers; the corpus churns constantly; you're generating ground truth | Anything interactive above ~180k vectors |
| **IVFFlat** | Frequent rebuilds; write-heavy corpora; tight build-memory budgets; you want easy, legible tuning | Latency-critical read paths — ~10× slower than HNSW at matched recall |
| **HNSW** | Read-mostly corpora; latency-critical retrieval; you can afford a slow build | Rapidly churning data (no cheap incremental rebuild); very tight build-time windows |

### What this project is *not* for

Be honest about the edges before quoting these numbers anywhere:

- **Not a vector-database shootout.** It's pgvector on one box. Nothing here says
  anything about Qdrant, Milvus, Pinecone, or a distributed anything.
- **Not a test of filtered or hybrid search.** Every query is pure top-k cosine. Add
  a `WHERE` clause or BM25 fusion and the planner interactions — and the recall
  story — change completely.
- **No updates, deletes, or churn.** The corpus is append-only. Index degradation
  under churn (IVFFlat's stale centroids, HNSW's rebuild cost) is exactly the
  operational question this doesn't answer.
- **No quantization, no multi-tenancy, no sharding, no replication.**
- **Recall is scored by id, not distance.** On tie-heavy data that punishes a method
  for returning a different-but-equally-close vector. That's *why* the synthetic
  corpus produced an inverted ranking — and it's a caveat for any recall number
  measured this way.
- **Nothing about generation.** No LLM serving, batching, or throughput. This is the
  retrieval layer only.
- **Single machine, one hardware profile.** 12 vCPUs, 2 GB `shared_buffers`. The
  *shapes* of these curves travel; the absolute milliseconds do not.

---

## Where to go next

If you want to extend rather than reproduce, the three highest-value additions —
each one squarely in the "not for" list above — are **filtered search** (a `WHERE`
clause alongside the ANN scan, where pgvector's planner gets genuinely
interesting), **churn** (measure recall decay as rows are updated and deleted
without a rebuild), and **distance-based recall** as a second metric next to the
id-based one, so a benchmark can tell a wrong answer from a different-but-equivalent
one without a human reading the distances.

*Run it: [README.md](README.md) · The numbers: [WRITEUP.md](WRITEUP.md) · Why the
walls are there: [TEXTBOOK.md](TEXTBOOK.md) · Every gotcha in order:
[LESSONS.md](LESSONS.md) · Phase definitions: [PLAN.md](PLAN.md)*

# LESSONS

Engineering lessons from building a pgvector benchmark at 1–5M vectors. This is a
measurement project, so most of these are about **not fooling yourself with your
own numbers** — a metric that measures the wrong subsystem, an extrapolation that
lies, a cost that reshapes the whole harness. Each is tied to the concrete thing
that taught it. Kept as a running log; it grows as the phases land.

---

## 1. Know which subsystem your benchmark number actually measures

The phase-1 table has a "load time" column, and it reads like a pgvector ingest
number. It isn't. Loading 100k chunks took 604s at ~165 chunks/s — and that rate
is set entirely by the local embedding model (Ollama `nomic-embed-text`), not by
Postgres. The binary `COPY` into pgvector is a small fraction of the wall time;
the ten minutes is almost all embedding. A "load time" column in a *database*
benchmark that actually clocks your embedding server is a quiet lie, and it would
have shipped into the write-up unquestioned if I hadn't gone looking for why the
number was so large.

Takeaway: before you trust a benchmark figure, name the component it actually
measures. An end-to-end timing folds several subsystems together, and the one that
dominates is often not the one the column is named after.

## 2. A benchmark needs a real-scale anchor — tiny-N numbers are fixed cost wearing a costume

The smoke runs (2k–5k rows) showed 3–8ms query latencies, and I extrapolated them
to tell the user the "too slow for chat" (~200ms) crossing was around 100–150k.
That was wrong, and I had to walk it back one message later. Those small-N numbers
are dominated by *fixed* per-query overhead — client round-trip, query planning,
result marshalling — not by the sequential scan that's supposed to grow with N. The
first real anchor at 100k (91ms p50) reset the slope to ~0.9ms per 1k rows and
moved the crossing to ~220k.

Takeaway: the cost you're studying (here, the O(n) scan) stays buried under
constant overhead until N is genuinely large. Don't predict, tune, or draw the
curve from micro-scale points — get one anchor at real scale before you believe any
slope.

## 3. Shape the data generator so the expensive step runs exactly once

Embedding is the entire cost of this project — 5M rows is ~8.4 hours of it. The
phase-1 sweep visits 100k → 500k → 1M → 5M, and the naive harness re-embeds the
whole corpus at each point: 6.6M embeddings for 5M rows of data. Making
`corpus.generate()` **prefix-consistent** — row *i* is identical regardless of the
requested *n* — lets the harness grow the table in place, embedding only the delta
at each step. The whole sweep then embeds each chunk exactly once (5M total, not
6.6M), and re-running a point is free.

Takeaway: when one operation dominates the cost, make it a property of your data
model that nothing repeats that operation. Prefix-consistency wasn't free prose in
the generator's docstring — it's what makes the sweep affordable.

## 4. Freeze the ground truth when you take it, don't plan to recompute it

Phase 1's exact (sequential-scan) results are the reference every later recall
number is scored against. The harness writes the top-k ids for all 100 held-out
queries to disk *at the moment it measures them*, rather than deferring to
"recompute the exact results during phase 2." Recomputing later invites silent
drift — a reloaded table with different row ids, a changed query seed, a corpus
regenerated at a different N — and any of those quietly corrupts every recall
figure that depends on it. A comparison is only honest if its reference was frozen
at the point of measurement.

Takeaway: ground truth is an artifact to capture, not a computation to repeat.
Persist the reference the first time you have it, keyed to the exact inputs that
produced it.

## 5. The bottleneck decides the architecture — so measure it, then attack it

The whole harness is shaped by one fact: embedding, not pgvector, is the
expensive step. Local Ollama did ~165 chunks/s, which makes 5M rows an ~8.4-hour
job — long enough that it dictated every design choice (grow-in-place so we embed
once, run in the background, ask before committing the hours). When that became
the wall, the fix wasn't a cleverer batch loop against the same local model — the
local model *is* the compute ceiling. Switching to a hosted embedder (Voyage) with
many concurrent batched requests took throughput from ~165 to ~1800 chunks/s and
turned 8.4 hours into ~46 minutes, for ~$0 within the free tier.

But two sub-lessons made it actually work. First, **the speedup is in the
concurrency, and its knobs must be measured, not guessed**: at the naive 128/8
batch/workers it did ~450 chunks/s; a 5-minute probe of a few settings found
768/32 at ~1800/s, a 4× swing that's the difference between 45 minutes and 3
hours. Second, **swapping the embedder is not a drop-in** — Voyage's 1024-d vs
nomic's 768-d changes scan cost, table size, and index size, so the entire
phase-1 table had to be restated at the new dimension; a mixed-dimension
comparison would have been meaningless.

Takeaway: profile to find the one operation that owns your wall-clock, then ask
whether it's even the right *tool* before optimizing around it — and when you
swap the tool, re-baseline everything the swap silently changed.

## 6. Brute force hits a memory cliff, not a linear wall

The phase-1 thesis was "exact search grows linearly with N; find where it crosses
too-slow." It does grow linearly — right up until it doesn't. Latency ran 110ms →
779ms → 1.48s across 100k → 500k → 1M (roughly linear), then **jumped to 14.3s at
5M — 9.7× the latency for 5× the rows.** The extra factor isn't scan arithmetic;
it's memory. At 1M the 5.3 GB table still fits in `shared_buffers` + OS page
cache, so the scan is CPU/bandwidth bound in RAM. At 5M the table is 27 GB, blows
past memory, and every query reads it from disk (the `IO` waits were plainly
visible in `pg_stat_activity`, scanned across parallel workers). So the curve has
a knee exactly where the working set stops fitting in RAM.

Takeaway: for anything that scans a large structure, "O(n)" is the optimistic
region. The real cliff is the memory hierarchy — the moment the hot data spills
from cache to RAM to disk, the constant factor changes by an order of magnitude.
When you extrapolate a scaling curve, ask where it crosses each memory boundary,
not just how it grows in the fast region. (And it reframes phases 2–3: an ANN
index earns its keep partly by keeping the *working set* small enough to stay in
memory — index size can matter as much as index algorithm.)

## 7. Building an index is a different resource regime than querying it

The very first IVFFlat build in phase 2 failed instantly:
`ProgramLimitExceeded: memory required is 1438 MB, maintenance_work_mem is
1024 MB`. Nothing about the *data* had changed — the 5M rows had loaded and
queried fine — but the build has its own appetite. IVFFlat's k-means samples on
the order of `lists × 50` vectors and needs working memory that scales with the
index parameters (`lists`) and the dimension, not with how much RAM the queries
use. And `maintenance_work_mem` is a *separate* knob from the query-path memory
(`shared_buffers`, `work_mem`); the 1 GB we'd set was sized for the data, not for
clustering 5M × 1024-d into a few thousand cells. The fix is a one-line,
build-time-only bump (`SET maintenance_work_mem = '4GB'`), and the requirement
grows with `lists` — the same knob that will make the sweep's bigger builds cost
even more.

Two things kept this cheap, and both generalize. First: **it failed loudly and
immediately**, with the exact numbers (needed vs. available) in the message — so
the fix was arithmetic, not a debugging session. A build that instead ran 20
minutes and then OOM-killed the server would have been far worse; a clear
resource error up front is a gift. Second: **I probed a single build before
launching the three-build sweep.** The failure surfaced in seconds against one
`lists` value, not partway through a long multi-build run — the same
anchor-before-you-commit discipline from §2, applied to an expensive operation
instead of a measurement.

Takeaway: an index build's cost model (memory *and* time) is its own thing, keyed
to the index parameters, and defaults that suit the data will not automatically
suit the build. Size the build knobs to `(rows, dim, index params)` deliberately —
and dry-run one build before sweeping many.

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

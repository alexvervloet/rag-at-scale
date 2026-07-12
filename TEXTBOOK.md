# RAG at Scale: What Actually Breaks at a Few Million Vectors

*This is the textbook companion for the rag-at-scale project. The [README](README.md) tells you how to run it; the [WRITEUP](WRITEUP.md) gives the numbers with a chart per phase; [LESSONS.md](LESSONS.md) logs every gotcha. This piece is the lecture: it explains why the walls are where they are, so the results read as physics rather than trivia. It assumes you already understand basic retrieval-augmented generation (embed text into vectors, find the nearest ones to a query, feed them to a model). Here we ask the question a tutorial corpus of four files can never answer: what happens when there are five million of them?*

---

## 1. The gap between a demo and a system

Almost every RAG tutorial, including good ones, proves its ideas on a tiny corpus. A handful of markdown files, a few dozen chunks, a brute-force scan that returns in a millisecond because there is nothing to scan. This is the right way to learn the mechanics. It is also, quietly, a lie of scale, because every expensive thing in a real retrieval system is invisible at that size. The scan is free. The index is unnecessary. Memory is a non-issue. The database's query planner never has an interesting decision to make.

The moment you go from four files to a few million chunks, all of that changes, and it does not change gradually or all at once. It changes as a sequence of separate walls, each one a different subsystem reaching its limit, each arriving at a different scale. This project's whole purpose is to hit those walls on purpose, with a stopwatch, and write down where each one is. The setup is deliberately ordinary: pgvector (the vector extension for Postgres) running in Docker on a single laptop, embeddings from a hosted provider, one machine doing everything. No cluster, no exotic hardware. That ordinariness is the point, because the walls you meet here are the walls most teams meet, and meeting them on a laptop makes the mechanism legible in a way a 200-node cluster never would.

The deliverable of a project like this is not code. It is numbers you can trust: recall (did you find the right vectors?), latency at the median and the tail, index build time, and memory. Everything below is measured, and the measurements are more interesting than any of them looks at first, because several of them are traps.

## 2. The first wall is not in the database at all

Before any query runs, you have to turn millions of chunks of text into millions of vectors, and this is where the project's first lesson landed, before the interesting part even began.

The original plan was to keep everything free and local: run the embedding model on the laptop with Ollama, pay nothing, send no data anywhere. It worked, and it was the wrong tool, for a reason that had nothing to do with correctness. The local model embedded at roughly 165 chunks per second, and it was completely bottlenecked by the laptop's own compute. At that rate, embedding five million chunks would have taken about eight and a half hours of the machine doing nothing else. That single number reshaped the whole project. Switching to a hosted embedding provider, which accepts many batched requests at once, pushed throughput to roughly 1,800 chunks per second and turned eight and a half hours into about forty-six minutes, and because the full sweep fit inside the provider's free token tier, it stayed effectively free anyway.

The lesson here is one of the most transferable in all of performance work, and it is worth stating plainly because it is so often skipped: **profile to find the operation that owns your wall-clock time, then ask whether it is even the right tool, before you optimize anything around it.** Cost was never the real constraint. The local compute ceiling was. A great deal of optimization effort in the world is spent making the wrong operation faster, and the way you avoid that is by measuring where the time actually goes before you touch anything. In this project, the slowest thing was not the database everyone expected to be the bottleneck. It was the embedding step, and it was slow for a reason (a single machine's compute) that no amount of database tuning would ever have fixed.

## 3. Brute force falls off a memory cliff, not an O(n) slope

With vectors in the database, the baseline is the simplest possible retrieval: an exact scan. Compare the query vector against every stored vector, sort, return the top few. This is the ground truth that every later, cleverer method gets measured against, because it is guaranteed correct by construction. It is also the baseline that is supposed to get slow, and the interesting question is not whether it gets slow but *how*.

Here is what the scan actually did as the corpus grew:

| Vectors | Table size | Median query | 95th-percentile query |
|---|---|---|---|
| 100,000 | 546 MB | 110 ms | 119 ms |
| 500,000 | 2.7 GB | 779 ms | 828 ms |
| 1,000,000 | 5.3 GB | 1,482 ms | 1,554 ms |
| 5,000,000 | 27 GB | 14,308 ms | 16,586 ms |

Two things in this table matter more than the headline "it gets slow." First, the line where brute force becomes unusable for an interactive application (roughly 200 milliseconds, the threshold past which a chat interface starts to feel laggy) is crossed at around 180,000 vectors. That is far, far below a million. Brute force is finished as a primary retrieval path long before you reach the scale people imagine when they say "large."

Second, and more important, look at the jump from one million to five million. That is five times the data, but the latency did not grow five times. It grew about 9.7 times. That extra factor is the whole lesson, and it is not scan arithmetic. It is memory. At one million vectors, the 5.3 GB table still fits comfortably in the machine's RAM and page cache, so every query reads from memory. At five million, the 27 GB table does not fit, so every query has to read the bulk of it from disk, and disk is slower than memory by an order of magnitude. The performance curve has a knee, and the knee sits exactly where the working set stops fitting in memory.

This is the deepest and most portable idea in the whole project, so it deserves to be said carefully. **"O(n)" describes the optimistic region, where your data lives in fast memory and the constant factor is small.** The real cliff is the memory hierarchy. Modern computers are built as a stack of storage tiers (CPU cache, then RAM, then disk), each one larger and dramatically slower than the one above it, and an algorithm's complexity class tells you nothing about which tier it is running in. When your hot data spills from one tier down to the next, the constant factor in front of your nice clean big-O jumps, and the curve bends upward in a way the complexity analysis never predicted. A lot of "mysterious" performance cliffs in real systems are exactly this: not an algorithm changing, but data crossing a memory boundary.

## 4. IVFFlat, and the index that quietly turned itself off

Brute force having tapped out, the answer is an approximate index: a data structure that trades a little accuracy for an enormous speed gain by not looking at every vector. The first one this project tests is IVFFlat, which works by clustering the vectors into a set of cells (the `lists` parameter) and, at query time, only scanning the handful of cells nearest the query (the `probes` parameter). At five million vectors it delivered a roughly 350-fold speedup over brute force, reaching perfect recall at around 40 milliseconds against the scan's 14 seconds. That is the payoff the memory cliff was setting up.

Two findings from tuning it are worth more than that headline number.

The first is about which knob does what, and it corrects a common misreading. The `lists` parameter (how many cells to divide the vectors into) is widely tuned with a rule of thumb of roughly the square root of the number of vectors. But at matched recall, changing `lists` barely moved query latency at all: 39 milliseconds at 1,000 lists versus 37 milliseconds at 5,000. The reason is that reaching a target recall means scanning a roughly constant number of candidate vectors regardless of how you partition them, and `lists` just changes the partitioning. What `lists` *does* move, steeply, is build time: 637 seconds, then 1,260, then 2,959, as it climbs from 1,000 to 5,000 cells. So `lists` is a build-cost decision, not a query-speed knob, and pushing it past the safe default buys you nothing at query time while costing you dearly at build time. The knob that actually trades recall for latency is `probes`.

The second finding is the kind of thing that only surfaces at scale, and it is a genuine trap. In the first sweep, two configurations came back at around 14 seconds, which is brute-force latency, on an indexed query. The tempting explanation was cache eviction, and it was wrong. Reading the query plan (`EXPLAIN`, the command that shows how the database actually intends to run a query) revealed a parallel sequential scan: the database's cost estimator, which weighs the predicted cost of using the index against the predicted cost of just scanning everything, had decided at high `probes` values that a full scan was cheaper, and silently took it. Forcing the index instead ran the same query in 269 milliseconds, roughly 50 times faster than the plan the database chose on its own.

The lesson is a debugging discipline: **when a number looks wrong, read what actually produced it before you invent a cause.** The cache-eviction theory was plausible, and plausibility is exactly the danger, because a plausible mechanism that fits half the evidence is how a wrong explanation gets comfortable and stops you looking further. The query plan was the ground truth, and it said something the intuition never would have. At scale, "we're using the index" is an assumption to verify, not a fact to rely on, because the database is allowed to change its mind about your index whenever its cost model says so.

## 5. HNSW, and the benchmark that lied

The second approximate index is HNSW, which organizes vectors into a navigable graph and answers a query by walking the graph greedily toward the target. On real workloads it is generally the strongest option on the speed-versus-accuracy frontier. This project's HNSW phase produced the single most important result in the whole effort, and it is a result about the benchmark itself rather than about HNSW.

The synthetic corpus used in the early phases was generated from a small set of near-identical templates per topic. That produced vectors sitting in extremely tight clumps: a query's true top ten would all be at nearly the same distance, with a crowd of near-ties just beyond. On that data, the two indexes ranked backwards from reality. IVFFlat looked perfect, because it exhaustively scans whole cells and does not care about ties. HNSW looked broken, with recall stuck at 0.77 even when tuned aggressively, because its greedy graph walk gets trapped among the near-equal vectors and never reaches the specific ones the exact tie-break happened to pick.

That ranking is the exact inverse of the truth. A naive write-up would have "proven" that IVFFlat beats HNSW, which is the opposite of the real-world ordering, and it would have been the data doing the talking, not the algorithms. The fix is not to spend hours tuning HNSW to overcome bad data. The fix is to fix the data. Rerunning the comparison on a real corpus (a slice of Wikipedia, where neighborhoods are genuinely spread out rather than clumped) flipped the ranking back to reality. On honest data, at matched recall of about 0.95:

| Method | Config | Recall@10 | Median | vs brute force |
|---|---|---|---|---|
| Brute force | exact scan | 1.000 | 889 ms | 1x |
| IVFFlat | lists=707, probes=20 | 0.955 | 16 ms | 55x |
| HNSW | m=16, ef_construction=200, ef_search=40 | 0.949 | 1.5 ms | ~600x |

HNSW wins by roughly ten times over IVFFlat at the same recall, which is the textbook ordering and the precise opposite of what the synthetic data claimed. Same code, same knobs, honest data. It pays for that speed with a slower build and a fixed graph that cannot be cheaply rebuilt, while IVFFlat builds far faster and tunes more easily, which is its own kind of advantage.

The lesson is the one every benchmark author eventually learns the hard way: **a benchmark's data can decide its winner.** If a well-regarded method posts absurd numbers, suspect the corpus before you suspect the method. And there is a sharp diagnostic tucked inside this one: the synthetic failure looked like HNSW returning *wrong* answers, but it was really returning *different, equally-good* answers among a crowd of ties. Telling those two apart means measuring by distance (how close is the returned vector to the query?) rather than by identity (is it the exact vector the exact scan picked?). On clumped data, "a different vector at the same distance" is not an error, and a benchmark that scores by identity will punish a correct method for it.

## 6. Does the tail hold under load?

The final phase asks the question the first four skipped: everything above measured one query at a time, but a real application serves many at once. Running many threads against the live HNSW index, each on its own connection:

| Concurrency | Median | 95th pct | 99th pct | Throughput |
|---|---|---|---|---|
| 1 | 1.3 ms | 2.1 ms | 2.2 ms | 669 qps |
| 10 | 1.9 ms | 3.0 ms | 4.8 ms | 3,850 qps |
| 50 | 7.0 ms | 22.8 ms | 35.3 ms | 4,551 qps |

Through ten concurrent queries, the tail latency holds around 3 milliseconds while throughput scales almost six-fold. At fifty, it degrades, but gracefully: the tail rises to about 23 milliseconds and throughput plateaus near 4,550 queries per second rather than collapsing. This is textbook queueing behavior. Throughput maxes out around the machine's core count (twelve virtual CPUs here), and once you are past that point, adding load buys you latency, not throughput, because the work is already fully packed onto the available cores. There is a small operational footnote worth keeping: warm the index first, because a cold multi-gigabyte index makes the very first measurement clock the disk rather than the concurrency, which is the memory-hierarchy lesson from section 3 sneaking back in.

## 7. So what actually breaks?

Nothing breaks all at once, which is the whole point. Different layers tap out at different scales, and knowing which wall is coming next is most of the engineering:

1. Around **180,000 vectors**, brute-force latency crosses the "too slow for chat" line. This is far earlier than intuition suggests.
2. Around **five million vectors / 27 GB**, the table exceeds RAM and brute force goes super-linear, because it is now reading from disk. This is a memory-hierarchy cliff, not a complexity-class one.
3. At **high probe counts**, the database's query planner silently abandons the IVFFlat index for a sequential scan unless you force it. The index being present does not guarantee the index being used.
4. At **index build time**, a large approximate index can exceed the memory budgets for building it, and can end up larger than the table it indexes.
5. In **the benchmark itself**, synthetic data inverted the ranking of two indexes. This is the most dangerous break of all, because it does not look like a break. It looks like a result.

The reassuring conclusion underneath all of it is that the retrieval layer scales beautifully once you choose the right index: HNSW answered a 500,000-vector query in about 1.5 milliseconds at 95 percent recall and held thousands of queries per second. The hard part was never the retrieval. The hard part is knowing which wall is coming next, and trusting a number only after you have checked what actually produced it. Those two habits (anticipate the next limit, and interrogate every surprising measurement) are the real transferable skill, and they generalize far beyond vectors and databases to any system you push past the scale it was demonstrated at.

---

*Run it: [README.md](README.md) · The numbers with charts: [WRITEUP.md](WRITEUP.md) · Phase definitions and full tables: [PLAN.md](PLAN.md) · Every gotcha in order: [LESSONS.md](LESSONS.md)*

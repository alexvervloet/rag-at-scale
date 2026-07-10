"""The measurement harness — the smoke run and the phase-1 exact-search sweep.

    python -m ragscale.bench --n 1000                       # smoke: whole pipeline
    python -m ragscale.bench --grow 100000,500000,1000000   # phase 1 sweep
    python -m ragscale.bench --n 5000000                    # single point

The table is grown in place: because corpus.generate is prefix-consistent, each
step only embeds and loads the *new* rows, so the whole sweep embeds every chunk
exactly once. At each step we run 100 held-out queries with exact (sequential
scan) search, record p50/p95 latency, and save the exact top-k ids as the ground
truth every later phase measures recall against.

The recall@k comparison against IVFFlat/HNSW is phases 2–3 — see PLAN.md.
"""

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from . import corpus, db
from .embed import embed_batch, embed_iter

RESULTS_DIR = "results"

# A tiny human-readable sanity set, kept separate from the 100 generated
# held-out queries used for the actual latency/recall numbers.
SANITY_QUERIES = [
    "my card was charged twice, how do I get money back?",
    "I lost my phone and can't do two-factor login",
    "why did my file upload stop syncing?",
]


def load_corpus(conn, n: int, fresh: bool) -> float:
    """Ensure the table holds n chunks, embedding+loading only the new rows.

    Returns wall-clock seconds spent on this step's load (0.0 if nothing to do).
    """
    db.create_schema(conn, fresh=fresh)
    existing = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if existing >= n:
        print(f"table already has {existing} rows — nothing to add for n={n}")
        return 0.0

    to_add = n - existing
    print(f"embedding + loading {to_add} chunks ({existing} → {n})...")
    t0 = time.perf_counter()

    def rows():
        # Two independent deterministic passes stay aligned (same seed); skip the
        # rows already loaded so we only embed the delta.
        meta = corpus.generate(n)
        bodies = (body for _, body in corpus.generate(n))
        for _ in range(existing):
            next(meta)
            next(bodies)
        for (topic, body), (_, vec) in zip(meta, embed_iter(bodies, input_type="document")):
            yield topic, body, vec

    loaded = db.bulk_load(conn, tqdm(rows(), total=to_add, unit="chunk"))
    dt = time.perf_counter() - t0
    print(f"loaded {loaded} chunks in {dt:.1f}s ({loaded / dt:.0f} chunks/s incl. embedding)")
    return dt


def bench_exact(conn, qvecs, k: int, repeats: int):
    """Measure latency AND capture the exact top-k ids in one pass.

    A brute-force scan at 5M is ~seconds per query, so scanning the table a
    second time just to record ground truth would double the phase's wall time.
    Results are deterministic, so we keep each query's ids from the first repeat
    while still timing every repeat.

    Returns (p50_ms, p95_ms, n_samples, ground_truth).
    """
    db.search(conn, qvecs[0], k=k)  # warm up caches before timing
    latencies = []
    ground_truth: list[list[int]] = [[] for _ in qvecs]
    for r in range(repeats):
        for i, qv in enumerate(qvecs):
            t0 = time.perf_counter()
            rows = db.search(conn, qv, k=k)
            latencies.append((time.perf_counter() - t0) * 1000)
            if r == 0:
                ground_truth[i] = [row[0] for row in rows]
    arr = np.array(latencies)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), len(latencies), ground_truth


def record(n: int, load_s: float, size: str, p50: float, p95: float, ground_truth) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "phase1.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if is_new:
            f.write("n,load_s,table_size,p50_ms,p95_ms\n")
        f.write(f"{n},{load_s:.1f},{size},{p50:.2f},{p95:.2f}\n")

    with open(os.path.join(RESULTS_DIR, f"ground_truth_{n}.json"), "w") as f:
        json.dump({"n": n, "queries": ground_truth}, f)

    print(f"\nPLAN.md row → | {n} | {load_s:.1f}s | {size} | {p50:.1f}ms | {p95:.1f}ms |")


def run_point(conn, n: int, k: int, repeats: int, fresh: bool) -> None:
    load_s = load_corpus(conn, n, fresh)
    actual = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    size = conn.execute(
        "SELECT pg_size_pretty(pg_total_relation_size('chunks'))"
    ).fetchone()[0]
    print(f"\ncorpus: {actual} chunks, table size {size}")

    print("\n--- exact search (no index = ground truth) ---")
    db.drop_indexes(conn)
    qvecs = embed_batch(corpus.generate_queries(100), input_type="query")
    p50, p95, samples, ground_truth = bench_exact(conn, qvecs, k, repeats)
    print(f"exact query latency over {samples} runs: p50 {p50:.1f}ms  p95 {p95:.1f}ms")

    record(actual, load_s, size, p50, p95, ground_truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="corpus size to ensure loaded")
    parser.add_argument("--grow", type=str, help="comma-separated sizes to sweep, e.g. 100000,500000")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1, help="times to loop the 100 queries")
    parser.add_argument("--fresh", action="store_true", help="drop and rebuild the table first")
    args = parser.parse_args()

    conn = db.connect()
    steps = [int(s) for s in args.grow.split(",")] if args.grow else [args.n]
    for i, n in enumerate(sorted(steps)):
        # --fresh only on the first step; later steps grow the same table.
        run_point(conn, n, args.k, args.repeats, fresh=args.fresh and i == 0)

    # A quick human sanity check on the last-loaded corpus.
    top = db.search(conn, embed_batch([SANITY_QUERIES[0]], input_type="query")[0], k=3)
    print(f"\nsanity — top hits for {SANITY_QUERIES[0]!r}:")
    for _id, body, dist in top:
        print(f"  {dist:.3f}  {body[:90]}")


if __name__ == "__main__":
    main()

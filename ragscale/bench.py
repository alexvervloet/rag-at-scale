"""The measurement harness — and the end-to-end smoke run.

    python -m ragscale.bench --n 1000            # smoke: whole pipeline, tiny N
    python -m ragscale.bench --n 100000 --fresh  # phase 1 starts here

This file deliberately only implements the *baseline* path (load + exact
search + latency). The recall@k comparison against IVFFlat/HNSW is the actual
project — see PLAN.md phases 2–3. The intended shape: run the same queries
exact (ground truth) and indexed, and recall@k = |indexed ∩ exact| / k.
"""

import argparse
import statistics
import time

from tqdm import tqdm

from . import corpus, db
from .embed import embed_batch, embed_iter

QUERIES = [
    "my card was charged twice, how do I get money back?",
    "I lost my phone and can't do two-factor login",
    "why did my file upload stop syncing?",
    "how many API requests am I allowed per minute?",
    "notifications drain my battery on the app",
]


def load_corpus(conn, n: int, fresh: bool) -> None:
    db.create_schema(conn, fresh=fresh)
    existing = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if existing >= n:
        print(f"table already has {existing} rows — skipping load (use --fresh to rebuild)")
        return
    print(f"embedding + loading {n} chunks...")
    t0 = time.perf_counter()

    def rows():
        # corpus.generate is deterministic, so iterating it twice (once for
        # metadata, once for the text fed to the embedder) stays aligned.
        bodies = (body for _, body in corpus.generate(n))
        for (topic, body), (_, vec) in zip(corpus.generate(n), embed_iter(bodies)):
            yield topic, body, vec

    loaded = db.bulk_load(conn, tqdm(rows(), total=n, unit="chunk"))
    dt = time.perf_counter() - t0
    print(f"loaded {loaded} chunks in {dt:.1f}s ({loaded / dt:.0f} chunks/s incl. embedding)")


def bench_queries(conn, k: int, repeats: int = 5) -> None:
    latencies = []
    qvecs = embed_batch(QUERIES)
    for qvec in qvecs * repeats:
        t0 = time.perf_counter()
        db.search(conn, qvec, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    print(f"query latency over {len(latencies)} runs: p50 {p50:.1f}ms  p95 {p95:.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="corpus size to ensure loaded")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fresh", action="store_true", help="drop and rebuild the table")
    args = parser.parse_args()

    conn = db.connect()
    load_corpus(conn, args.n, args.fresh)
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'))").fetchone()[0]
    print(f"\ncorpus: {n} chunks, table size {size}")

    print("\n--- exact search (no index = ground truth) ---")
    db.drop_indexes(conn)
    bench_queries(conn, args.k)

    top = db.search(conn, embed_batch([QUERIES[0]])[0], k=3)
    print(f"\nsanity — top hits for {QUERIES[0]!r}:")
    for _id, body, dist in top:
        print(f"  {dist:.3f}  {body[:90]}")


if __name__ == "__main__":
    main()

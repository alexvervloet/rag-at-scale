"""Phase 4 — concurrency: does p95 hold under simultaneous queries?

    python -m ragscale.phase4 --concurrency 1,10,50 --ef-search 40

Phases 1–3 measured one query at a time. A chat app serves many at once. This
fires N threads, each on its own connection, hammering the live index, and reports
per-query p50/p95 and aggregate QPS as concurrency rises — so we see whether tail
latency holds or collapses once queries contend for the VM's CPUs.

Runs against whatever index is live (HNSW here); forces the index like phases 2–3.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import corpus, db
from .config import CORPUS_SOURCE, RESULTS_DIR
from .embed import embed_batch


def worker(qvecs, k: int, ef_search: int, n_queries: int) -> list[float]:
    conn = db.connect()
    conn.execute("SET enable_seqscan = off")
    db.set_ef_search(conn, ef_search)
    db.search(conn, qvecs[0], k=k)  # warm this connection
    lat = []
    for i in range(n_queries):
        qv = qvecs[i % len(qvecs)]
        t0 = time.perf_counter()
        db.search(conn, qv, k=k)
        lat.append((time.perf_counter() - t0) * 1000)
    conn.close()
    return lat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=str, default="1,10,50")
    parser.add_argument("--per-thread", type=int, default=200, help="queries per worker")
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    qvecs = embed_batch(corpus.generate_queries(100), input_type="query")
    print(f"phase 4: concurrency test on {CORPUS_SOURCE} (ef_search={args.ef_search}, "
          f"{args.per_thread} queries/thread)\n")

    # Warm the index into page cache first (the 3.9GB index is cold after a
    # container restart); otherwise the first level measures disk, not concurrency.
    warm = db.connect()
    warm.execute("SET enable_seqscan = off")
    db.set_ef_search(warm, args.ef_search)
    for _ in range(3):
        for qv in qvecs:
            db.search(warm, qv, k=args.k)
    warm.close()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"phase4_{CORPUS_SOURCE}.csv")
    with open(path, "w") as f:
        f.write("concurrency,total_queries,p50_ms,p95_ms,p99_ms,qps\n")
        for c in levels:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as pool:
                futures = [pool.submit(worker, qvecs, args.k, args.ef_search, args.per_thread)
                           for _ in range(c)]
                lat = [x for fut in futures for x in fut.result()]
            wall = time.perf_counter() - t0
            arr = np.array(lat)
            p50, p95, p99 = (float(np.percentile(arr, p)) for p in (50, 95, 99))
            qps = len(lat) / wall
            f.write(f"{c},{len(lat)},{p50:.2f},{p95:.2f},{p99:.2f},{qps:.0f}\n")
            print(f"concurrency={c:3d}  p50 {p50:6.2f}ms  p95 {p95:6.2f}ms  "
                  f"p99 {p99:6.2f}ms  throughput {qps:6.0f} qps")


if __name__ == "__main__":
    main()

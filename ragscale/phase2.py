"""Phase 2 — IVFFlat recall/latency sweep against phase-1 ground truth.

    python -m ragscale.phase2 --lists 1000,2236,5000 --probes 1,5,10,20,50,100

IVFFlat clusters the vectors into `lists` cells at build time; a query scans the
`probes` nearest cells instead of the whole table. More probes = higher recall,
slower query. We build each `lists` on the existing table (no reload — the row
ids still match the saved ground truth), sweep `probes`, and for each combo
record recall@10 vs the exact results, p50/p95 latency, and one-time build cost.

Two rules of thumb for `lists` are worth comparing directly: sqrt(N) (~2236 at
5M) and N/1000 (5000 at 5M).
"""

import argparse
import json
import os
import time

import numpy as np

from . import corpus, db
from .embed import embed_batch

RESULTS_DIR = "results"


def load_ground_truth(n: int) -> list[list[int]]:
    with open(os.path.join(RESULTS_DIR, f"ground_truth_{n}.json")) as f:
        return json.load(f)["queries"]


def recall_at_k(got: list[list[int]], truth: list[list[int]], k: int) -> float:
    """Mean over queries of |approx top-k ∩ exact top-k| / k."""
    scores = [len(set(g[:k]) & set(t[:k])) / k for g, t in zip(got, truth)]
    return sum(scores) / len(scores)


def measure(conn, qvecs, k: int, repeats: int):
    """Return (p50_ms, p95_ms, topk_ids). Ids captured on the first repeat.

    At 5M the index (38 GB) exceeds RAM, so a single warmup query isn't enough —
    a batch can hit cold pages and spike to ~seconds. Warm the *whole* query set
    first so we measure representative warm-cache latency, not cache misses.
    """
    for qv in qvecs:
        db.search(conn, qv, k=k)  # full warmup pass over every query's cells
    latencies = []
    ids: list[list[int]] = [[] for _ in qvecs]
    for r in range(repeats):
        for i, qv in enumerate(qvecs):
            t0 = time.perf_counter()
            rows = db.search(conn, qv, k=k)
            latencies.append((time.perf_counter() - t0) * 1000)
            if r == 0:
                ids[i] = [row[0] for row in rows]
    arr = np.array(latencies)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), ids


def record(row: dict, csv_name: str = "phase2.csv") -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, csv_name)
    is_new = not os.path.exists(path)
    cols = ["n", "lists", "probes", "recall_at_10", "p50_ms", "p95_ms", "build_s", "index_size"]
    with open(path, "a") as f:
        if is_new:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(row[c]) for c in cols) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lists", type=str, default="1000,2236,5000")
    parser.add_argument("--probes", type=str, default="1,5,10,20,50,100")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3, help="loops of the 100 queries for latency")
    parser.add_argument("--maintenance-mem", type=str, default="8GB",
                        help="maintenance_work_mem for the build; measured need at 5M×1024d: "
                             "~1.4GB at lists=2236, ~5.9GB at lists=5000")
    parser.add_argument("--measure-only", action="store_true",
                        help="skip the build and measure the existing index (re-measure "
                             "without a costly rebuild); pass the live index's --lists as the label")
    args = parser.parse_args()

    lists_vals = [int(x) for x in args.lists.split(",")]
    probes_vals = [int(x) for x in args.probes.split(",")]

    conn = db.connect()
    # IVFFlat's k-means needs more working memory than the server default at this
    # scale; raise it for this session (build-time only, does not affect queries).
    conn.execute(f"SET maintenance_work_mem = '{args.maintenance_mem}'")
    # Force the index. pgvector's IVFFlat cost estimate grows with probes; past a
    # threshold the planner picks a parallel seq scan (brute force, ~14s at 5M)
    # instead of the index — so without this we'd measure the fallback, not IVFFlat.
    # The fallback itself is a real gotcha (see LESSONS); here we characterize the index.
    conn.execute("SET enable_seqscan = off")
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    truth = load_ground_truth(n)
    qvecs = embed_batch(corpus.generate_queries(100), input_type="query")
    print(f"phase 2: IVFFlat on {n} vectors, {len(truth)} queries, k={args.k}\n")

    for lists in lists_vals:
        if args.measure_only:
            build_s = 0.0  # not rebuilt; caller carries the known build time
        else:
            t0 = time.perf_counter()
            db.create_ivfflat(conn, lists)
            build_s = time.perf_counter() - t0
        size = conn.execute(
            "SELECT pg_size_pretty(pg_relation_size('chunks_ivfflat'))"
        ).fetchone()[0]
        print(f"lists={lists}: {'(existing)' if args.measure_only else f'built in {build_s:.1f}s'}, index {size}")

        for probes in probes_vals:
            if probes > lists:
                continue
            db.set_probes(conn, probes)
            p50, p95, ids = measure(conn, qvecs, args.k, args.repeats)
            recall = recall_at_k(ids, truth, args.k)
            record({
                "n": n, "lists": lists, "probes": probes,
                "recall_at_10": f"{recall:.4f}", "p50_ms": f"{p50:.2f}",
                "p95_ms": f"{p95:.2f}", "build_s": f"{build_s:.1f}", "index_size": size,
            })
            print(f"  probes={probes:4d}  recall@10 {recall:.3f}  p50 {p50:6.1f}ms  p95 {p95:6.1f}ms")
        print()


if __name__ == "__main__":
    main()

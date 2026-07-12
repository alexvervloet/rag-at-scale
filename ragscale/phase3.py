"""Phase 3 — HNSW recall/latency sweep against phase-1 ground truth.

    python -m ragscale.phase3 --m 16 --ef-construction 64 --ef-search 10,40,100,200

HNSW builds a navigable small-world graph: `m` (links per node) and
`ef_construction` (candidate list at build) are build knobs; `ef_search` (candidate
list at query) is the query knob, analogous to IVFFlat's `probes`. We build each
(m, ef_construction) on the existing table (ids still match phase-1 ground truth),
sweep `ef_search`, and record recall@10, p50/p95, build time, and index size.

Reuses phase-2's warmed measurement and recall helpers. Like phase 2, it forces
the index (`enable_seqscan=off`) so we characterize HNSW, not a planner fallback.
"""

import argparse
import os
import time

from . import corpus, db
from .config import CORPUS_SOURCE, RESULTS_DIR
from .embed import embed_batch
from .phase2 import load_ground_truth, measure, recall_at_k


def record(row: dict, csv_name: str | None = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, csv_name or f"phase3_{CORPUS_SOURCE}.csv")
    is_new = not os.path.exists(path)
    cols = ["n", "m", "ef_construction", "ef_search", "recall_at_10",
            "p50_ms", "p95_ms", "build_s", "index_size"]
    with open(path, "a") as f:
        if is_new:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(row[c]) for c in cols) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=str, default="16", help="comma list of m values")
    parser.add_argument("--ef-construction", type=str, default="64", help="comma list")
    parser.add_argument("--ef-search", type=str, default="10,40,100,200", help="comma list (query knob)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maintenance-mem", type=str, default="10GB",
                        help="maintenance_work_mem for the build (HNSW graph is large at 5M)")
    parser.add_argument("--measure-only", action="store_true",
                        help="skip the build and measure the existing hnsw index")
    args = parser.parse_args()

    m_vals = [int(x) for x in args.m.split(",")]
    efc_vals = [int(x) for x in args.ef_construction.split(",")]
    efs_vals = [int(x) for x in args.ef_search.split(",")]

    conn = db.connect()
    db.set_maintenance_mem(conn, args.maintenance_mem)
    conn.execute("SET enable_seqscan = off")  # force the index (see phase 2 / LESSONS #9)
    n = db.scalar(conn, "SELECT count(*) FROM chunks")
    truth = load_ground_truth(n)
    qvecs = embed_batch(corpus.generate_queries(100), input_type="query")
    print(f"phase 3: HNSW on {n} vectors, {len(truth)} queries, k={args.k}\n")

    for m in m_vals:
        for efc in efc_vals:
            if args.measure_only:
                build_s = 0.0
            else:
                t0 = time.perf_counter()
                db.create_hnsw(conn, m=m, ef_construction=efc)
                build_s = time.perf_counter() - t0
            size = db.scalar(conn, "SELECT pg_size_pretty(pg_relation_size('chunks_hnsw'))")
            print(f"m={m} ef_construction={efc}: "
                  f"{'(existing)' if args.measure_only else f'built in {build_s:.1f}s'}, index {size}")

            for efs in efs_vals:
                if efs < args.k:
                    continue
                db.set_ef_search(conn, efs)
                p50, p95, ids = measure(conn, qvecs, args.k, args.repeats)
                recall = recall_at_k(ids, truth, args.k)
                record({
                    "n": n, "m": m, "ef_construction": efc, "ef_search": efs,
                    "recall_at_10": f"{recall:.4f}", "p50_ms": f"{p50:.2f}",
                    "p95_ms": f"{p95:.2f}", "build_s": f"{build_s:.1f}", "index_size": size,
                })
                print(f"  ef_search={efs:4d}  recall@10 {recall:.3f}  p50 {p50:6.1f}ms  p95 {p95:6.1f}ms")
            print()


if __name__ == "__main__":
    main()

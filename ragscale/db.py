"""pgvector schema, bulk loading, and the three query paths under test.

The whole project is a comparison of three ways to answer "top-k nearest":
  - exact()    — sequential scan, ground truth, O(n)
  - IVFFlat    — cluster then probe (build with create_ivfflat)
  - HNSW       — graph walk (build with create_hnsw)

Postgres picks the index automatically once one exists, so `search()` is the
same SQL either way; `drop_indexes()` returns you to brute force.
"""

from typing import Any, LiteralString

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from .config import EMBED_DIM, PG_DSN


def connect() -> psycopg.Connection:
    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_schema(conn: psycopg.Connection, fresh: bool = False) -> None:
    if fresh:
        conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        sql.SQL(
            """CREATE TABLE IF NOT EXISTS chunks (
            id bigserial PRIMARY KEY,
            topic text NOT NULL,
            body text NOT NULL,
            embedding vector({dim}) NOT NULL
        )"""
        ).format(dim=sql.Literal(EMBED_DIM))
    )


def bulk_load(conn: psycopg.Connection, rows) -> int:
    """COPY (topic, body, embedding) rows in — the fast path for millions."""
    count = 0
    with conn.cursor().copy(
        "COPY chunks (topic, body, embedding) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["text", "text", "vector"])
        for topic, body, vector in rows:
            copy.write_row((topic, body, np.array(vector, dtype=np.float32)))
            count += 1
    return count


def search(conn: psycopg.Connection, query_vec, k: int = 10) -> list[tuple[int, str, float]]:
    """Top-k by cosine distance. Uses an ANN index if one exists, else exact."""
    rows = conn.execute(
        "SELECT id, body, embedding <=> %s AS dist FROM chunks ORDER BY dist LIMIT %s",
        (np.array(query_vec, dtype=np.float32), k),
    ).fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows]


def drop_indexes(conn: psycopg.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS chunks_ivfflat")
    conn.execute("DROP INDEX IF EXISTS chunks_hnsw")


def create_ivfflat(conn: psycopg.Connection, lists: int) -> None:
    drop_indexes(conn)
    conn.execute(
        sql.SQL(
            "CREATE INDEX chunks_ivfflat ON chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        ).format(lists=sql.Literal(lists))
    )


def create_hnsw(conn: psycopg.Connection, m: int = 16, ef_construction: int = 64) -> None:
    drop_indexes(conn)
    conn.execute(
        sql.SQL(
            "CREATE INDEX chunks_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = {m}, ef_construction = {efc})"
        ).format(m=sql.Literal(m), efc=sql.Literal(ef_construction))
    )


def set_probes(conn: psycopg.Connection, probes: int) -> None:
    """IVFFlat query-time knob (per session)."""
    conn.execute(sql.SQL("SET ivfflat.probes = {}").format(sql.Literal(probes)))


def set_ef_search(conn: psycopg.Connection, ef: int) -> None:
    """HNSW query-time knob (per session)."""
    conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(ef)))


def set_maintenance_mem(conn: psycopg.Connection, mem: str) -> None:
    """Build-time working memory (per session) — index builds need more than the default."""
    conn.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(mem)))


def scalar(conn: psycopg.Connection, query: LiteralString) -> Any:
    """First column of the query's single row (count(*), pg_size_pretty, ...)."""
    row = conn.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {query}")
    return row[0]

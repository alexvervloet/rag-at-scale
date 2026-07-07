#!/usr/bin/env python3
"""Verify the environment for rag-at-scale. Makes no paid call — everything
in this project is local (Docker + Postgres/pgvector + Ollama)."""

import shutil
import sys

OK, BAD = "  \033[32m✓\033[0m", "  \033[31m✗\033[0m"
failures = 0


def check(label: str, ok: bool, fix: str = "") -> None:
    global failures
    print(f"{OK if ok else BAD} {label}" + ("" if ok else f"\n      fix: {fix}"))
    failures += 0 if ok else 1


print("rag-at-scale setup check\n")

check(
    f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.11+)",
    sys.version_info >= (3, 11),
    "install a newer Python",
)

for pkg in ("psycopg", "pgvector", "numpy", "requests", "dotenv", "tqdm"):
    try:
        __import__(pkg)
        check(f"package: {pkg}", True)
    except ImportError:
        check(f"package: {pkg}", False, "pip install -r requirements.txt")

check("docker on PATH", shutil.which("docker") is not None, "install Docker Desktop")

try:
    from ragscale.config import EMBED_DIM, EMBED_MODEL, OLLAMA_URL

    import requests

    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    models = [m["name"] for m in resp.json().get("models", [])]
    check(f"Ollama reachable at {OLLAMA_URL}", True)
    have = any(m.split(":")[0] == EMBED_MODEL for m in models)
    check(f"embed model pulled: {EMBED_MODEL}", have, f"ollama pull {EMBED_MODEL}")
    if have:
        vec = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": ["hello"]},
            timeout=30,
        ).json()["embeddings"][0]
        check(
            f"embedding dim {len(vec)} matches EMBED_DIM={EMBED_DIM}",
            len(vec) == EMBED_DIM,
            "set EMBED_DIM in .env to match the model",
        )
except Exception as exc:  # noqa: BLE001 — a setup checker should report, not crash
    check("Ollama reachable", False, f"start Ollama ({exc.__class__.__name__})")

try:
    from ragscale import db

    conn = db.connect()
    ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    check(f"Postgres + pgvector {ver[0] if ver else '?'} reachable", True)
except Exception as exc:  # noqa: BLE001
    check("Postgres reachable", False, f"docker compose up -d  ({exc.__class__.__name__})")

print()
if failures:
    sys.exit(f"{failures} check(s) failed — fix the items above and rerun.")
print("All good. Smoke-test the pipeline:  python -m ragscale.bench --n 1000")

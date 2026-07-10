"""Batched embeddings, from either a local Ollama server or the Voyage API.

Ollama is free but compute-bound (~165 chunks/s here); embedding a few million
chunks is hours. Voyage removes that ceiling — many concurrent batched requests
push throughput far higher — for ~free within its token tier. Pick with
EMBED_PROVIDER; the rest of the pipeline is provider-agnostic.

`input_type` uses asymmetric embeddings when the provider supports them (Voyage):
the corpus is embedded as "document", queries as "query", which measurably helps
retrieval. Providers that don't support it just ignore the hint.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import requests

from .config import (
    EMBED_BATCH,
    EMBED_MODEL,
    EMBED_PROVIDER,
    EMBED_WORKERS,
    OLLAMA_URL,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def _embed_ollama(texts: list[str], input_type: str | None = None) -> list[list[float]]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _embed_voyage(
    texts: list[str], input_type: str | None = None, retries: int = 6
) -> list[list[float]]:
    payload = {"model": VOYAGE_MODEL, "input": texts}
    if input_type:
        payload["input_type"] = input_type
    headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}"}
    for attempt in range(retries):
        resp = requests.post(VOYAGE_URL, json=payload, headers=headers, timeout=300)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(2**attempt, 30))  # backoff on rate limit / transient error
            continue
        resp.raise_for_status()
        # Voyage tags each embedding with its input index; sort to guarantee order.
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
    resp.raise_for_status()  # out of retries — surface the last error
    raise RuntimeError("unreachable")


_PROVIDERS = {"ollama": _embed_ollama, "voyage": _embed_voyage}


def embed_batch(texts: list[str], input_type: str | None = None) -> list[list[float]]:
    """Embed a list of texts in one request via the configured provider."""
    return _PROVIDERS[EMBED_PROVIDER](texts, input_type)


def embed_iter(texts, batch_size: int | None = None, workers: int | None = None, input_type=None):
    """Yield (text, vector) pairs, embedding in batches.

    For Voyage we fan out `workers` batches concurrently to beat the network
    round-trip; results are still yielded in input order, and only
    workers × batch_size texts are held in memory at once. Ollama stays serial
    (a single local server is compute-bound — concurrency wouldn't help).
    """
    is_voyage = EMBED_PROVIDER == "voyage"
    batch_size = batch_size or EMBED_BATCH or (128 if is_voyage else 64)
    workers = workers or EMBED_WORKERS or (8 if is_voyage else 1)
    it = iter(texts)

    if workers == 1:
        while batch := list(islice(it, batch_size)):
            yield from zip(batch, embed_batch(batch, input_type))
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            batches = []
            for _ in range(workers):
                batch = list(islice(it, batch_size))
                if not batch:
                    break
                batches.append(batch)
            if not batches:
                break
            for batch, vecs in zip(batches, pool.map(lambda b: embed_batch(b, input_type), batches)):
                yield from zip(batch, vecs)

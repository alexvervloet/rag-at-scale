"""Batched embeddings from the local Ollama server. Free, but not instant —
embedding a few million chunks is hours of compute, which is why bench.py
reports embed throughput from the start.
"""

import requests

from .config import EMBED_MODEL, OLLAMA_URL


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a list of texts in one request (Ollama's /api/embed is batched)."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_iter(texts, batch_size: int = 64):
    """Yield (text, vector) pairs, embedding in batches of `batch_size`.

    Tune batch_size against your machine: bigger batches amortize HTTP overhead
    until the model server becomes the bottleneck. Measure, don't guess.
    """
    buffer: list[str] = []
    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield from zip(buffer, embed_batch(buffer))
            buffer = []
    if buffer:
        yield from zip(buffer, embed_batch(buffer))

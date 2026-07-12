"""Real corpus — Wikipedia paragraphs from wikitext-103 (via HF datasets).

The synthetic generator (corpus.py) is fine for latency/throughput, but its
near-duplicate geometry breaks the IVFFlat-vs-HNSW comparison (LESSONS #11): it
flatters IVFFlat and traps HNSW. Real Wikipedia prose has the diverse, spread-out
neighborhoods an honest ANN comparison needs. Selected via CORPUS_SOURCE=wiki.

Deterministic and prefix-consistent (the dataset order is fixed), so it slots into
the same load/ground-truth machinery. Queries come from the *validation* split, so
they're real held-out text, never a corpus row.
"""

from collections.abc import Iterable, Iterator
from typing import cast

from datasets import load_dataset

WIKI_DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1")
MIN_CHARS = 200   # skip fragments/stubs
CHUNK_CHARS = 800  # ~150-200 tokens per chunk


def _chunks(split: str) -> Iterator[tuple[str, str]]:
    # load_dataset's return type is a broad union; rows here are {"text": str}.
    ds = cast(Iterable[dict[str, str]], load_dataset(*WIKI_DATASET, split=split))
    title = "unknown"
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        if t.startswith("=") and t.endswith("="):
            # " = Title = " is a top-level article; " = = Section = = " is deeper.
            if t.startswith("= ") and not t.startswith("= ="):
                title = t.strip("= ").strip()
            continue
        if len(t) < MIN_CHARS:
            continue
        for i in range(0, len(t), CHUNK_CHARS):
            chunk = t[i:i + CHUNK_CHARS]
            if len(chunk) >= MIN_CHARS:
                yield title, chunk


def generate(n: int, seed: int = 42) -> Iterator[tuple[str, str]]:
    """Yield (title, chunk) pairs from the train split; first n are stable."""
    count = 0
    for title, chunk in _chunks("train"):
        yield title, chunk
        count += 1
        if count >= n:
            return


def generate_queries(n: int = 100, seed: int = 1234) -> list[str]:
    """Held-out real queries: the first n chunks of the validation split."""
    out = []
    for _title, chunk in _chunks("validation"):
        out.append(chunk)
        if len(out) >= n:
            break
    return out

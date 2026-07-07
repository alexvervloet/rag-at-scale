"""Deterministic synthetic corpus.

For retrieval *mechanics* (recall, latency, memory), what matters is the number
and geometry of vectors, not prose quality — so a seeded generator that emits
varied, topic-clustered sentences is enough for phases 1–3. Phase 4 swaps in a
real corpus (e.g. a Wikipedia slice) for the write-up.
"""

import random
from collections.abc import Iterator

TOPICS = {
    "billing": ["invoice", "refund", "proration", "credit card", "tax", "receipt"],
    "auth": ["password", "two-factor", "session", "SSO", "recovery code", "login"],
    "storage": ["quota", "upload", "attachment", "sync conflict", "version history"],
    "api": ["rate limit", "webhook", "token", "endpoint", "pagination", "SDK"],
    "mobile": ["offline mode", "push notification", "widget", "battery", "camera scan"],
}

TEMPLATES = [
    "To resolve a {a} issue, open Settings and review the {b} panel before retrying.",
    "The {a} limit resets monthly; upgrading changes how {b} is calculated.",
    "If {a} fails repeatedly, clear the cached {b} state and sign in again.",
    "Admins can audit {a} events from the console, filtered by {b}.",
    "Our docs cover {a} edge cases, including how {b} interacts with team plans.",
]


def generate(n: int, seed: int = 42) -> Iterator[tuple[str, str]]:
    """Yield (topic, chunk_text) pairs, deterministically for a given seed."""
    rng = random.Random(seed)
    topics = list(TOPICS)
    for i in range(n):
        topic = topics[i % len(topics)]
        words = TOPICS[topic]
        a, b = rng.sample(words, 2)
        template = rng.choice(TEMPLATES)
        # The index i keeps every chunk unique so exact-duplicate vectors don't
        # flatter recall numbers.
        yield topic, f"[{topic} note {i}] " + template.format(a=a, b=b)

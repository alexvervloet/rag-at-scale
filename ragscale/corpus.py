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
    """Yield (topic, chunk_text) pairs, deterministically for a given seed.

    Prefix-consistent: row i is identical regardless of n, so growing the corpus
    (100k → 500k → …) only ever appends new rows — the harness embeds each chunk
    exactly once across the whole phase-1 sweep.
    """
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


# Question-style templates, deliberately distinct from the statement-style
# TEMPLATES above so a generated query is never a verbatim corpus row — the
# queries are genuinely held out.
QUERY_TEMPLATES = [
    "how do I fix {a} problems that also affect {b}?",
    "why does {a} keep failing after I change {b}?",
    "what is the limit on {a} and how does {b} change it?",
    "my {a} stopped working, is it related to {b}?",
    "can I recover {a} when {b} is misconfigured?",
    "steps to troubleshoot {a} when {b} looks wrong?",
]


def generate_queries(n: int = 100, seed: int = 1234) -> list[str]:
    """Deterministic held-out query set — the same 100 questions every run, so
    phase 1's exact results and later phases' indexed results are comparable."""
    rng = random.Random(seed)
    topics = list(TOPICS)
    out = []
    for i in range(n):
        topic = topics[i % len(topics)]
        a, b = rng.sample(TOPICS[topic], 2)
        out.append(rng.choice(QUERY_TEMPLATES).format(a=a, b=b))
    return out

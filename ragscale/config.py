"""Env-driven settings.

Two embedding providers, chosen by EMBED_PROVIDER:
  - "ollama" — local, $0, ~165 chunks/s (compute-bound on this machine).
  - "voyage" — hosted, far faster (concurrent batched requests), ~free within
    Voyage's 200M-token tier. Needs VOYAGE_API_KEY.
The only secret is VOYAGE_API_KEY, and it lives in .env (gitignored).
"""

import os

from dotenv import load_dotenv

load_dotenv()

PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5433/ragscale")

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Ollama (local) provider
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Voyage (hosted) provider
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4")

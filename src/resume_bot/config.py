"""
Central configuration for the AI Resume Bot.
All RAG parameters are precision-tuned for 1-2 page resume documents.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "resumes"
CHROMA_DIR = ROOT_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Google / Gemini ───────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# LLM — gemini-3-flash-preview: fast, responsive, active free tier quota
LLM_MODEL: str = "gemini-3.5-flash-lite"
LLM_TEMPERATURE: float = 0.1          # Near-deterministic for factual accuracy
LLM_MAX_OUTPUT_TOKENS: int = 1024

# Embedding — gemini-embedding-2: latest Google embedding model
# Native output: 3072 dims. We truncate to 768 for storage efficiency.
EMBEDDING_MODEL: str = "models/gemini-embedding-2"
EMBEDDING_DIMENSIONS: int = 768        # Matryoshka truncation to 768 dims

# ── Chunking ──────────────────────────────────────────────────────────────────
# For 1-2 page resumes:
#   chunk_size=500  → captures a complete job entry or skill block
#   chunk_overlap=75 → 15% overlap prevents context bleed at boundaries
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 75

# Resume-specific separators — double newline first, then bullet points
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", "•", "●", "◦", "-", " ", ""]

# ── Retriever ─────────────────────────────────────────────────────────────────
# MMR (Maximal Marginal Relevance) prevents returning duplicate chunks
# when the same skill / tool appears across multiple job entries.
RETRIEVER_SEARCH_TYPE: str = "mmr"
RETRIEVER_K: int = 5          # Final chunks returned to LLM
RETRIEVER_FETCH_K: int = 10   # Candidate pool for MMR diversity selection
RETRIEVER_LAMBDA: float = 0.6 # MMR tradeoff: 1.0=pure similarity, 0.0=pure diversity

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "resume_collection"

# ── CLI ───────────────────────────────────────────────────────────────────────
APP_NAME: str = "AI Resume Bot"
APP_VERSION: str = "0.1.0"
WELCOME_BANNER: str = f"[bold cyan]{APP_NAME}[/bold cyan] [dim]v{APP_VERSION}[/dim]"

"""
Embedding model setup.

Using Google's gemini-embedding-2:
  - Latest Google embedding model (replaces text-embedding-004)
  - Native output: 3072 dimensions (Matryoshka — supports truncation)
  - We use output_dimensionality=768 for efficient storage without quality loss
  - task_type="RETRIEVAL_DOCUMENT" for indexing corpus text
  - task_type="RETRIEVAL_QUERY"    for embedding user queries at retrieval time
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from resume_bot.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)


@lru_cache(maxsize=1)
def get_document_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached embedding model configured for document indexing.
    task_type='RETRIEVAL_DOCUMENT' — optimized for embedding corpus text.
    output_dimensionality=768 — Matryoshka truncation for storage efficiency.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )


@lru_cache(maxsize=1)
def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached embedding model configured for query embedding.
    task_type='RETRIEVAL_QUERY' — optimized for embedding user questions.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )

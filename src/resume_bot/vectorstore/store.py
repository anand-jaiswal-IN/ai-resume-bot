"""
ChromaDB vector store — persistent, developer-friendly, metadata-filterable.

Why ChromaDB for dev/testing?
  - Zero-config local SQLite persistence
  - Effortless metadata filtering (filter by section, candidate, etc.)
  - Python-native API
  - Easy to swap for FAISS/Pinecone in production (same LangChain interface)

ChromaDB → FAISS migration path (Phase 4):
  Replace Chroma(...) with FAISS.from_documents(...) and
  faiss_index.save_local() / FAISS.load_local() — zero chain changes needed.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from resume_bot.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA,
)
from resume_bot.vectorstore.embeddings import get_document_embeddings


def get_vector_store(collection_name: str = CHROMA_COLLECTION_NAME) -> Chroma:
    """
    Return (or create) a persistent ChromaDB collection.
    The store is loaded from disk if it already exists.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_document_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def index_documents(
    chunks: list[Document],
    collection_name: str = CHROMA_COLLECTION_NAME,
    reset: bool = False,
) -> Chroma:
    """
    Embed and store document chunks into ChromaDB.

    Args:
        chunks:          Enriched chunks from the chunker.
        collection_name: ChromaDB collection to use.
        reset:           If True, drops and recreates the collection
                         (useful when uploading a new resume).

    Returns:
        The populated Chroma vector store.
    """
    embeddings = get_document_embeddings()

    if reset:
        # Drop existing collection so old resume data is cleared
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        store.delete_collection()

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    return store


def get_retriever(store: Chroma | None = None):
    """
    Build an MMR retriever from the vector store.

    MMR (Maximal Marginal Relevance) configuration:
      - k=5 final chunks returned to LLM
      - fetch_k=10 candidate pool considered before diversity re-ranking
      - lambda_mult=0.6 balances relevance (1.0) vs diversity (0.0)

    Args:
        store: An existing Chroma store. If None, loads from disk.

    Returns:
        A LangChain VectorStoreRetriever ready for use in LCEL chains.
    """
    if store is None:
        store = get_vector_store()

    return store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA,
        },
    )


def collection_is_populated(collection_name: str = CHROMA_COLLECTION_NAME) -> bool:
    """Check if the ChromaDB collection already has documents indexed."""
    try:
        store = get_vector_store(collection_name)
        return store._collection.count() > 0
    except Exception:
        return False


def get_collection_stats(collection_name: str = CHROMA_COLLECTION_NAME) -> dict:
    """Return stats about the current collection for CLI display."""
    try:
        store = get_vector_store(collection_name)
        count = store._collection.count()
        return {"document_count": count, "collection": collection_name}
    except Exception:
        return {"document_count": 0, "collection": collection_name}

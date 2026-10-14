"""
LCEL RAG chain — the core reasoning pipeline.

Chain topology (LCEL pipe syntax):
  ┌─────────────────────────────────────────────────────────┐
  │  {"context": retriever | format_docs,                   │
  │   "question": RunnablePassthrough()}                    │
  │       │                                                 │
  │       ▼                                                 │
  │   resume_qa_prompt                                      │
  │       │                                                 │
  │       ▼                                                 │
  │   ChatGoogleGenerativeAI (gemini-2.0-flash)             │
  │       │                                                 │
  │       ▼                                                 │
  │   StrOutputParser                                       │
  └─────────────────────────────────────────────────────────┘

Multi-turn variant adds MessagesPlaceholder for chat_history and
uses RunnableWithMessageHistory to auto-manage session memory.
"""

from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from resume_bot.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)
from resume_bot.rag.prompts import resume_qa_prompt, resume_chat_prompt
from resume_bot.vectorstore.store import get_retriever


# ── LLM initialisation ────────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """Return the Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )


# ── Context formatting ────────────────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """
    Join retrieved document chunks into a single context string.
    Prefixes each chunk with its section label for traceability.
    """
    parts: list[str] = []
    for doc in docs:
        section = doc.metadata.get("section", "general").upper()
        page = doc.metadata.get("page", "?")
        parts.append(f"[{section} | Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Simple one-shot RAG chain ─────────────────────────────────────────────────

def build_rag_chain(retriever: VectorStoreRetriever | None = None):
    """
    Build a single-turn LCEL RAG chain.

    Returns a Runnable that accepts a plain string question and
    returns a plain string answer.

    Usage:
        chain = build_rag_chain(retriever)
        answer = chain.invoke("What are the candidate's skills?")
    """
    if retriever is None:
        retriever = get_retriever()

    llm = get_llm()

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | resume_qa_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Conversational multi-turn RAG chain ───────────────────────────────────────

# In-memory session store: session_id → ChatMessageHistory
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return or create a ChatMessageHistory for a given session."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_conversational_chain(retriever: VectorStoreRetriever | None = None):
    """
    Build a multi-turn conversational LCEL RAG chain with session memory.

    The chain maintains separate chat histories per session_id, allowing
    follow-up questions like "What about their latest project?" to work.

    Usage:
        chain = build_conversational_chain(retriever)
        response = chain.invoke(
            {"question": "What skills does the candidate have?"},
            config={"configurable": {"session_id": "user-1"}}
        )
    """
    if retriever is None:
        retriever = get_retriever()

    llm = get_llm()

    # Core chain (without history wrapper)
    core_chain = (
        {
            "context": RunnableLambda(
                lambda x: format_docs(retriever.invoke(x["question"]))
            ),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | resume_chat_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with session-based message history
    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def stream_answer(chain, question: str) -> Iterator[str]:
    """
    Stream the LLM answer token-by-token from a simple RAG chain.
    Yields string chunks as they arrive.
    """
    for chunk in chain.stream(question):
        yield chunk


def stream_conversational_answer(
    chain, question: str, session_id: str = "default"
) -> Iterator[str]:
    """
    Stream from the conversational chain with a specific session.
    Yields string chunks as they arrive.
    """
    for chunk in chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk


def clear_session(session_id: str = "default") -> None:
    """Clear the chat history for a given session."""
    if session_id in _session_store:
        _session_store[session_id].clear()


def get_session_history_messages(session_id: str = "default") -> list:
    """Return the chat history messages for a session."""
    return _get_session_history(session_id).messages

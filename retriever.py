"""Score and merge results from the FAQ, ticket, and guide collections."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_store")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SIMILARITY_SCORE = float(os.getenv("RAG_MIN_SIMILARITY", "0.40"))
RESULTS_PER_SOURCE = 3
MAX_CONTEXT_DOCUMENTS = 6


def _build_vectorstores() -> dict[str, Chroma]:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return {
        name: Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        for name in ("faq", "tickets", "guides")
    }


def build_ticket_store() -> Chroma:
    """Return the ticket collection for the retrieval-quality evaluation."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def distance_to_similarity(distance: float) -> float:
    """Map Chroma's nonnegative distance (lower is better) to a 0..1 score."""
    return 1.0 / (1.0 + max(float(distance), 0.0))


def retrieve_scored(query: str, stores: dict[str, Chroma] | None = None) -> list[Document]:
    """Return high-confidence documents sorted by normalized similarity."""
    stores = stores or _build_vectorstores()
    scored: list[tuple[float, Document]] = []
    for store in stores.values():
        for document, distance in store.similarity_search_with_score(
            query, k=RESULTS_PER_SOURCE
        ):
            similarity = distance_to_similarity(distance)
            if similarity >= MIN_SIMILARITY_SCORE:
                document.metadata["retrieval_score"] = similarity
                scored.append((similarity, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:MAX_CONTEXT_DOCUMENTS]]


def build_retriever() -> RunnableLambda:
    """Expose scored retrieval as a LangChain runnable."""
    stores = _build_vectorstores()
    return RunnableLambda(lambda query: retrieve_scored(query, stores))

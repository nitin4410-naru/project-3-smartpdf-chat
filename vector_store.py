"""Vector store creation and querying."""

from __future__ import annotations

import os
import shutil

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import EMBEDDING_MODEL, FAISS_INDEX_DIR, TOP_K_RESULTS, get_google_api_key


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_google_api_key(),
    )


def create_vector_store(text_chunks: list[Document]) -> FAISS:
    """Embed and persist documents to a local FAISS index."""
    if not text_chunks:
        raise ValueError("No text chunks available to create the vector store.")

    if os.path.isdir(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)

    vector_store = FAISS.from_documents(text_chunks, embedding=_get_embeddings())
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def query_vector_store(user_question: str, top_k: int = TOP_K_RESULTS) -> list[Document]:
    """Run similarity search and attach a simple confidence estimate to each hit."""
    if not os.path.isdir(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            "FAISS index not found. Process documents before asking questions."
        )

    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    results = vector_store.similarity_search_with_score(user_question, k=top_k)

    docs: list[Document] = []
    max_score = max((score for _, score in results), default=1.0)

    for doc, score in results:
        normalized = 1.0 if max_score <= 0 else max(0.0, min(1.0, 1 - (score / max_score)))
        doc.metadata["confidence"] = round(normalized * 100, 1)
        docs.append(doc)

    return docs

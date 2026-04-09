"""Application-wide configuration for SmartPDF Chat."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
TOP_K_RESULTS = 4
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
FAISS_INDEX_DIR = "faiss_index"
APP_TITLE = "SmartPDF Chat"

QA_PROMPT_TEMPLATE = (
    "Answer the question as detailed as possible from the provided context. "
    "If the answer is not in the context, say 'Answer is not available in the "
    "provided documents.' Mention which document the information comes from.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

SUMMARY_PROMPT_TEMPLATE = (
    "Provide a comprehensive summary of the following document content. "
    "Highlight key topics, findings, and important points.\n\n"
    "Content:\n{context}\n\n"
    "Summary:"
)

ANSWER_MODES = {
    "Detailed": (
        "Provide a rich, complete answer with important supporting details, "
        "organized clearly."
    ),
    "Concise": "Provide a crisp answer in a few focused sentences.",
}


def get_google_api_key() -> str:
    """Return the configured Google API key or raise a helpful error."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file before using the app."
        )
    return api_key

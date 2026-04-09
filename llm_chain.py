"""LLM helpers for question answering and summarization."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    ANSWER_MODES,
    MODEL_NAME,
    QA_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    get_google_api_key,
)


def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model client."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=get_google_api_key(),
        temperature=temperature,
    )


def get_qa_chain(answer_mode: str = "Detailed"):
    """Create a prompt runnable for document-grounded QA."""
    mode_instruction = ANSWER_MODES.get(answer_mode, ANSWER_MODES["Detailed"])
    prompt = PromptTemplate(
        template=f"{QA_PROMPT_TEMPLATE}\n\nAdditional instruction: {mode_instruction}",
        input_variables=["context", "question"],
    )
    return prompt | _get_llm(temperature=0.3)


def get_answer(user_question: str, relevant_docs, answer_mode: str = "Detailed") -> dict[str, Any]:
    """Generate an answer and return the supporting source documents."""
    if not relevant_docs:
        return {
            "answer": "Answer is not available in the provided documents.",
            "source_docs": [],
        }

    chain = get_qa_chain(answer_mode=answer_mode)
    context = "\n\n".join(
        (
            f"Document: {doc.metadata.get('source', 'Unknown')}\n"
            f"Pages: {doc.metadata.get('page_start', '?')}-{doc.metadata.get('page_end', '?')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in relevant_docs
    )
    response = chain.invoke({"context": context, "question": user_question})
    answer = getattr(response, "content", str(response))
    return {"answer": answer.strip(), "source_docs": relevant_docs}


def get_summary(all_chunks, answer_mode: str = "Detailed") -> dict[str, Any]:
    """Summarize the most relevant chunk set for the uploaded documents."""
    if not all_chunks:
        raise ValueError("No document chunks are available for summarization.")

    mode_instruction = ANSWER_MODES.get(answer_mode, ANSWER_MODES["Detailed"])
    llm = _get_llm(temperature=0.2)
    summary_prompt = PromptTemplate(
        template=(
            f"{SUMMARY_PROMPT_TEMPLATE}\n\nAdditional instruction: {mode_instruction}"
        ),
        input_variables=["context"],
    )

    combined_context = "\n\n".join(chunk.page_content for chunk in all_chunks[:8])
    response = llm.invoke(summary_prompt.format(context=combined_context))
    summary_text = getattr(response, "content", str(response)).strip()

    return {
        "answer": summary_text,
        "source_docs": all_chunks[:8],
    }

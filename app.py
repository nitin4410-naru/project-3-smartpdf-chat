"""Streamlit entrypoint for SmartPDF Chat."""

from __future__ import annotations

from io import StringIO

import streamlit as st

from llm_chain import get_answer, get_summary
from pdf_processor import chunk_text, extract_text_from_pdfs
from ui_components import apply_custom_css, render_chat_history, render_sidebar
from vector_store import create_vector_store, query_vector_store


st.set_page_config(
    page_title="SmartPDF Chat",
    page_icon="📄",
    layout="wide",
)


def _initialize_session_state() -> None:
    defaults = {
        "chat_history": [],
        "documents_processed": False,
        "vector_store_ready": False,
        "all_chunks": [],
        "last_uploaded_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_message(role: str, message: str, source_docs=None, confidence=None) -> None:
    st.session_state.chat_history.append(
        {
            "role": role,
            "message": message,
            "sources": source_docs or [],
            "confidence": confidence,
        }
    )


def _serialize_sources(source_docs):
    serialized = []
    for doc in source_docs:
        serialized.append({"content": doc.page_content, "metadata": doc.metadata})
    return serialized


def _export_chat_history() -> str:
    buffer = StringIO()
    for entry in st.session_state.chat_history:
        role = entry["role"].upper()
        buffer.write(f"{role}: {entry['message']}\n")
        if entry.get("sources"):
            buffer.write("Sources:\n")
            for source in entry["sources"]:
                metadata = source["metadata"]
                buffer.write(
                    f"- {metadata.get('source', 'Unknown')} "
                    f"(pages {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')})\n"
                )
        buffer.write("\n")
    return buffer.getvalue()


def main() -> None:
    _initialize_session_state()
    apply_custom_css()

    uploaded_files, process_clicked, summarize_clicked, answer_mode = render_sidebar()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Chat With Your PDFs</div>
            <p class="hero-copy">
                Build a searchable knowledge base from multiple PDF files, ask grounded questions,
                and inspect the exact source passages behind each answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        st.download_button(
            "Export Chat History",
            data=_export_chat_history(),
            file_name="smartpdf-chat-history.txt",
            mime="text/plain",
            use_container_width=False,
        )

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF before processing.")
        else:
            st.session_state.chat_history = []
            progress = st.progress(0, text="Reading uploaded PDFs...")
            try:
                raw_text = extract_text_from_pdfs(uploaded_files)
                progress.progress(35, text="Splitting extracted text into chunks...")
                text_chunks = chunk_text(raw_text)
                if not text_chunks:
                    raise ValueError(
                        "No readable text was found in the uploaded PDFs. Try different files."
                    )
                progress.progress(70, text="Creating FAISS vector index...")
                create_vector_store(text_chunks)
                progress.progress(100, text="Documents processed successfully.")

                st.session_state.documents_processed = True
                st.session_state.vector_store_ready = True
                st.session_state.all_chunks = text_chunks
                st.session_state.last_uploaded_names = [file.name for file in uploaded_files]
                st.success(
                    f"Processed {len(uploaded_files)} document(s) into {len(text_chunks)} searchable chunks."
                )
            except Exception as exc:
                st.session_state.documents_processed = False
                st.session_state.vector_store_ready = False
                st.session_state.all_chunks = []
                st.error(f"Processing failed: {exc}")

    if summarize_clicked:
        if not st.session_state.documents_processed or not st.session_state.all_chunks:
            st.warning("Process documents before requesting a summary.")
        else:
            try:
                with st.spinner("Generating document summary..."):
                    summary_result = get_summary(st.session_state.all_chunks, answer_mode=answer_mode)
                serialized_sources = _serialize_sources(summary_result["source_docs"])
                _append_message(
                    "assistant",
                    summary_result["answer"],
                    source_docs=serialized_sources,
                )
            except Exception as exc:
                st.error(f"Summary failed: {exc}")

    question = st.text_input(
        "Ask a question about your documents",
        placeholder="What are the key findings across these PDFs?",
    )

    if st.button("Send", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question before sending.")
        elif not st.session_state.vector_store_ready:
            st.warning("Process your documents before starting the chat.")
        else:
            _append_message("user", question.strip())
            try:
                with st.spinner("Searching documents and drafting answer..."):
                    relevant_docs = query_vector_store(question.strip())
                    answer_result = get_answer(
                        question.strip(),
                        relevant_docs,
                        answer_mode=answer_mode,
                    )
                serialized_sources = _serialize_sources(answer_result["source_docs"])
                confidence_values = [
                    doc.metadata.get("confidence", 0.0) for doc in answer_result["source_docs"]
                ]
                avg_confidence = (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else None
                )
                _append_message(
                    "assistant",
                    answer_result["answer"],
                    source_docs=serialized_sources,
                    confidence=avg_confidence,
                )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

    render_chat_history(st.session_state.chat_history)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

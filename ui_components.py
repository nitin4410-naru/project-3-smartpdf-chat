"""Reusable Streamlit UI components."""

from __future__ import annotations

import html

import streamlit as st


def apply_custom_css() -> None:
    """Inject portfolio-style dark theme CSS."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #07111f;
                --panel: #0d1b2a;
                --panel-2: #13263a;
                --text: #e8f1ff;
                --muted: #9fb3c8;
                --accent: #4fd1c5;
                --accent-2: #ffb454;
                --border: rgba(159, 179, 200, 0.18);
                --user: linear-gradient(135deg, #1f6feb, #2ea043);
                --assistant: linear-gradient(135deg, #17283d, #0f2235);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(79, 209, 197, 0.18), transparent 24%),
                    radial-gradient(circle at top right, rgba(255, 180, 84, 0.14), transparent 22%),
                    linear-gradient(180deg, #040b14 0%, var(--bg) 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #081321 0%, #0d1b2a 100%);
                border-right: 1px solid var(--border);
            }

            .app-shell {
                padding: 1.5rem 0 2rem;
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(19, 38, 58, 0.95), rgba(10, 22, 36, 0.92));
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 1.5rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
            }

            .hero-title {
                font-size: 2.1rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                margin-bottom: 0.4rem;
            }

            .hero-copy {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.6;
                margin: 0;
            }

            .chat-bubble {
                border-radius: 20px;
                padding: 1rem 1.1rem;
                margin: 0.75rem 0;
                border: 1px solid var(--border);
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
            }

            .chat-user {
                background: var(--user);
                margin-left: 15%;
            }

            .chat-assistant {
                background: var(--assistant);
                margin-right: 15%;
            }

            .chat-role {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #d8e7ff;
                opacity: 0.82;
                margin-bottom: 0.45rem;
            }

            .source-pill {
                display: inline-block;
                margin: 0.2rem 0.35rem 0 0;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: rgba(79, 209, 197, 0.12);
                border: 1px solid rgba(79, 209, 197, 0.25);
                color: #d9fffb;
                font-size: 0.8rem;
            }

            .stButton > button {
                width: 100%;
                border-radius: 14px;
                border: 1px solid rgba(79, 209, 197, 0.32);
                background: linear-gradient(135deg, #103554, #0d5b62);
                color: white;
                font-weight: 600;
            }

            .stDownloadButton > button {
                border-radius: 12px;
            }

            .stTextInput input {
                background: rgba(13, 27, 42, 0.82);
                color: var(--text);
                border-radius: 14px;
                border: 1px solid var(--border);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar controls and return their state."""
    with st.sidebar:
        st.title("SmartPDF Chat")
        st.caption("Upload PDFs, build a local FAISS index, and chat with document-grounded answers.")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDFs and process them together.",
        )
        answer_mode = st.radio(
            "Answer mode",
            options=["Detailed", "Concise"],
            index=0,
            help="Choose whether the assistant should respond with more depth or brevity.",
        )
        process_clicked = st.button("Process Documents", use_container_width=True)
        summarize_clicked = st.button("Summarize Documents", use_container_width=True)
        st.markdown(
            "Free stack: Gemini free tier, local FAISS search, and Streamlit UI."
        )

    return uploaded_files, process_clicked, summarize_clicked, answer_mode


def render_chat_history(chat_history) -> None:
    """Render user and assistant turns with optional source context."""
    for index, entry in enumerate(chat_history):
        role = entry["role"]
        message = html.escape(entry["message"]).replace("\n", "<br>")
        sources = entry.get("sources", [])
        confidence = entry.get("confidence")
        bubble_class = "chat-user" if role == "user" else "chat-assistant"
        role_label = "You" if role == "user" else "AI Assistant"

        st.markdown(
            f"""
            <div class="chat-bubble {bubble_class}">
                <div class="chat-role">{role_label}</div>
                <div>{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if role == "assistant" and confidence is not None:
            st.caption(f"Confidence estimate: {confidence:.1f}%")

        if role == "assistant" and sources:
            with st.expander(f"View Source Context #{index + 1}"):
                for source in sources:
                    page_start = source["metadata"].get("page_start", "?")
                    page_end = source["metadata"].get("page_end", "?")
                    source_name = source["metadata"].get("source", "Unknown document")
                    st.markdown(
                        f"<span class='source-pill'>{html.escape(source_name)} | Pages {page_start}-{page_end}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(source["content"])


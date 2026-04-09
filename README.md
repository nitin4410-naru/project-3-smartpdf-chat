# Project 3: SmartPDF Chat

SmartPDF Chat is a Streamlit-based Generative AI app that lets users upload multiple PDFs, build a local FAISS index, and chat with their content using Google Gemini and LangChain. It is designed to be fully free to run with the Google AI Studio free tier and local vector search.

## Features

- Upload and process multiple PDF files in one session
- Metadata-aware chunking with source document and page references
- Ask grounded questions and inspect the exact source context used
- Summarize uploaded documents in one click
- Dark-themed UI with polished chat bubbles and sidebar controls
- Session-persistent chat history using Streamlit `session_state`
- Confidence estimate for answers
- Toggle between detailed and concise answer modes
- Export chat history as a text file

## Tech Stack

- Python 3.10+
- Streamlit
- PyPDF2
- LangChain
- Google Gemini `gemini-2.0-flash`
- Google Generative AI Embeddings
- FAISS (`faiss-cpu`)
- `python-dotenv`

## Project Structure

```text
project-3-smartpdf-chat/
├── app.py
├── pdf_processor.py
├── vector_store.py
├── llm_chain.py
├── ui_components.py
├── config.py
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── faiss_index/
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get a free API key from [Google AI Studio](https://aistudio.google.com/).

4. Update the `.env` file:

```env
GOOGLE_API_KEY=your_key_here
```

## Run The App

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, usually `http://localhost:8501`.

## How To Use

1. Upload one or more PDF files from the sidebar.
2. Click `Process Documents` to extract text, chunk it, and build the FAISS index.
3. Click `Summarize Documents` for a high-level overview.
4. Ask questions in the main chat input and inspect `View Source Context` for citations.
5. Export the conversation if you want to keep a text transcript.

## Notes

- This project uses local FAISS storage, so the vector index is rebuilt when you process a new set of PDFs.
- If a PDF contains scanned images instead of selectable text, PyPDF2 may not extract useful content.
- The `.env` file is ignored by git, so your API key stays local.
- Current Gemini model names are used for compatibility: `gemini-2.0-flash` for chat and `models/gemini-embedding-001` for embeddings.

## Screenshots

Add screenshots here after running the app locally:

- `assets/home.png`
- `assets/chat.png`

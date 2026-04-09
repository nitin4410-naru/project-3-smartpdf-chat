"""PDF extraction and chunking utilities."""

from __future__ import annotations

from typing import Iterable

from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text_from_pdfs(uploaded_files) -> list[dict]:
    """Extract text from uploaded PDFs while preserving file and page metadata."""
    extracted_pages: list[dict] = []

    for uploaded_file in uploaded_files or []:
        reader = PdfReader(uploaded_file)
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            extracted_pages.append(
                {
                    "file_name": uploaded_file.name,
                    "page_number": page_index,
                    "text": page_text,
                }
            )

    return extracted_pages


def _build_file_text_with_spans(pages: Iterable[dict]) -> tuple[str, list[dict]]:
    full_text_parts: list[str] = []
    page_spans: list[dict] = []
    cursor = 0

    for page in pages:
        page_text = page["text"]
        if full_text_parts:
            separator = "\n\n"
            full_text_parts.append(separator)
            cursor += len(separator)

        start = cursor
        full_text_parts.append(page_text)
        cursor += len(page_text)
        end = cursor

        page_spans.append(
            {
                "page_number": page["page_number"],
                "start": start,
                "end": end,
            }
        )

    return "".join(full_text_parts), page_spans


def _pages_for_chunk(page_spans: list[dict], chunk_start: int, chunk_end: int) -> list[int]:
    pages: list[int] = []
    for span in page_spans:
        overlaps = span["start"] < chunk_end and span["end"] > chunk_start
        if overlaps:
            pages.append(span["page_number"])
    return pages


def chunk_text(raw_text_with_metadata: list[dict]) -> list[Document]:
    """Split extracted PDF text into metadata-rich LangChain documents."""
    if not raw_text_with_metadata:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    text_chunks: list[Document] = []
    file_names = sorted({item["file_name"] for item in raw_text_with_metadata})

    for file_name in file_names:
        file_pages = [item for item in raw_text_with_metadata if item["file_name"] == file_name]
        full_text, page_spans = _build_file_text_with_spans(file_pages)
        if not full_text.strip():
            continue

        chunks = splitter.split_text(full_text)
        search_start = 0

        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_start = full_text.find(chunk, max(0, search_start - CHUNK_OVERLAP))
            if chunk_start == -1:
                chunk_start = search_start
            chunk_end = chunk_start + len(chunk)
            search_start = chunk_end

            page_numbers = _pages_for_chunk(page_spans, chunk_start, chunk_end)
            if not page_numbers:
                page_numbers = [file_pages[0]["page_number"]]

            text_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "chunk_id": f"{file_name}-chunk-{chunk_index}",
                        "page_start": min(page_numbers),
                        "page_end": max(page_numbers),
                        "page_numbers": page_numbers,
                    },
                )
            )

    return text_chunks

"""
Extracts raw text from uploaded resume files (PDF or DOCX).
"""

import io
import pymupdf  # PyMuPDF
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_chunks = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text())
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    text_chunks = [p.text for p in document.paragraphs]

    # Many resume templates use tables for layout (columns, sidebars, etc.)
    # python-docx's .paragraphs only covers top-level body text, so pull
    # table cell text separately or it gets silently skipped.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text_chunks.append(cell.text)

    return "\n".join(chunk for chunk in text_chunks if chunk.strip())


def extract_text_from_resume(file_bytes: bytes, filename: str) -> str:
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

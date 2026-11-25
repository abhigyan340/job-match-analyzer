"""
Extracts raw text from uploaded files (PDF or DOCX).
Used for both resumes and job description uploads.
"""

import io
import pymupdf  # PyMuPDF
import docx


def _extract_pdf(file_bytes: bytes) -> str:
    """Pull all text from every page of a PDF."""
    text_chunks = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text())
    return "\n".join(text_chunks)


def _extract_docx(file_bytes: bytes) -> str:
    """
    Pull text from paragraphs and table cells.
    Many resume templates use tables for layout, so table cells
    must be read or entire sections go missing.
    """
    document = docx.Document(io.BytesIO(file_bytes))
    text_chunks = [p.text for p in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text_chunks.append(cell.text)

    return "\n".join(chunk for chunk in text_chunks if chunk.strip())


# 5 MB limit per file
MAX_FILE_BYTES = 5 * 1024 * 1024


def extract_text_from_file(file_bytes: bytes, filename: str, label: str = "file") -> str:
    """
    Extract text from a PDF or DOCX file.

    Args:
        file_bytes: raw bytes of the uploaded file.
        filename:   original filename — extension determines the parser.
        label:      human-readable name used in error messages (e.g. "resume"
                    or "job description").

    Raises:
        ValueError: if the file type is unsupported, over the size limit, or
                    yields no extractable text.
    """
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ValueError(
            f"The {label} file is too large (max 5 MB). "
            f"Got {len(file_bytes) / (1024 * 1024):.1f} MB."
        )

    lower = filename.lower()
    if lower.endswith(".pdf"):
        text = _extract_pdf(file_bytes)
    elif lower.endswith(".docx"):
        text = _extract_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type for {label}: '{filename}'. "
            "Only .pdf and .docx are accepted."
        )

    return text


# Keep the old name as an alias so nothing that might import it elsewhere breaks.
def extract_text_from_resume(file_bytes: bytes, filename: str) -> str:
    return extract_text_from_file(file_bytes, filename, label="resume")

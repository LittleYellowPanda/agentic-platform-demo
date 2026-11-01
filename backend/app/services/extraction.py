import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    """
    Extract plain text from a .txt, .md or .pdf file.

    Args:
        filename: Name of the file, used to detect its type from the extension.
        content: Raw bytes of the file.

    Returns:
        The extracted text.

    Raises:
        ValueError: If the extension is not supported, the file cannot be read
            (not UTF-8, corrupted PDF), or no text can be extracted
            (empty file, scanned PDF).
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}.")

    if extension == ".pdf":
        text = _extract_pdf(content)
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("The file is not UTF-8 encoded.") from e

    # PostgreSQL text columns can't store NUL characters, which some PDFs produce
    text = text.replace("\x00", "")

    if not text.strip():
        raise ValueError("No extractable text (empty file or scanned PDF?).")

    return text


def _extract_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise ValueError("The PDF file could not be read (corrupted?).") from e
    return "\n\n".join(pages)
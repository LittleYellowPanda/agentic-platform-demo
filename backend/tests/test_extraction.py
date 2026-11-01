import pytest

from app.services.extraction import SUPPORTED_EXTENSIONS, extract_text


# --- Plain text and Markdown --------------------------------------------------


def test_txt_returns_decoded_text():
    assert extract_text("notes.txt", "Le fonds investit.".encode()) == "Le fonds investit."


def test_markdown_is_kept_as_is():
    content = "# Title\n\n- **bold** item"
    assert extract_text("readme.md", content.encode()) == content


def test_accents_are_preserved():
    assert extract_text("a.txt", "européennes".encode("utf-8")) == "européennes"


def test_extension_is_case_insensitive():
    assert extract_text("REPORT.TXT", b"hello") == "hello"


# --- PDF ------------------------------------------------------------------------


def test_pdf_text_is_extracted(make_pdf):
    text = extract_text("fund.pdf", make_pdf(["Fund prospectus"]))
    assert "Fund prospectus" in text


def test_pdf_pages_are_separated_by_a_blank_line(make_pdf):
    text = extract_text("fund.pdf", make_pdf(["First page", "Second page"]))
    first, second = text.split("\n\n")
    assert "First page" in first
    assert "Second page" in second


def test_pdf_without_text_is_rejected(make_pdf):
    # A page without text layer behaves like a scanned page
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("scan.pdf", make_pdf([""]))


def test_corrupted_pdf_is_rejected():
    with pytest.raises(ValueError, match="could not be read"):
        extract_text("broken.pdf", b"this is not a PDF")


# --- Rejected inputs ------------------------------------------------------------


@pytest.mark.parametrize("filename", ["report.docx", "data.csv", "README"])
def test_unsupported_extension_is_rejected(filename):
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(filename, b"some content")


@pytest.mark.parametrize("content", [b"", b"   ", b"\n\n\t \n"])
def test_empty_or_whitespace_text_is_rejected(content):
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("empty.txt", content)


def test_non_utf8_file_is_rejected():
    latin1 = "européennes".encode("latin-1")  # e.g. an old Windows text file
    with pytest.raises(ValueError, match="not UTF-8"):
        extract_text("old.txt", latin1)


def test_nul_characters_are_removed():
    # PostgreSQL refuses NUL characters in text columns: the insert would fail
    assert extract_text("a.txt", b"Le\x00 fonds") == "Le fonds"


def test_only_nul_characters_is_rejected():
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("a.txt", b"\x00\x00")


def test_supported_extensions():
    # Guards the public contract: the upload route will rely on this constant
    assert SUPPORTED_EXTENSIONS == {".txt", ".md", ".pdf"}

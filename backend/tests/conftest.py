"""Shared test helpers.

pytest loads this file automatically: every fixture defined here is available
in all test files, just by naming it as a test argument.
"""
import re
from dataclasses import dataclass

import pytest


# --- A fake tokenizer --------------------------------------------------------
# chunk_text receives its tokenizer as an argument, so the tests can give it
# this tiny fake one instead of a real model's tokenizer:
#   - no download, no network: tests run anywhere, in milliseconds;
#   - predictable: 1 word = 1 token, so we know exactly where chunks should cut.
# It mimics the only part of the real Tokenizer that chunk_text uses:
# encode(...) returning an object with .ids and .offsets.


@dataclass
class FakeEncoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


class WhitespaceTokenizer:
    """One token per word, with (start, end) character offsets like the real one."""

    def encode(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
        matches = list(re.finditer(r"\S+", text))
        return FakeEncoding(
            ids=list(range(len(matches))),
            offsets=[(m.start(), m.end()) for m in matches],
        )


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


# --- A tiny PDF builder -------------------------------------------------------
# Builds a valid PDF in memory, one page per string, without any extra library.
# An empty string gives a page with no text: it behaves like a scanned page.


def _build_pdf(pages: list[str]) -> bytes:
    objects: list[bytes] = []

    # 1: catalog, 2: page tree, 3: font, then for each page: page + content stream
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    # Assemble the file, recording each object's byte offset for the xref table
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def make_pdf():
    return _build_pdf

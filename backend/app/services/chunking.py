
from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass
class TextChunk:
    index: int
    text: str
    token_count: int


def chunk_text(
    text: str, tokenizer: Tokenizer, chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    """
    Split a text into overlapping chunks, counted in tokens.

    Args:
        text: The text to split.
        tokenizer: The tokenizer of the embedding model that will consume the chunks.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens shared between two consecutive chunks.

    Returns:
        The list of chunks, numbered from 0. Empty if the text is empty.

    Raises:
        ValueError: If chunk_size is not positive, or chunk_overlap is negative
            or not strictly smaller than chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and strictly smaller than chunk_size.")

    encoding = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(encoding.ids)
    step = chunk_size - chunk_overlap

    chunks: list[TextChunk] = []
    start = 0

    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)

        # Cut the original text between the first and last token of the window
        start_char = encoding.offsets[start][0]
        end_char = encoding.offsets[end - 1][1]
        chunk_content = text[start_char:end_char]

        chunks.append(
            TextChunk(index=len(chunks), text=chunk_content, token_count=end - start)
        )

        if end == n_tokens:
            break

        start += step

    return chunks
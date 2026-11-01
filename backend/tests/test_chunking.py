import socket

import pytest

from app.services.chunking import chunk_text


def words(n: int) -> str:
    """'w0 w1 w2 ...': with the whitespace tokenizer, n words = n tokens."""
    return " ".join(f"w{i}" for i in range(n))


# --- Basic behaviour ------------------------------------------------------------


def test_empty_text_gives_no_chunks(tokenizer):
    assert chunk_text("", tokenizer, chunk_size=10, chunk_overlap=2) == []


def test_short_text_gives_a_single_chunk(tokenizer):
    chunks = chunk_text("Le fonds investit.", tokenizer, chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "Le fonds investit."
    assert chunks[0].token_count == 3


def test_exact_fit_does_not_create_an_extra_chunk(tokenizer):
    # 4 tokens, window of 4: one chunk. Without the early stop, a second
    # chunk made only of overlap would appear.
    chunks = chunk_text(words(4), tokenizer, chunk_size=4, chunk_overlap=1)
    assert len(chunks) == 1


def test_sliding_window_positions(tokenizer):
    # size 4, overlap 1 → step 3 → windows [0-4), [3-7), [6-10)
    chunks = chunk_text(words(10), tokenizer, chunk_size=4, chunk_overlap=1)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


# --- Invariants: properties that must hold for any text -------------------------


def test_chunks_are_numbered_from_zero(tokenizer):
    chunks = chunk_text(words(20), tokenizer, chunk_size=5, chunk_overlap=2)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_exceeds_chunk_size(tokenizer):
    chunks = chunk_text(words(57), tokenizer, chunk_size=10, chunk_overlap=3)
    assert all(c.token_count <= 10 for c in chunks)


def test_consecutive_chunks_share_exactly_the_overlap(tokenizer):
    chunks = chunk_text(words(30), tokenizer, chunk_size=8, chunk_overlap=3)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text.split()[-3:] == current.text.split()[:3]


def test_zero_overlap_gives_disjoint_chunks(tokenizer):
    chunks = chunk_text(words(9), tokenizer, chunk_size=3, chunk_overlap=0)
    assert [c.text for c in chunks] == ["w0 w1 w2", "w3 w4 w5", "w6 w7 w8"]


def test_whole_text_is_covered(tokenizer):
    text = words(23)
    chunks = chunk_text(text, tokenizer, chunk_size=6, chunk_overlap=2)
    assert text.startswith(chunks[0].text)
    assert text.endswith(chunks[-1].text)


def test_original_text_is_preserved(tokenizer):
    # Chunks are cut from the original text (offsets), not rebuilt from tokens:
    # capitals, accents and punctuation must survive.
    text = "Le Fonds investit en actions EUROPÉENNES, à long terme."
    chunks = chunk_text(text, tokenizer, chunk_size=4, chunk_overlap=1)
    for chunk in chunks:
        assert chunk.text in text
    assert "EUROPÉENNES," in chunks[1].text


# --- Invalid parameters -------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [
        (0, 0),     # size must be positive
        (-5, 0),
        (10, -1),   # overlap can't be negative
        (10, 10),   # overlap == size: step would be 0, infinite loop
        (10, 15),   # overlap > size: step would be negative
    ],
)
def test_invalid_parameters_are_rejected(tokenizer, chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- With a real tokenizer (optional) -------------------------------------------


def test_real_tokenizer_keeps_chunks_under_the_limit():
    """Integration test: downloads a real tokenizer, so it's skipped offline."""
    tokenizers = pytest.importorskip("tokenizers")
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        pytest.skip("Hugging Face unreachable (offline?)")
    # Outside the try: any other failure must fail the test, not skip it
    tok = tokenizers.Tokenizer.from_pretrained("intfloat/multilingual-e5-small")

    text = "Le fonds investit principalement en actions européennes. " * 200
    chunks = chunk_text(text, tok, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 500
        # What the embedding model will really receive: E5 prefix + special
        # tokens included. It must fit in the model's 512-token window, or the
        # end of the chunk would be silently truncated.
        model_input = tok.encode("passage: " + chunk.text, add_special_tokens=True)
        assert len(model_input.ids) <= 512

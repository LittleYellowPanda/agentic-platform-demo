import socket

import numpy as np
import pytest

from app.services import embeddings
from app.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    embed_passages,
    embed_query,
    get_model_config,
)

E5 = "intfloat/multilingual-e5-small"
PARAPHRASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FakeModel:
    """Stands in for a FastEmbed model: records its inputs, returns fixed vectors."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.received: list[str] = []

    def embed(self, texts):
        for text in texts:
            self.received.append(text)
            yield np.full(self.dimension, 0.5, dtype=np.float32)


@pytest.fixture
def fake_model(monkeypatch) -> FakeModel:
    """Replace get_model so no real model is downloaded or loaded."""
    model = FakeModel()
    monkeypatch.setattr(embeddings, "get_model", lambda name: model)
    return model


# --- Registry ---------------------------------------------------------------------


def test_default_model_is_in_the_registry():
    assert DEFAULT_EMBEDDING_MODEL in EMBEDDING_MODELS


def test_registry_dimensions_match_the_database_column():
    # A model with another dimension could not be stored in the vector(384) column
    pytest.importorskip("sqlalchemy")
    from app.db.models import EMBEDDING_DIM

    assert all(config.dimension == EMBEDDING_DIM for config in EMBEDDING_MODELS.values())


def test_max_chunk_size_leaves_room_for_prefix_and_special_tokens():
    for config in EMBEDDING_MODELS.values():
        assert config.max_chunk_size < config.max_tokens


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_model_config("some/random-model")


def test_get_model_rejects_unknown_names_before_any_download():
    embeddings.get_model.cache_clear()
    with pytest.raises(ValueError, match="Unknown embedding model"):
        embeddings.get_model("some/random-model")


# --- Passages ---------------------------------------------------------------------


def test_passages_get_the_e5_passage_prefix(fake_model):
    embed_passages(E5, ["Le fonds investit.", "En actions."])
    assert fake_model.received == ["passage: Le fonds investit.", "passage: En actions."]


def test_models_without_prefix_receive_the_raw_text(fake_model):
    embed_passages(PARAPHRASE, ["Le fonds investit."])
    assert fake_model.received == ["Le fonds investit."]


def test_passages_return_one_plain_list_per_text(fake_model):
    vectors = embed_passages(E5, ["a", "b", "c"])
    assert len(vectors) == 3
    assert all(isinstance(v, list) and len(v) == 384 for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_no_passages_means_no_model_call(monkeypatch):
    def fail(name):
        raise AssertionError("the model should not be loaded for an empty list")

    monkeypatch.setattr(embeddings, "get_model", fail)
    assert embed_passages(E5, []) == []


def test_passages_with_unknown_model_are_rejected(fake_model):
    with pytest.raises(ValueError):
        embed_passages("some/random-model", ["text"])


# --- Queries ----------------------------------------------------------------------


def test_query_gets_the_e5_query_prefix(fake_model):
    embed_query(E5, "Quelle est la limite d'exposition ?")
    assert fake_model.received == ["query: Quelle est la limite d'exposition ?"]


def test_query_returns_a_single_vector(fake_model):
    vector = embed_query(E5, "question")
    assert isinstance(vector, list) and len(vector) == 384


def test_query_and_passage_prefixes_differ_for_e5(fake_model):
    # The asymmetric-model trap: same text, different prefix depending on the role
    embed_query(E5, "texte")
    embed_passages(E5, ["texte"])
    assert fake_model.received == ["query: texte", "passage: texte"]


# --- With the real libraries (optional, need network) -----------------------------


def huggingface_reachable() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False




@pytest.mark.parametrize("name", list(EMBEDDING_MODELS))
def test_real_tokenizer_sees_the_whole_text(name):
    """A tokenizer file shipped with truncation would cut long documents."""
    pytest.importorskip("tokenizers")
    if not huggingface_reachable():
        pytest.skip("Hugging Face unreachable (offline?)")
    tokenizer = embeddings.get_tokenizer(name)

    long_text = "Le fonds investit en actions européennes. " * 300
    n_tokens = len(tokenizer.encode(long_text, add_special_tokens=False).ids)
    assert n_tokens > 1000


def test_every_registry_model_can_be_loaded_by_fastembed():
    """Offline check: each model is in FastEmbed's catalog, or we register it.

    This is the test that was missing when E5-small turned out not to be in the
    Python FastEmbed catalog: the app only failed at the first upload.
    """
    fastembed = pytest.importorskip("fastembed")
    embeddings.register_custom_models()
    supported = {m["model"] for m in fastembed.TextEmbedding.list_supported_models()}
    for config in EMBEDDING_MODELS.values():
        assert config.name in supported or config.custom_onnx_file, (
            f"{config.name} is unknown to FastEmbed and has no custom_onnx_file"
        )


@pytest.mark.parametrize("name", list(EMBEDDING_MODELS))
def test_real_models_produce_vectors_of_the_right_size(name):
    """Downloads each model the first time (a few hundred MB), then uses the cache."""
    pytest.importorskip("fastembed")
    # Skip ONLY when offline. A broad `except Exception: skip` here once turned
    # "model not supported" into a silent skip, and the bug reached the app.
    if not huggingface_reachable():
        pytest.skip("Hugging Face unreachable (offline?)")

    vectors = embed_passages(name, ["Le fonds investit en actions."])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_MODELS[name].dimension

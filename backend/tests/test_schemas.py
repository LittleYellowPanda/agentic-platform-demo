import pytest
from pydantic import ValidationError

from app.schemas.knowledge_bases import KnowledgeBaseCreate, SearchRequest
from app.services.embeddings import DEFAULT_EMBEDDING_MODEL

E5 = "intfloat/multilingual-e5-small"
PARAPHRASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# --- Knowledge base creation --------------------------------------------------------


def test_defaults_come_from_the_model():
    kb = KnowledgeBaseCreate(name="Prospectus")
    assert kb.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert kb.chunk_size == 500
    assert kb.chunk_overlap == 50  # 10% of the chunk size


def test_short_context_model_gets_small_chunks_by_default():
    kb = KnowledgeBaseCreate(name="Short", embedding_model=PARAPHRASE)
    assert kb.chunk_size == 120
    assert kb.chunk_overlap == 12


def test_explicit_values_are_kept():
    kb = KnowledgeBaseCreate(name="Custom", chunk_size=300, chunk_overlap=30)
    assert (kb.chunk_size, kb.chunk_overlap) == (300, 30)


def test_name_is_stripped():
    assert KnowledgeBaseCreate(name="  Prospectus  ").name == "Prospectus"


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_name_is_rejected(name):
    with pytest.raises(ValidationError):
        KnowledgeBaseCreate(name=name)


def test_unknown_model_is_rejected():
    with pytest.raises(ValidationError, match="Unknown embedding model"):
        KnowledgeBaseCreate(name="KB", embedding_model="some/random-model")


def test_chunk_size_too_large_for_the_model_is_rejected():
    # 500 tokens is fine for E5, but MiniLM was trained on 128-token sequences
    with pytest.raises(ValidationError, match="too large"):
        KnowledgeBaseCreate(name="KB", embedding_model=PARAPHRASE, chunk_size=500)


def test_chunk_size_too_small_is_rejected():
    with pytest.raises(ValidationError):
        KnowledgeBaseCreate(name="KB", chunk_size=5)


@pytest.mark.parametrize("overlap", [500, 600])
def test_overlap_not_smaller_than_chunk_size_is_rejected(overlap):
    with pytest.raises(ValidationError, match="strictly smaller"):
        KnowledgeBaseCreate(name="KB", chunk_size=500, chunk_overlap=overlap)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValidationError):
        KnowledgeBaseCreate(name="KB", chunk_overlap=-1)


# --- Search request -----------------------------------------------------------------


def test_search_defaults_to_five_results():
    assert SearchRequest(query="limite d'exposition").top_k == 5


@pytest.mark.parametrize("top_k", [0, 21])
def test_top_k_out_of_range_is_rejected(top_k):
    with pytest.raises(ValidationError):
        SearchRequest(query="question", top_k=top_k)


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")

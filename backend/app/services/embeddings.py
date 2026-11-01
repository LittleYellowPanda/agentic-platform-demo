"""Embedding models: which ones are allowed, how they're loaded, how texts are encoded.

The rest of the application never talks to FastEmbed or handles prefixes directly:
it calls embed_passages() when indexing chunks and embed_query() when searching.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingModelConfig:
    name: str  # exact FastEmbed / Hugging Face name
    dimension: int  # size of the vectors; must match the pgvector column
    max_tokens: int  # input length the model handles well (prefix + special tokens included)
    max_chunk_size: int  # largest safe chunk (room left for prefix + special tokens); the default
    query_prefix: str  # added before a search query ("" if the model needs none)
    passage_prefix: str  # added before each chunk at indexing time
    description: str  # shown in the UI when creating a KB
    # Set only for models FastEmbed doesn't ship: path of the ONNX file in the
    # model's Hugging Face repo, so we can register it ourselves (see below)
    custom_onnx_file: str | None = None


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    config.name: config
    for config in [
        EmbeddingModelConfig(
            name="intfloat/multilingual-e5-small",
            dimension=384,
            max_tokens=512,
            max_chunk_size=500,
            query_prefix="query: ",
            passage_prefix="passage: ",
            description="Multilingual (French, English and ~100 languages). Recommended default.",
            # Not in FastEmbed's built-in catalog (only the large E5 is), but the
            # model's repo ships an ONNX export that FastEmbed can load
            custom_onnx_file="onnx/model.onnx",
        ),
        EmbeddingModelConfig(
            name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            dimension=384,
            # Trained on sequences of 128 tokens: longer chunks would be truncated
            # or poorly represented, hence much smaller chunks.
            max_tokens=128,
            max_chunk_size=120,
            query_prefix="",
            passage_prefix="",
            description="Multilingual, fast. Short chunks only (about 120 tokens).",
        ),
        EmbeddingModelConfig(
            name="BAAI/bge-small-en-v1.5",
            dimension=384,
            max_tokens=512,
            max_chunk_size=500,
            # BGE recommends this instruction for short queries searching longer passages
            query_prefix="Represent this sentence for searching relevant passages: ",
            passage_prefix="",
            description="English only. Good choice for English-language documents.",
        ),
    ]
}

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def get_model_config(name: str) -> EmbeddingModelConfig:
    """Return the configuration of an allowed model, or raise ValueError."""
    try:
        return EMBEDDING_MODELS[name]
    except KeyError:
        allowed = ", ".join(EMBEDDING_MODELS)
        raise ValueError(f"Unknown embedding model '{name}'. Allowed: {allowed}.") from None


@lru_cache(maxsize=1)
def register_custom_models() -> None:
    """Teach FastEmbed the registry models it doesn't ship. Runs once.

    FastEmbed only loads models from its own catalog. For the others, it needs
    to be told where the ONNX file is and how to turn token vectors into one
    sentence vector (E5: mean pooling, then normalization).
    """
    supported = {model["model"] for model in TextEmbedding.list_supported_models()}
    for config in EMBEDDING_MODELS.values():
        if config.name in supported or config.custom_onnx_file is None:
            continue
        TextEmbedding.add_custom_model(
            model=config.name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=config.name),
            dim=config.dimension,
            model_file=config.custom_onnx_file,
        )


@lru_cache(maxsize=None)
def get_model(name: str) -> TextEmbedding:
    """Load a model once; later calls with the same name return the same object.

    Loading takes a few seconds and downloads the weights on first use.
    """
    get_model_config(name)  # validates the name before any download
    register_custom_models()
    # Imported here rather than at the top: importing this module (e.g. in unit
    # tests) must not require the database settings to be present.
    from app.core.config import settings

    return TextEmbedding(model_name=name, cache_dir=settings.model_cache_dir)


@lru_cache(maxsize=None)
def get_tokenizer(name: str) -> Tokenizer:
    """Load the tokenizer of a model, configured for counting tokens in chunking."""
    get_model_config(name)
    tokenizer = Tokenizer.from_pretrained(name)
    # Some tokenizer files ship with truncation (e.g. to 128 tokens) and padding
    # enabled. For chunking we must see the WHOLE text, unpadded: otherwise a
    # 40-page document would silently be cut down to its first chunk.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def embed_passages(model_name: str, texts: list[str]) -> list[list[float]]:
    """Embed chunks for indexing. One vector per text, in the same order."""
    config = get_model_config(model_name)
    if not texts:
        return []
    model = get_model(model_name)
    prefixed = [config.passage_prefix + text for text in texts]
    # embed() processes the texts in batches and yields one numpy array per text
    return [vector.tolist() for vector in model.embed(prefixed)]


def embed_query(model_name: str, text: str) -> list[float]:
    """Embed a search query, with the model's query prefix."""
    config = get_model_config(model_name)
    model = get_model(model_name)
    vectors = list(model.embed([config.query_prefix + text]))
    return vectors[0].tolist()


def warm_up(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Load (and download if needed) a model and its tokenizer ahead of time.

    Called in the background at startup, so the first upload isn't slowed down by
    a download. Failures are only logged: the model will be loaded on first use.
    """
    try:
        get_tokenizer(model_name)
        get_model(model_name)
        logger.info("Embedding model '%s' ready.", model_name)
    except Exception:
        logger.exception("Could not preload embedding model '%s'.", model_name)

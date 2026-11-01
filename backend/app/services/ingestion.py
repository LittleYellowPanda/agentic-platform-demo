"""Background indexing of an uploaded document: text → chunks → vectors → database.

Runs AFTER the upload request has returned, so it opens its own database session:
the request's session is already closed by then.
"""
import logging

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.services.chunking import chunk_text
from app.services.embeddings import embed_passages, get_tokenizer
from app.services.extraction import extract_text

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Indexing was interrupted (server restart). Please upload the file again."
UNEXPECTED_MESSAGE = "Unexpected error during indexing. See the server logs for details."


def index_document(document_id: int, content: bytes) -> None:
    """Index one document. Never raises: failures are recorded on the document."""
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:  # deleted before the task started
            return

        document.status = "processing"
        db.commit()  # visible right away to the UI polling the status

        try:
            kb = document.knowledge_base
            text = extract_text(document.filename, content)
            tokenizer = get_tokenizer(kb.embedding_model)
            pieces = chunk_text(text, tokenizer, kb.chunk_size, kb.chunk_overlap)
            vectors = embed_passages(kb.embedding_model, [piece.text for piece in pieces])

            db.add_all(
                Chunk(
                    document_id=document.id,
                    knowledge_base_id=kb.id,
                    chunk_index=piece.index,
                    content=piece.text,
                    token_count=piece.token_count,
                    embedding=vector,
                )
                for piece, vector in zip(pieces, vectors, strict=True)
            )
            document.status = "ready"
            document.error_message = None
            # ONE commit for all the chunks AND the status: search never sees a
            # half-indexed document, and a failure leaves no orphan chunks.
            db.commit()

        except ValueError as error:
            # Expected failures (unsupported content, scanned PDF...): the message
            # is written for the user
            _mark_failed(db, document_id, str(error))
        except Exception:
            # Bugs or infrastructure problems: full details in the logs only
            logger.exception("Indexing failed for document %s", document_id)
            _mark_failed(db, document_id, UNEXPECTED_MESSAGE)


def _mark_failed(db: Session, document_id: int, message: str) -> None:
    db.rollback()  # discard any chunks added before the failure
    document = db.get(Document, document_id)
    if document is None:  # deleted while it was being indexed
        return
    document.status = "error"
    document.error_message = message
    db.commit()


def recover_interrupted_documents(db: Session) -> int:
    """Mark documents left 'pending' or 'processing' by a previous run as failed.

    Background tasks live in the server's memory: if the server stops, they are
    lost, and their documents would otherwise stay 'processing' forever.
    Called once at startup. Returns the number of documents updated.
    """
    result = db.execute(
        update(Document)
        .where(Document.status.in_(["pending", "processing"]))
        .values(status="error", error_message=INTERRUPTED_MESSAGE)
    )
    db.commit()
    return result.rowcount

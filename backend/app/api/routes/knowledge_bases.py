"""HTTP routes for knowledge bases, documents and semantic search.
"""
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Chunk, Document, KnowledgeBase
from app.db.session import get_db
from app.schemas.knowledge_bases import (
    DocumentRead,
    EmbeddingModelRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    SearchRequest,
    SearchResponse,
)
from app.services.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from app.services.extraction import SUPPORTED_EXTENSIONS
from app.services.ingestion import index_document
from app.services.search import search_knowledge_bases

router = APIRouter(tags=["knowledge bases"])

DbSession = Annotated[Session, Depends(get_db)]


def _get_kb_or_404(db: Session, kb_id: int) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_id} not found.")
    return kb


# --- Embedding models -----------------------------------------------------------


@router.get("/embedding-models", response_model=list[EmbeddingModelRead])
def list_embedding_models() -> list[EmbeddingModelRead]:
    """The models a KB can be created with (to fill the dropdown in the UI)."""
    return [
        EmbeddingModelRead(
            name=config.name,
            description=config.description,
            dimension=config.dimension,
            max_chunk_size=config.max_chunk_size,
            is_default=config.name == DEFAULT_EMBEDDING_MODEL,
        )
        for config in EMBEDDING_MODELS.values()
    ]


# --- Knowledge bases --------------------------------------------------------------


@router.post("/knowledge-bases", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(payload: KnowledgeBaseCreate, db: DbSession) -> KnowledgeBase:
    kb = KnowledgeBase(**payload.model_dump())
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(db: DbSession) -> list[KnowledgeBaseRead]:
    # One query with a GROUP BY for the counts, instead of one query per KB (N+1)
    rows = db.execute(
        select(KnowledgeBase, func.count(Document.id))
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.created_at.desc())
    ).all()
    return [
        KnowledgeBaseRead.model_validate(kb).model_copy(update={"document_count": count})
        for kb, count in rows
    ]


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(kb_id: int, db: DbSession) -> KnowledgeBaseRead:
    kb = _get_kb_or_404(db, kb_id)
    count = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == kb_id))
    return KnowledgeBaseRead.model_validate(kb).model_copy(update={"document_count": count})


@router.delete("/knowledge-bases/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: int, db: DbSession) -> Response:
    _get_kb_or_404(db, kb_id)
    # A direct DELETE: the database's ON DELETE CASCADE removes documents and
    # chunks in one statement. db.delete(kb) would make the ORM load every chunk
    # into memory first, to delete them one by one.
    db.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    db.commit()
    return Response(status_code=204)


# --- Documents ------------------------------------------------------------------


@router.post(
    "/knowledge-bases/{kb_id}/documents", response_model=DocumentRead, status_code=202
)
def upload_document(
    kb_id: int, file: UploadFile, background_tasks: BackgroundTasks, db: DbSession
) -> Document:
    """Store the document as 'pending' and index it in the background.

    Returns 202 Accepted immediately: the client polls the document's status.
    """
    kb = _get_kb_or_404(db, kb_id)

    filename = Path(file.filename or "").name  # drop any directory part
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension}'. Supported: {supported}.",
        )

    # Read one byte more than allowed: enough to know the file is too big,
    # without loading a huge file entirely into memory
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {settings.max_upload_size_mb} MB)."
        )
    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    document = Document(knowledge_base_id=kb.id, filename=filename[:255], status="pending")
    db.add(document)
    db.commit()
    db.refresh(document)

    # Runs after the response is sent, in a worker thread
    background_tasks.add_task(index_document, document.id, content)
    return document


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentRead])
def list_documents(kb_id: int, db: DbSession) -> list[DocumentRead]:
    _get_kb_or_404(db, kb_id)
    rows = db.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(Document.knowledge_base_id == kb_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentRead.model_validate(document).model_copy(update={"chunk_count": count})
        for document, count in rows
    ]


@router.delete("/knowledge-bases/{kb_id}/documents/{document_id}", status_code=204)
def delete_document(kb_id: int, document_id: int, db: DbSession) -> Response:
    document = db.get(Document, document_id)
    if document is None or document.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    db.execute(delete(Document).where(Document.id == document_id))
    db.commit()
    return Response(status_code=204)


# --- Search ---------------------------------------------------------------------


@router.post("/knowledge-bases/{kb_id}/search", response_model=SearchResponse)
def search(kb_id: int, payload: SearchRequest, db: DbSession) -> SearchResponse:
    kb = _get_kb_or_404(db, kb_id)
    results = search_knowledge_bases(db, [kb], payload.query, payload.top_k)
    return SearchResponse(
        query=payload.query, embedding_model=kb.embedding_model, results=results
    )

"""Semantic search: the chunks closest to a query, within one or several KBs."""
from collections import defaultdict

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, KnowledgeBase
from app.schemas.knowledge_bases import SearchHit
from app.services.embeddings import embed_query


def search_knowledge_bases(
    db: Session, knowledge_bases: list[KnowledgeBase], query: str, top_k: int
) -> list[SearchHit]:
    """Return the top_k chunks most similar to the query, best first.

    Works for one KB (the search tester) or several (an agent linked to many KBs).
    """
    # Vectors from different models live in different spaces: the query must be
    # embedded once per model, and compared only with chunks of that model.
    kb_ids_by_model: dict[str, list[int]] = defaultdict(list)
    for kb in knowledge_bases:
        kb_ids_by_model[kb.embedding_model].append(kb.id)

    # The HNSW index finds nearest neighbours FIRST, then the KB filter is applied.
    # By default it only explores ~40 candidates: if most belong to other KBs, we'd
    # get fewer than top_k results. Iterative scan (pgvector >= 0.8) keeps exploring
    # until enough rows pass the filter. SET LOCAL limits this to the current
    # transaction, so it never leaks to other requests through the connection pool.
    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))

    hits: list[SearchHit] = []
    for model_name, kb_ids in kb_ids_by_model.items():
        query_vector = embed_query(model_name, query)
        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.knowledge_base_id,
                Chunk.chunk_index,
                Chunk.content,
                Document.filename,
                distance,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.knowledge_base_id.in_(kb_ids))
            .order_by(distance)
            .limit(top_k)
        )

        for row in db.execute(statement):
            hits.append(
                SearchHit(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    knowledge_base_id=row.knowledge_base_id,
                    filename=row.filename,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    # pgvector's <=> returns the cosine DISTANCE = 1 - similarity
                    score=round(1 - row.distance, 4),
                )
            )

    # With several models, scores aren't strictly comparable across models
    # (different spaces): an approximation, documented as a known limitation.
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]

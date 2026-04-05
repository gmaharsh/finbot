import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from finbot.embeddings import embed_texts
from finbot.settings import get_settings


def client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(vector_size: int) -> None:
    s = get_settings()
    cl = client()
    if cl.collection_exists(s.qdrant_collection_name):
        return
    cl.create_collection(
        collection_name=s.qdrant_collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def upsert_points(points: list[dict[str, Any]]) -> None:
    """Each point: id (str), vector (list[float]), payload (dict)."""
    if not points:
        return
    s = get_settings()
    cl = client()
    ensure_collection(len(points[0]["vector"]))
    cl.upsert(
        collection_name=s.qdrant_collection_name,
        points=[
            qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points
        ],
        wait=True,
    )


def delete_by_source_document(source_document: str) -> None:
    s = get_settings()
    cl = client()
    if not cl.collection_exists(s.qdrant_collection_name):
        return
    flt = qm.Filter(
        must=[qm.FieldCondition(key="source_document", match=qm.MatchValue(value=source_document))]
    )
    # qdrant-client accepts Filter as points selector in recent versions
    cl.delete(collection_name=s.qdrant_collection_name, points_selector=flt, wait=True)


def search_filtered(
    query: str,
    allowed_collections: list[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    s = get_settings()
    cl = client()
    if not cl.collection_exists(s.qdrant_collection_name):
        return []
    if not allowed_collections:
        return []
    vec = embed_texts([query])[0]
    flt = qm.Filter(
        must=[
            qm.FieldCondition(
                key="collection",
                match=qm.MatchAny(any=allowed_collections),
            )
        ]
    )
    hits = cl.search(
        collection_name=s.qdrant_collection_name,
        query_vector=vec,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    out = []
    for h in hits:
        pl = h.payload or {}
        out.append(
            {
                "id": str(h.id),
                "score": h.score,
                "content": pl.get("content", ""),
                "source_document": pl.get("source_document", ""),
                "collection": pl.get("collection", ""),
                "access_roles": pl.get("access_roles", []),
                "section_title": pl.get("section_title", ""),
                "page_number": pl.get("page_number", 0),
                "chunk_type": pl.get("chunk_type", "text"),
                "parent_chunk_id": pl.get("parent_chunk_id"),
            }
        )
    return out


def collection_vector_size() -> int | None:
    s = get_settings()
    cl = client()
    if not cl.collection_exists(s.qdrant_collection_name):
        return None
    info = cl.get_collection(s.qdrant_collection_name)
    params = info.config.params.vectors
    if isinstance(params, qm.VectorParams):
        return params.size
    if isinstance(params, dict):
        v = next(iter(params.values()))
        return getattr(v, "size", None)
    return None

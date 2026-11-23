import atexit
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    Filter, FieldCondition, MatchValue,
)
from src import config as cfg

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if cfg.QDRANT_URL:
            _client = QdrantClient(url=cfg.QDRANT_URL)
        else:
            _client = QdrantClient(path=str(cfg.QDRANT_PATH))
    return _client


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be uint or UUID. Deterministic UUID makes re-indexing idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection() -> None:
    c = get_client()
    if not c.collection_exists(cfg.QDRANT_COLLECTION):
        c.create_collection(
            collection_name=cfg.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=cfg.EMBED_DIM, distance=Distance.COSINE),
        )
        if cfg.QDRANT_URL:   # payload indexes only matter on a server
            c.create_payload_index(cfg.QDRANT_COLLECTION, "video_id", PayloadSchemaType.KEYWORD)
            c.create_payload_index(cfg.QDRANT_COLLECTION, "playlist_index", PayloadSchemaType.INTEGER)


def upsert_chunks(chunks: list[dict], vectors: list[list[float]], batch: int = 100) -> None:
    c = get_client()
    points = [
        PointStruct(id=point_id(ch["chunk_id"]), vector=vec, payload=ch)
        for ch, vec in zip(chunks, vectors)
    ]
    for i in range(0, len(points), batch):
        c.upsert(collection_name=cfg.QDRANT_COLLECTION, points=points[i:i + batch])


def search(vector: list[float], top_k: int, video_id: str | None = None):
    flt = None
    if video_id:
        flt = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
    res = get_client().query_points(
        collection_name=cfg.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    )
    return res.points


def count() -> int:
    return get_client().count(cfg.QDRANT_COLLECTION).count

def get_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch chunk payloads by chunk_id (used to pull in neighbors of a hit)."""
    if not chunk_ids:
        return {}
    points = get_client().retrieve(
        collection_name=cfg.QDRANT_COLLECTION,
        ids=[point_id(c) for c in chunk_ids],
        with_payload=True,
    )
    return {p.payload["chunk_id"]: p.payload for p in points}    


def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


atexit.register(close)   # close cleanly before Python starts shutting down

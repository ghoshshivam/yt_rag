import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from src import config as cfg

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg.EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vecs = get_model().encode(
        texts,
        batch_size=cfg.EMBED_BATCH,
        normalize_embeddings=True,     # cosine similarity works best with normalized vectors
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(question: str) -> list[float]:
    return embed_texts([cfg.QUERY_PREFIX + question])[0]

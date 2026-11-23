from src import config as cfg
from src import embedder, vector_store


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def make_url(video_id: str, start: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"


def _split_id(chunk_id: str) -> tuple[str, int]:
    video_id, k = chunk_id.rsplit("_", 1)      # video ids can contain underscores
    return video_id, int(k)


def _result(p: dict, score: float, text: str | None = None, end: float | None = None) -> dict:
    return {
        **p,
        "text": text if text is not None else p["text"],
        "end": end if end is not None else p["end"],
        "score": round(score, 4),
        "timestamp": fmt_time(p["start"]),
        "url": make_url(p["video_id"], p["start"]),   # built from data, never from the LLM
    }


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(cfg.RERANK_MODEL)
    return _reranker


def _rerank(question: str, hits: list, keep: int) -> list:
    scores = _get_reranker().predict([(question, h.payload["text"]) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
    return [hits[i] for i in order[:keep]]


def retrieve(question: str, top_k: int | None = None, video_id: str | None = None,
             expand: bool = True, rerank: bool | None = None) -> list[dict]:
    n = top_k or cfg.TOP_K
    use_rerank = cfg.RERANK if rerank is None else rerank

    vec = embedder.embed_query(question)
    if use_rerank:
        candidates = vector_store.search(vec, max(cfg.RERANK_CANDIDATES, n), video_id)
        hits = _rerank(question, candidates, n)          # best first, after reranking
    else:
        hits = vector_store.search(vec, n, video_id)

    if not expand or cfg.NEIGHBORS <= 0:
        return [_result(h.payload, h.score) for h in hits]

    scored = {h.payload["chunk_id"]: (h.payload, h.score) for h in hits}
    rank = {h.payload["chunk_id"]: i for i, h in enumerate(hits)}

    # neighbors (previous/next chunks) of the best hits only
    need: set[str] = set()
    for h in hits[: cfg.EXPAND_TOP_N]:
        vid, k = _split_id(h.payload["chunk_id"])
        for d in range(1, cfg.NEIGHBORS + 1):
            for kk in (k - d, k + d):
                if kk >= 0:
                    need.add(f"{vid}_{kk:04d}")
    need -= set(scored)
    extra = vector_store.get_chunks(sorted(need))

    pool = {cid: p for cid, (p, _) in scored.items()}
    pool.update(extra)

    # group consecutive chunks of the same video into one passage
    by_video: dict[str, list[tuple[int, dict]]] = {}
    for cid, p in pool.items():
        vid, k = _split_id(cid)
        by_video.setdefault(vid, []).append((k, p))

    runs: list[list[tuple[int, dict]]] = []
    for items in by_video.values():
        items.sort(key=lambda x: x[0])
        run = [items[0]]
        for it in items[1:]:
            if it[0] == run[-1][0] + 1:
                run.append(it)
            else:
                runs.append(run)
                run = [it]
        runs.append(run)

    ranked = []
    for run in runs:
        members = [(p, rank[p["chunk_id"]]) for _, p in run if p["chunk_id"] in rank]
        best_p, best_rank = min(members, key=lambda x: x[1])    # every run contains at least one hit
        text = " ".join(p["text"] for _, p in run)
        ranked.append((best_rank, _result(best_p, scored[best_p["chunk_id"]][1],
                                          text=text, end=run[-1][1]["end"])))

    ranked.sort(key=lambda x: x[0])
    return [r for _, r in ranked]
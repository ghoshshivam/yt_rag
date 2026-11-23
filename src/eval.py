import argparse
import json
from pathlib import Path

from src import config as cfg
from src import retriever


def overlaps(chunk: dict, lo: float, hi: float) -> bool:
    return chunk["start"] <= hi and chunk["end"] >= lo


def is_correct(chunk: dict, item: dict) -> bool:
    targets = [item] + item.get("also", [])        # other videos that also answer it
    return any(chunk["video_id"] == t["video_id"] and overlaps(chunk, t["from"], t["to"])
               for t in targets)


def rank_of(item: dict, rerank: bool, kmax: int):
    results = retriever.retrieve(item["q"], top_k=kmax, expand=False, rerank=rerank)
    for i, c in enumerate(results, start=1):
        if is_correct(c, item):
            return i
    return None


def metrics(ranks: list, ks) -> dict:
    n = len(ranks)
    out = {f"Hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    out["MRR"] = sum((1 / r) if r else 0 for r in ranks) / n
    return out


def run(path: str, ks=(1, 3, 5, 10)) -> None:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    kmax = max(ks)
    base = [rank_of(q, False, kmax) for q in questions]
    rer = [rank_of(q, True, kmax) for q in questions]

    fmt = lambda r: str(r) if r else "MISS"
    print("  plain -> rerank | question")
    for q, b, r in zip(questions, base, rer):
        print(f"{fmt(b):>7} -> {fmt(r):>6} | {q['q']}")

    mb, mr = metrics(base, ks), metrics(rer, ks)
    print("\n" + "=" * 46)
    print(f"{'':8}{'plain':>10}{'rerank':>10}")
    for key in mb:
        if key == "MRR":
            print(f"{key:<8}{mb[key]:>10.3f}{mr[key]:>10.3f}")
        else:
            print(f"{key:<8}{mb[key] * 100:>9.0f}%{mr[key] * 100:>9.0f}%")
    print(f"questions: {len(questions)} | embed: {cfg.EMBED_MODEL}")
    print(f"reranker : {cfg.RERANK_MODEL} ({cfg.RERANK_CANDIDATES} candidates)")
    print("=" * 46)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="eval_questions.json")
    run(p.parse_args().file)
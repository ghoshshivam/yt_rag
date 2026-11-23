from src import retriever, generator


def ask(question: str, top_k: int | None = None, video_id: str | None = None) -> dict:
    chunks = retriever.retrieve(question, top_k, video_id)
    try:
        answer = generator.generate_answer(question, chunks)
    except Exception as e:
        answer = (
            f"The language model is unavailable right now ({type(e).__name__}). "
            "The sources below are still the most relevant moments in the lectures."
        )
    sources = [
        {
            "n": i,
            "title": c["video_title"],
            "timestamp": c["timestamp"],
            "start": c["start"],
            "url": c["url"],
            "score": c["score"],
        }
        for i, c in enumerate(chunks, start=1)
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--video", default=None, help="restrict to one video_id")
    p.add_argument("--no-llm", action="store_true", help="show retrieved chunks only (no API key needed)")
    a = p.parse_args()

    if a.no_llm:
        for c in retriever.retrieve(a.question, a.top_k, a.video):
            print(f"[{c['score']}] {c['video_title']} @ {c['timestamp']}\n  {c['url']}\n  {c['text'][:200]}...\n")
    else:
        out = ask(a.question, a.top_k, a.video)
        print(out["answer"], "\n\nSources:")
        for s in out["sources"]:
            print(f"  [{s['n']}] {s['title']} @ {s['timestamp']}  {s['url']}  (score {s['score']})")

import json
from src import config as cfg
from src import embedder, vector_store


def run_indexing() -> None:
    vector_store.ensure_collection()
    files = sorted(cfg.CHUNKS_DIR.glob("*.json"))
    total = 0
    for i, f in enumerate(files, start=1):
        with open(f, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)
        # Prepend the title so topic context is part of the embedding
        texts = [f"{c['video_title']}: {c['text']}" for c in chunks]
        vectors = embedder.embed_texts(texts)
        vector_store.upsert_chunks(chunks, vectors)
        total += len(chunks)
        print(f"[{i}/{len(files)}] {f.stem}: indexed {len(chunks)} chunks")
    print(f"Done. Collection now has {vector_store.count()} points (this run: {total})")


if __name__ == "__main__":
    run_indexing()

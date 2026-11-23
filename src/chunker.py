import json
from src import config as cfg


def build_chunks(transcript: dict) -> list[dict]:
    segs = sorted(transcript["segments"], key=lambda s: s["start"])
    n = len(segs)
    chunks, i, k = [], 0, 0

    while i < n:
        start = segs[i]["start"]
        j = i
        while j < n and (segs[j]["end"] - start) <= cfg.CHUNK_SECONDS:
            j += 1
        if j == i:                      # one segment longer than the window
            j = i + 1

        group = segs[i:j]
        chunks.append({
            "chunk_id": f"{transcript['video_id']}_{k:04d}",
            "video_id": transcript["video_id"],
            "video_title": transcript.get("title", ""),
            "playlist_index": transcript.get("playlist_index"),
            "start": group[0]["start"],
            "end": max(s["end"] for s in group),
            "text": " ".join(s["text"] for s in group),
        })
        k += 1

        if j >= n:
            break

        # next chunk starts about CHUNK_OVERLAP_SECONDS before this one ended
        cutoff = group[-1]["end"] - cfg.CHUNK_OVERLAP_SECONDS
        nxt = j
        while nxt - 1 > i and segs[nxt - 1]["start"] >= cutoff:
            nxt -= 1
        i = nxt                          # always advances by at least one segment

    return chunks


def run_chunking(force: bool = False) -> None:
    files = sorted(cfg.TRANSCRIPTS_DIR.glob("*.json"))
    done = skipped = 0
    for f in files:
        out = cfg.CHUNKS_DIR / f.name
        if out.exists() and not force:
            skipped += 1
            continue
        with open(f, "r", encoding="utf-8") as fh:
            transcript = json.load(fh)
        chunks = build_chunks(transcript)
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False, indent=2)
        tmp.replace(out)
        print(f"{f.stem}: {len(transcript['segments'])} segments -> {len(chunks)} chunks")
        done += 1
    print(f"Summary: chunked={done} skipped={skipped}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    run_chunking(force=p.parse_args().force)
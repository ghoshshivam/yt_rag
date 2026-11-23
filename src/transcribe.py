import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # must be set before faster_whisper import

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel, decode_audio

try:
    import config as cfg            # python src/transcribe.py
except ModuleNotFoundError:
    from src import config as cfg   # python -m src.transcribe

SAMPLE_RATE = 16000


def load_videos() -> list[dict]:
    """Read playlist_meta.json (accepts either {'videos': [...]} or a plain list)."""
    with open(cfg.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["videos"] if isinstance(meta, dict) else meta


def load_model() -> WhisperModel:
    print(
        f"Loading model={cfg.WHISPER_MODEL} device={cfg.WHISPER_DEVICE} "
        f"compute={cfg.WHISPER_COMPUTE} language={cfg.WHISPER_LANGUAGE} task={cfg.WHISPER_TASK}"
    )
    return WhisperModel(
        cfg.WHISPER_MODEL,
        device=cfg.WHISPER_DEVICE,
        compute_type=cfg.WHISPER_COMPUTE,
        cpu_threads=cfg.WHISPER_CPU_THREADS,
    )


def transcribe_video(audio_path: Path, model: WhisperModel):
    """Transcribe in fixed windows to keep RAM flat, then shift timestamps back to video time.
    Returns (segments, info) where info.duration is the total audio length in seconds."""
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    total_seconds = len(audio) / SAMPLE_RATE
    window_samples = cfg.WHISPER_WINDOW_SECONDS * SAMPLE_RATE
    n_windows = (len(audio) + window_samples - 1) // window_samples

    segments = []
    for w, begin in enumerate(range(0, len(audio), window_samples), start=1):
        piece = audio[begin:begin + window_samples]
        if len(piece) < SAMPLE_RATE:          # skip a trailing sliver under 1 second
            continue
        offset = begin / SAMPLE_RATE

        segments_iter, _ = model.transcribe(
            piece,
            language=cfg.WHISPER_LANGUAGE,
            task=cfg.WHISPER_TASK,
            beam_size=cfg.WHISPER_BEAM_SIZE,
            vad_filter=cfg.WHISPER_VAD_FILTER,
            initial_prompt=cfg.WHISPER_INITIAL_PROMPT,
            condition_on_previous_text=False,  # reduces repeated/looping text
        )
        for s in segments_iter:                # generator must be consumed
            text = s.text.strip()
            if not text:
                continue
            segments.append({
                "start": round(s.start + offset, 2),
                "end": round(s.end + offset, 2),
                "text": text,
            })
        print(f"    window {w}/{n_windows} done ({len(segments)} segments so far)")

    return segments, SimpleNamespace(duration=total_seconds)


def save_transcript(video: dict, segments: list[dict], info, out_dir: Path) -> Path:
    """Atomic write: write .tmp then rename, so a crash never leaves a half-written JSON."""
    out_path = out_dir / f"{video['video_id']}.json"
    tmp_path = out_dir / f"{video['video_id']}.json.tmp"
    data = {
        "video_id": video["video_id"],
        "title": video.get("title", ""),
        "playlist_index": video.get("playlist_index"),
        "language": cfg.WHISPER_LANGUAGE,
        "task": cfg.WHISPER_TASK,
        "model": cfg.WHISPER_MODEL,
        "duration": round(info.duration, 2),
        "segments": segments,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)
    return out_path


def run_transcription(limit: int | None = None, force: bool = False) -> None:
    videos = load_videos()
    if limit:
        videos = videos[:limit]

    model = load_model()  # load ONCE, outside the loop
    done, skipped, failed = 0, 0, []
    total = len(videos)

    for i, video in enumerate(videos, start=1):
        vid = video["video_id"]
        title = video.get("title", "")
        audio_path = cfg.AUDIO_DIR / f"{vid}.{cfg.AUDIO_FORMAT}"
        out_path = cfg.TRANSCRIPTS_DIR / f"{vid}.json"

        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            print(f"[{i}/{total}] SKIP (exists) {title}")
            skipped += 1
            continue

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            print(f"[{i}/{total}] SKIP (no audio) {title}")
            failed.append({"video_id": vid, "title": title, "error": "audio file missing"})
            continue

        try:
            t0 = time.time()
            segments, info = transcribe_video(audio_path, model)
            if not segments:
                raise RuntimeError("no segments produced")
            save_transcript(video, segments, info, cfg.TRANSCRIPTS_DIR)
            elapsed = time.time() - t0
            speed = info.duration / elapsed if elapsed else 0
            print(
                f"[{i}/{total}] OK {title} | {len(segments)} segments | "
                f"{elapsed:.0f}s for {info.duration:.0f}s audio ({speed:.1f}x realtime)"
            )
            done += 1
        except Exception as e:  # KeyboardInterrupt still stops the run
            print(f"[{i}/{total}] FAILED {title}: {e}")
            failed.append({"video_id": vid, "title": title, "error": str(e)})

    if failed:
        with open(cfg.FAILED_TRANSCRIPTS_PATH, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
    elif cfg.FAILED_TRANSCRIPTS_PATH.exists():
        cfg.FAILED_TRANSCRIPTS_PATH.unlink()

    print(f"\nSummary: done={done} skipped={skipped} failed={len(failed)}")
    if failed:
        print(f"Failures written to {cfg.FAILED_TRANSCRIPTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: transcribe audio with faster-whisper")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N videos")
    parser.add_argument("--force", action="store_true", help="re-transcribe even if JSON exists")
    args = parser.parse_args()
    run_transcription(limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
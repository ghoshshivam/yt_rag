import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import yt_dlp

from config import (
    AUDIO_BITRATE,
    AUDIO_DIR,
    AUDIO_FORMAT,
    CAPTIONS_DIR,
    COOKIES_BROWSER,
    DEFAULT_PLAYLIST_URL,
    FAILED_PATH,
    META_PATH,
)

# Configure clean, timestamped logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def _base_ydl_opts(use_default_clients: bool = False) -> dict[str, Any]:
    """Return shared yt-dlp options, including browser cookies if configured."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,  # give up on a stalled connection instead of hanging forever
    }
    if not use_default_clients:
        opts["extractor_args"] = {"youtube": {"player_client": ["mweb", "android_vr"]}}
    if COOKIES_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_BROWSER,)
    return opts


def fetch_playlist_meta(playlist_url: str) -> dict[str, Any]:
    """Fetch playlist metadata and video list without downloading media."""
    logger.info("Fetching playlist metadata...")
    ydl_opts = {
        **_base_ydl_opts(),
        "extract_flat": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if not info or "entries" not in info:
        raise ValueError("Could not extract entries. Check if playlist is public/unlisted.")

    videos: list[dict[str, Any]] = []
    # Use enumerate(..., start=1) as the source of truth for playlist_index
    for idx, entry in enumerate(info["entries"], start=1):
        if entry is None:
            logger.warning(f"Skipping unavailable/private video at position {idx}")
            continue

        video_id = entry.get("id")
        if not video_id:
            logger.warning(f"Skipping entry without video_id at position {idx}")
            continue

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title") or f"Untitled_{video_id}",
                "playlist_index": idx,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration": entry.get("duration"),  # Optional in flat mode
            }
        )

    playlist_data = {
        "playlist_url": playlist_url,
        "playlist_title": info.get("title", "Unknown Playlist"),
        "total_available": len(videos),
        "videos": videos,
    }
    logger.info(f"Found {len(videos)} available videos in '{playlist_data['playlist_title']}'.")
    return playlist_data


def save_meta(playlist_data: dict[str, Any], path: Path) -> None:
    """Persist playlist metadata to JSON as the single source of truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(playlist_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {path}")


def download_audio(video: dict[str, Any], out_dir: Path) -> tuple[str, Path | None]:
    """
    Download and convert a single video's audio to MP3.
    Prefers the original (non-dubbed) audio track, tries the forced clients first,
    then falls back to yt-dlp's default clients.
    Returns status ('skipped', 'downloaded', 'failed') and the file path.
    """
    video_id = video["video_id"]
    expected_path = out_dir / f"{video_id}.{AUDIO_FORMAT}"

    # Resumability check: skip if file exists and is non-empty
    if expected_path.exists() and expected_path.stat().st_size > 0:
        return "skipped", expected_path

    last_error: Any = None
    for use_default in (False, True):
        ydl_opts = {
            **_base_ydl_opts(use_default_clients=use_default),
            "format": "bestaudio[format_note*=original]/bestaudio*/best",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "noprogress": True,
            "retries": 5,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": AUDIO_FORMAT,
                    "preferredquality": AUDIO_BITRATE,
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video["url"]])

            if expected_path.exists() and expected_path.stat().st_size > 0:
                return "downloaded", expected_path
            last_error = "download finished but output file is missing/empty"
        except Exception as exc:
            last_error = exc

        if not use_default:
            logger.warning(f"[{video_id}] first attempt failed, retrying with default clients")

    logger.error(f"[{video_id}] Failed to download '{video['title']}': {last_error}")
    return "failed", None


def download_captions(video: dict[str, Any], out_dir: Path) -> tuple[str, Path | None]:
    """Optional Step 9 fallback: fetch English auto-captions in json3 format."""
    video_id = video["video_id"]
    expected_path = out_dir / f"{video_id}.en.json3"

    if expected_path.exists() and expected_path.stat().st_size > 0:
        return "skipped", expected_path

    last_error: Any = None
    for use_default in (False, True):
        ydl_opts = {
            **_base_ydl_opts(use_default_clients=use_default),
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitlesformat": "json3",
            "subtitleslangs": ["en"],
            "outtmpl": str(out_dir / "%(id)s"),
            "noprogress": True,
            "retries": 5,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video["url"]])

            if expected_path.exists() and expected_path.stat().st_size > 0:
                return "downloaded", expected_path
            last_error = "no English captions available"
        except Exception as exc:
            last_error = exc

        if not use_default:
            logger.warning(f"[{video_id}] caption first attempt failed, retrying with default clients")

    logger.warning(f"[{video_id}] Caption fetch failed: {last_error}")
    return "failed", None


def run_ingestion(playlist_url: str, limit: int | None = None, captions_only: bool = False) -> None:
    """Main driver orchestrating metadata fetch, batch download, and failure tracking."""
    playlist_data = fetch_playlist_meta(playlist_url)
    save_meta(playlist_data, META_PATH)

    videos = playlist_data["videos"]
    if limit is not None and limit > 0:
        videos = videos[:limit]
        logger.info(f"Applying --limit {limit}: processing first {len(videos)} videos.")

    total = len(videos)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed_videos: list[dict[str, Any]] = []

    for i, video in enumerate(videos, start=1):
        prefix = f"[{i}/{total}]"
        if captions_only:
            status, _ = download_captions(video, CAPTIONS_DIR)
        else:
            status, _ = download_audio(video, AUDIO_DIR)

        stats[status] += 1

        if status == "skipped":
            logger.info(f"{prefix} SKIPPED (exists): {video['video_id']} - {video['title']}")
        elif status == "downloaded":
            logger.info(f"{prefix} DOWNLOADED: {video['video_id']} - {video['title']}")
            # Polite delay only when an actual network download occurred
            if i < total:
                delay = random.uniform(1.0, 3.0)
                time.sleep(delay)
        else:
            failed_videos.append(video)

    # Save failed items for targeted inspection/retry
    if failed_videos:
        with FAILED_PATH.open("w", encoding="utf-8") as f:
            json.dump(failed_videos, f, indent=2, ensure_ascii=False)
        logger.warning(f"Saved {len(failed_videos)} failed entries to {FAILED_PATH}")
    elif FAILED_PATH.exists():
        FAILED_PATH.unlink()  # Clean up stale failure log if everything succeeded

    # Final Summary
    print("\n" + "=" * 50)
    print("PHASE 1 INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total Processed : {total}")
    print(f"Downloaded      : {stats['downloaded']}")
    print(f"Skipped         : {stats['skipped']}")
    print(f"Failed          : {stats['failed']}")
    if failed_videos:
        print("Failed IDs      :", ", ".join(v["video_id"] for v in failed_videos))
    print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: YouTube Playlist Audio Ingestion")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_PLAYLIST_URL,
        help="YouTube playlist URL (defaults to PLAYLIST_URL in .env)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N videos (useful for testing)",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Fetch English captions instead of downloading audio",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_ingestion(
            playlist_url=args.url,
            limit=args.limit,
            captions_only=args.captions_only,
        )
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user.")
        sys.exit(130)

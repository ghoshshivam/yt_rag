import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env at project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Data directories
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
CAPTIONS_DIR = DATA_DIR / "captions"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
META_PATH = DATA_DIR / "playlist_meta.json"
FAILED_PATH = DATA_DIR / "failed_downloads.json"
FAILED_TRANSCRIPTS_PATH = DATA_DIR / "failed_transcripts.json"

# Ensure directories exist automatically when config is imported
for _d in (AUDIO_DIR, CAPTIONS_DIR, TRANSCRIPTS_DIR, CHUNKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Audio extraction settings (64kbps mono/speech optimal)
AUDIO_FORMAT = "mp3"
AUDIO_BITRATE = "64"

# Runtime configuration
DEFAULT_PLAYLIST_URL = os.getenv("PLAYLIST_URL", "")
COOKIES_BROWSER = os.getenv("COOKIES_BROWSER", "").strip() or None

# --- Phase 2: Transcription (faster-whisper) ---
# Override any of these from .env, so switching between laptop and Colab
# needs no code change.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")       # Colab: "medium" or "large-v3"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")       # Colab: "cuda"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")    # Colab: "float16"
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))

# Hindi lecture audio -> English text (tested: clean on small, transcribe mode was garbage)
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "hi")
WHISPER_TASK = os.getenv("WHISPER_TASK", "translate")     # "translate" or "transcribe"

WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = True
WHISPER_INITIAL_PROMPT = (
    "Data structures and algorithms: array, linked list, recursion, "
    "time complexity, Big O, stack, queue, tree, graph, dynamic programming."
)

# Long videos are transcribed in fixed windows so RAM stays flat (fixes the
# "Unable to allocate ... MiB" error on 40+ minute lectures)
WHISPER_WINDOW_SECONDS = int(os.getenv("WHISPER_WINDOW_SECONDS", "600"))

# --- Phase 3: Chunking ---
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "75"))
CHUNK_OVERLAP_SECONDS = int(os.getenv("CHUNK_OVERLAP_SECONDS", "15"))  # time-based overlap

# --- Phase 4: Embeddings + Qdrant ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))   # must match the model
EMBED_BATCH = 32
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: queries only
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()  # empty = embedded local mode (no Docker)
QDRANT_PATH = DATA_DIR / "qdrant_local"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dsa_playlist")

# --- Phase 5: Retrieval + LLM ---
TOP_K = int(os.getenv("TOP_K", "6"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",")
    if m.strip()
]
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))

# --- Phase 8: neighbor expansion ---
NEIGHBORS = int(os.getenv("NEIGHBORS", "1"))           # chunks before/after each top hit
EXPAND_TOP_N = int(os.getenv("EXPAND_TOP_N", "4"))     # only expand the best N hits

# --- Phase 8: reranker ---
RERANK = os.getenv("RERANK", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
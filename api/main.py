from pathlib import Path
from fastapi.responses import FileResponse

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config as cfg
from src import embedder, rag, vector_store

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.get_model()      # load the embedding model once at startup, not on the first request
    yield


app = FastAPI(title="DSA Playlist RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],   # Vite and CRA dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    video_id: str | None = None


class Source(BaseModel):
    n: int
    title: str
    timestamp: str
    start: float
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    try:
        points = vector_store.count()
    except Exception:
        points = None
    return {"status": "ok", "points": points}


@app.get("/videos")
def videos():
    """List of videos, for a 'search only in this video' dropdown in the frontend."""
    with open(cfg.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return [
        {"video_id": v["video_id"], "title": v["title"], "playlist_index": v["playlist_index"]}
        for v in meta["videos"]
    ]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag.ask(req.question.strip(), req.top_k, req.video_id)
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")    
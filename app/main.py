"""FastAPI application that exposes Whisper transcription endpoints."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .whisper_service import TranscriptionError, TranscriptionService

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Memo Notes Transcriber",
    description="Upload an audio file and receive a transcript powered by OpenAI's Whisper.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_directory = BASE_DIR / "static"
if static_directory.exists():
    app.mount("/static", StaticFiles(directory=static_directory), name="static")

service = TranscriptionService()


@app.on_event("startup")
async def warm_up_model() -> None:
    """Warm the Whisper model so first request latency stays low."""

    try:
        await service.warm_up()
    except TranscriptionError as exc:  # pragma: no cover - best effort warm up
        LOGGER.warning("Could not warm Whisper model on startup: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the single-page front-end."""

    index_path = static_directory / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return index_path.read_text(encoding="utf-8")


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, str]:
    """Accept an uploaded file and return its transcription."""

    try:
        transcript = await service.transcribe_upload(file)
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"text": transcript}

"""Utilities for working with OpenAI's Whisper models."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - import guard
    whisper = None  # type: ignore


class TranscriptionError(RuntimeError):
    """Raised when the transcription pipeline cannot process an audio file."""


class TranscriptionService:
    """Service wrapper that handles Whisper model lifecycle and transcription."""

    def __init__(self, model_name: str = "base", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device_preference = device
        self._model = None
        self._load_error: Optional[TranscriptionError] = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model_loaded(self) -> None:
        """Load the Whisper model if it has not been initialised yet."""

        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        async with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise self._load_error

            if whisper is None:
                self._load_error = TranscriptionError(
                    "openai-whisper is not installed. Install it with 'pip install openai-whisper'."
                )
                raise self._load_error

            try:
                import torch  # type: ignore
            except Exception:
                torch = None  # type: ignore

            if self.device_preference:
                device = self.device_preference
            elif torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
                device = "cuda"
            else:
                device = "cpu"

            loop = asyncio.get_running_loop()

            try:
                self._model = await loop.run_in_executor(
                    None, lambda: whisper.load_model(self.model_name, device=device)
                )
            except Exception as exc:  # pragma: no cover - dependent on environment
                self._load_error = TranscriptionError(str(exc))
                raise self._load_error

    async def warm_up(self) -> None:
        """Ensure the underlying Whisper model is ready to serve requests."""

        await self._ensure_model_loaded()

    async def transcribe_upload(self, upload: UploadFile) -> str:
        """Transcribe an uploaded audio file and return the detected text."""

        if not upload.filename:
            raise TranscriptionError("Uploaded file must have a filename.")

        await self._ensure_model_loaded()

        suffix = Path(upload.filename).suffix or ".mp3"
        loop = asyncio.get_running_loop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            file_bytes = await upload.read()
            if not file_bytes:
                raise TranscriptionError("The uploaded file is empty.")
            temp_file.write(file_bytes)

        try:
            result = await loop.run_in_executor(None, lambda: self._model.transcribe(temp_path))
        except Exception as exc:  # pragma: no cover - dependent on environment
            raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        text = result.get("text") if isinstance(result, dict) else None
        if not text:
            raise TranscriptionError("Unable to extract transcription from model output.")

        return text.strip()

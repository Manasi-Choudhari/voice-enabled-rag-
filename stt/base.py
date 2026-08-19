"""Abstract base class for STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class STTProvider(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audioPath: Path | str, language: str = "hi-IN") -> dict[str, Any]:
        """Transcribe audio file to text.

        Returns:
            dict with keys: text (str), language (str), confidence (float), latency_ms (float), error (str|None)
        """

    def transcribeBytes(self, audioBytes: bytes, language: str = "unknown", strFormat: str = "wav") -> dict[str, Any]:
        """Transcribe raw audio bytes. Default implementation writes a temp file."""
        import tempfile

        strSuffix = strFormat.lstrip(".")
        with tempfile.NamedTemporaryFile(suffix=f".{strSuffix}", delete=False) as f:
            f.write(audioBytes)
            f.flush()
            return self.transcribe(Path(f.name), language=language)

"""ElevenLabs Speech-to-Text provider."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from stt.base import STTProvider

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT(STTProvider):
    name = "elevenlabs"

    def __init__(self, strApiKey: str | None = None) -> None:
        self.strApiKey = strApiKey or os.getenv("ELEVENLABS_API_KEY", "")

    def transcribe(self, audioPath: Path | str, language: str = "hi-IN") -> dict[str, Any]:
        if not self.strApiKey:
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "latency_ms": 0,
                "error": "ELEVENLABS_API_KEY not set",
            }

        audioPath = Path(audioPath)
        floatStart = time.perf_counter()
        try:
            with open(audioPath, "rb") as f:
                response = requests.post(
                    ELEVENLABS_STT_URL,
                    headers={"xi-api-key": self.strApiKey},
                    files={"file": (audioPath.name, f)},
                    data={"language_code": language[:2]},
                    timeout=30,
                )
            floatLatencyMs = (time.perf_counter() - floatStart) * 1000

            if response.status_code != 200:
                return {
                    "text": "",
                    "language": language,
                    "confidence": 0.0,
                    "latency_ms": round(floatLatencyMs, 1),
                    "error": f"ElevenLabs API error {response.status_code}: {response.text[:200]}",
                }

            dictResponse = response.json()
            return {
                "text": dictResponse.get("text", ""),
                "language": language,
                "confidence": 0.9,
                "latency_ms": round(floatLatencyMs, 1),
                "error": None,
            }

        except Exception as exc:
            floatLatencyMs = (time.perf_counter() - floatStart) * 1000
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "latency_ms": round(floatLatencyMs, 1),
                "error": str(exc),
            }

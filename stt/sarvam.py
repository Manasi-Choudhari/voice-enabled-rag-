"""Sarvam AI Speech-to-Text provider (Saaras v3 REST)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from stt.base import STTProvider

load_dotenv()

logger = logging.getLogger("stt.sarvam")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(STTProvider):
    name = "sarvam"

    def __init__(self, strApiKey: str | None = None) -> None:
        self.strApiKey = strApiKey or os.getenv("SARVAM_API_KEY", "")

    def _postOnce(
        self,
        audioPath: Path,
        strMimeType: str,
        strMode: str,
        strLanguage: str,
    ) -> tuple[int, dict[str, Any], float]:
        dictForm = {
            "model": "saaras:v3",
            "mode": strMode,
            "language_code": strLanguage or "unknown",
        }
        floatStart = time.perf_counter()
        with open(audioPath, "rb") as f:
            response = requests.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self.strApiKey},
                files={"file": (audioPath.name, f, strMimeType)},
                data=dictForm,
                timeout=45,
            )
        floatLatencyMs = (time.perf_counter() - floatStart) * 1000
        try:
            dictBody = response.json()
        except Exception:
            dictBody = {"raw": response.text[:500]}
        return response.status_code, dictBody, floatLatencyMs

    def transcribe(self, audioPath: Path | str, language: str = "unknown") -> dict[str, Any]:
        if not self.strApiKey:
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "latency_ms": 0,
                "error": "SARVAM_API_KEY not set",
            }

        audioPath = Path(audioPath)
        intSize = audioPath.stat().st_size if audioPath.exists() else 0
        if intSize < 2000:
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "latency_ms": 0,
                "error": f"audio_too_short ({intSize} bytes). Record at least 2 seconds.",
            }

        strSuffix = audioPath.suffix.lower()
        dictMime = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".m4a": "audio/mp4",
        }
        strMimeType = dictMime.get(strSuffix, "application/octet-stream")
        strLanguage = language if language else "unknown"

        floatTotal = 0.0
        strLastError = None
        try:
            for strMode in ("transcribe", "translate"):
                intStatus, dictBody, floatLatencyMs = self._postOnce(
                    audioPath, strMimeType, strMode, strLanguage
                )
                floatTotal += floatLatencyMs
                logger.info(
                    "Sarvam %s status=%s bytes=%s body=%s",
                    strMode,
                    intStatus,
                    intSize,
                    str(dictBody)[:400],
                )
                if intStatus != 200:
                    strLastError = f"Sarvam API error {intStatus}: {str(dictBody)[:300]}"
                    continue
                strText = str(
                    dictBody.get("transcript") or dictBody.get("text") or ""
                ).strip()
                if strText:
                    return {
                        "text": strText,
                        "language": dictBody.get("language_code") or strLanguage,
                        "confidence": float(dictBody.get("language_probability") or 0.8),
                        "latency_ms": round(floatTotal, 1),
                        "error": None,
                    }
            return {
                "text": "",
                "language": strLanguage,
                "confidence": 0.0,
                "latency_ms": round(floatTotal, 1),
                "error": strLastError or "empty_transcript (speak clearly for 2+ seconds)",
            }
        except Exception as exc:
            return {
                "text": "",
                "language": strLanguage,
                "confidence": 0.0,
                "latency_ms": round(floatTotal, 1),
                "error": str(exc),
            }

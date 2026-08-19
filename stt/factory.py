"""STT provider factory."""

from __future__ import annotations

import os

from stt.base import STTProvider


def getSTTProvider(strProvider: str | None = None) -> STTProvider:
    strProvider = strProvider or os.getenv("STT_PROVIDER", "sarvam")
    strProvider = strProvider.strip().lower()

    if strProvider == "sarvam":
        from stt.sarvam import SarvamSTT
        return SarvamSTT()
    elif strProvider == "elevenlabs":
        from stt.elevenlabs import ElevenLabsSTT
        return ElevenLabsSTT()
    else:
        raise ValueError(f"Unknown STT provider: {strProvider!r}. Use 'sarvam' or 'elevenlabs'.")

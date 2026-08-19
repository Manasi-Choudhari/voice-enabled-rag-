"""Embedding model wrapper (singleton for reuse)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def getModel() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embedTexts(listTexts: list[str]) -> np.ndarray:
    model = getModel()
    return model.encode(
        listTexts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embedQuery(strQuery: str) -> np.ndarray:
    return embedTexts([strQuery])[0]

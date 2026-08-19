"""Chunking strategy registry and factory."""

from __future__ import annotations

from typing import Any

from chunking.base import ChunkingStrategy
from chunking.fixed_size import FixedSizeChunker
from chunking.hierarchical import HierarchicalChunker
from chunking.metadata_aware import MetadataAwareChunker
from chunking.semantic import SemanticChunker
from chunking.sentence_boundary import SentenceBoundaryChunker

STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "fixed_size": FixedSizeChunker,
    "sentence_boundary": SentenceBoundaryChunker,
    "semantic": SemanticChunker,
    "metadata_aware": MetadataAwareChunker,
    "hierarchical": HierarchicalChunker,
}


def getStrategy(strName: str, dictOptions: dict[str, Any] | None = None) -> ChunkingStrategy:
    dictOptions = dictOptions or {}
    if strName not in STRATEGY_REGISTRY:
        listNames = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy {strName!r}. Available: {listNames}")
    return STRATEGY_REGISTRY[strName](**dictOptions)


def listStrategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY.keys())

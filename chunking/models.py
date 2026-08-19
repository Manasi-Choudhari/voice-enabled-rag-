"""Shared chunking data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunkId: str
    text: str
    docId: str
    passageIndex: int
    queryId: int
    queryType: str | None = None
    chunkIndex: int = 0
    parentChunkId: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def embeddingText(self) -> str:
        """Text used for dense retrieval (may include metadata prefix)."""
        strMetaPrefix = self.metadata.get("embedding_prefix", "")
        if strMetaPrefix:
            return f"{strMetaPrefix}\n{self.text}"
        return self.text


@dataclass(frozen=True)
class PassageRecord:
    queryId: int
    passageIndex: int
    text: str
    queryType: str | None
    isSelected: bool
    engText: str | None = None

    @property
    def docId(self) -> str:
        return f"{self.queryId}_{self.passageIndex}"

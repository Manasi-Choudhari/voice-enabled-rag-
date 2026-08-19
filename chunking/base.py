"""Abstract chunking strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chunking.models import Chunk, PassageRecord


class ChunkingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        """Split a single passage into zero or more chunks."""

    def chunkCorpus(self, listPassages: list[PassageRecord]) -> list[Chunk]:
        listChunks: list[Chunk] = []
        for passage in listPassages:
            listChunks.extend(self.chunkPassage(passage))
        return listChunks

    def config(self) -> dict[str, Any]:
        return {"strategy": self.name}

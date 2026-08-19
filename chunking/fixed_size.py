"""Fixed-size character window chunking with overlap."""

from __future__ import annotations

from chunking.base import ChunkingStrategy
from chunking.models import Chunk, PassageRecord
from chunking.utils import windowByChars


class FixedSizeChunker(ChunkingStrategy):
    name = "fixed_size"

    def __init__(self, intChunkSize: int = 512, intOverlap: int = 64) -> None:
        self.intChunkSize = intChunkSize
        self.intOverlap = intOverlap

    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        listTexts = windowByChars(passage.text, self.intChunkSize, self.intOverlap)
        return [
            Chunk(
                chunkId=f"{passage.docId}_fixed_{intIndex}",
                text=strText,
                docId=passage.docId,
                passageIndex=passage.passageIndex,
                queryId=passage.queryId,
                queryType=passage.queryType,
                chunkIndex=intIndex,
                metadata={"strategy": self.name, "chunk_size": self.intChunkSize},
            )
            for intIndex, strText in enumerate(listTexts)
        ]

    def config(self) -> dict:
        return {
            "strategy": self.name,
            "chunk_size": self.intChunkSize,
            "overlap": self.intOverlap,
        }

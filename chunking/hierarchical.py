"""Hierarchical parent-child chunking for precision retrieval + generation context."""

from __future__ import annotations

from chunking.base import ChunkingStrategy
from chunking.models import Chunk, PassageRecord
from chunking.utils import windowByChars


class HierarchicalChunker(ChunkingStrategy):
    name = "hierarchical"

    def __init__(
        self,
        intChildSize: int = 256,
        intChildOverlap: int = 32,
    ) -> None:
        self.intChildSize = intChildSize
        self.intChildOverlap = intChildOverlap

    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        strParentText = passage.text.strip()
        if not strParentText:
            return []

        strParentId = f"{passage.docId}_parent"
        parentChunk = Chunk(
            chunkId=strParentId,
            text=strParentText,
            docId=passage.docId,
            passageIndex=passage.passageIndex,
            queryId=passage.queryId,
            queryType=passage.queryType,
            chunkIndex=0,
            metadata={"strategy": self.name, "level": "parent"},
        )

        listChildTexts = windowByChars(
            strParentText,
            self.intChildSize,
            self.intChildOverlap,
        )
        listChunks: list[Chunk] = [parentChunk]
        for intIndex, strChildText in enumerate(listChildTexts):
            listChunks.append(
                Chunk(
                    chunkId=f"{passage.docId}_child_{intIndex}",
                    text=strChildText,
                    docId=passage.docId,
                    passageIndex=passage.passageIndex,
                    queryId=passage.queryId,
                    queryType=passage.queryType,
                    chunkIndex=intIndex + 1,
                    parentChunkId=strParentId,
                    metadata={"strategy": self.name, "level": "child"},
                )
            )
        return listChunks

    def config(self) -> dict:
        return {
            "strategy": self.name,
            "child_size": self.intChildSize,
            "child_overlap": self.intChildOverlap,
        }

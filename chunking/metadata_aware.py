"""Metadata-aware chunking with indexed/filterable metadata per chunk."""

from __future__ import annotations

from chunking.base import ChunkingStrategy
from chunking.sentence_boundary import SentenceBoundaryChunker
from chunking.models import Chunk, PassageRecord


class MetadataAwareChunker(ChunkingStrategy):
    name = "metadata_aware"

    def __init__(self, intMaxChars: int = 512) -> None:
        self.boundaryChunker = SentenceBoundaryChunker(intMaxChars=intMaxChars)
        self.intMaxChars = intMaxChars

    def _buildEmbeddingPrefix(self, passage: PassageRecord) -> str:
        listTags = [
            f"query_type:{passage.queryType or 'UNKNOWN'}",
            f"query_id:{passage.queryId}",
            f"passage_index:{passage.passageIndex}",
            f"language:translated",
        ]
        return " | ".join(listTags)

    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        listBaseChunks = self.boundaryChunker.chunkPassage(passage)
        strPrefix = self._buildEmbeddingPrefix(passage)
        listChunks: list[Chunk] = []

        for chunk in listBaseChunks:
            dictMetadata = {
                "strategy": self.name,
                "query_type": passage.queryType,
                "query_id": passage.queryId,
                "passage_index": passage.passageIndex,
                "doc_id": passage.docId,
                "is_selected": passage.isSelected,
                "embedding_prefix": strPrefix,
                "filter_query_type": passage.queryType,
            }
            listChunks.append(
                Chunk(
                    chunkId=f"{passage.docId}_meta_{chunk.chunkIndex}",
                    text=chunk.text,
                    docId=passage.docId,
                    passageIndex=passage.passageIndex,
                    queryId=passage.queryId,
                    queryType=passage.queryType,
                    chunkIndex=chunk.chunkIndex,
                    metadata=dictMetadata,
                )
            )
        return listChunks

    def config(self) -> dict:
        return {
            "strategy": self.name,
            "max_chars": self.intMaxChars,
            "indexed_fields": ["query_type", "query_id", "passage_index"],
        }

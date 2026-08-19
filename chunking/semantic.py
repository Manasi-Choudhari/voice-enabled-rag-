"""Semantic chunking via consecutive sentence embedding similarity breakpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chunking.base import ChunkingStrategy
from chunking.models import Chunk, PassageRecord
from chunking.utils import splitSentences

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SemanticChunker(ChunkingStrategy):
    name = "semantic"

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        strModelName: str = "sentence-transformers/all-MiniLM-L6-v2",
        floatSimilarityThreshold: float = 0.55,
        intMaxSentencesPerChunk: int = 8,
        intMinSentencesPerChunk: int = 1,
    ) -> None:
        self.model = model
        self.strModelName = strModelName
        self.floatSimilarityThreshold = floatSimilarityThreshold
        self.intMaxSentencesPerChunk = intMaxSentencesPerChunk
        self.intMinSentencesPerChunk = intMinSentencesPerChunk
        self._modelLoaded: SentenceTransformer | None = None

    def _getModel(self) -> SentenceTransformer:
        if self.model is not None:
            return self.model
        if self._modelLoaded is None:
            from sentence_transformers import SentenceTransformer

            self._modelLoaded = SentenceTransformer(self.strModelName)
        return self._modelLoaded

    def _groupSentences(self, listSentences: list[str]) -> list[str]:
        if len(listSentences) <= self.intMinSentencesPerChunk:
            return [" ".join(listSentences)] if listSentences else []

        model = self._getModel()
        arrayEmbeddings = model.encode(
            listSentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        listGroups: list[list[str]] = [[listSentences[0]]]
        for intIndex in range(1, len(listSentences)):
            arrayPrev = arrayEmbeddings[intIndex - 1]
            arrayCurr = arrayEmbeddings[intIndex]
            floatSimilarity = float(np.dot(arrayPrev, arrayCurr))
            boolShouldSplit = (
                floatSimilarity < self.floatSimilarityThreshold
                or len(listGroups[-1]) >= self.intMaxSentencesPerChunk
            )
            if boolShouldSplit:
                listGroups.append([listSentences[intIndex]])
            else:
                listGroups[-1].append(listSentences[intIndex])

        return [" ".join(group) for group in listGroups if group]

    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        listSentences = splitSentences(passage.text)
        if not listSentences:
            return []
        if len(listSentences) == 1:
            listTexts = listSentences
        else:
            listTexts = self._groupSentences(listSentences)

        return [
            Chunk(
                chunkId=f"{passage.docId}_sem_{intIndex}",
                text=strText,
                docId=passage.docId,
                passageIndex=passage.passageIndex,
                queryId=passage.queryId,
                queryType=passage.queryType,
                chunkIndex=intIndex,
                metadata={
                    "strategy": self.name,
                    "similarity_threshold": self.floatSimilarityThreshold,
                },
            )
            for intIndex, strText in enumerate(listTexts)
        ]

    def config(self) -> dict:
        return {
            "strategy": self.name,
            "model": self.strModelName,
            "similarity_threshold": self.floatSimilarityThreshold,
            "max_sentences_per_chunk": self.intMaxSentencesPerChunk,
        }

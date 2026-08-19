"""Sentence and paragraph boundary aware chunking."""

from __future__ import annotations

from chunking.base import ChunkingStrategy
from chunking.models import Chunk, PassageRecord
from chunking.utils import splitParagraphs, splitSentences


class SentenceBoundaryChunker(ChunkingStrategy):
    name = "sentence_boundary"

    def __init__(
        self,
        intMaxChars: int = 512,
        boolPreferParagraphs: bool = True,
    ) -> None:
        self.intMaxChars = intMaxChars
        self.boolPreferParagraphs = boolPreferParagraphs

    def _groupUnits(self, listUnits: list[str]) -> list[str]:
        listGroups: list[str] = []
        strCurrent = ""
        for strUnit in listUnits:
            if not strCurrent:
                strCurrent = strUnit
                continue
            if len(strCurrent) + 1 + len(strUnit) <= self.intMaxChars:
                strCurrent = f"{strCurrent} {strUnit}"
            else:
                listGroups.append(strCurrent)
                strCurrent = strUnit
        if strCurrent:
            listGroups.append(strCurrent)
        return listGroups

    def chunkPassage(self, passage: PassageRecord) -> list[Chunk]:
        if self.boolPreferParagraphs:
            listUnits = splitParagraphs(passage.text)
            if len(listUnits) == 1:
                listUnits = splitSentences(passage.text)
        else:
            listUnits = splitSentences(passage.text)

        listTexts = self._groupUnits(listUnits)
        if not listTexts and passage.text.strip():
            listTexts = [passage.text.strip()]

        return [
            Chunk(
                chunkId=f"{passage.docId}_sent_{intIndex}",
                text=strText,
                docId=passage.docId,
                passageIndex=passage.passageIndex,
                queryId=passage.queryId,
                queryType=passage.queryType,
                chunkIndex=intIndex,
                metadata={"strategy": self.name, "max_chars": self.intMaxChars},
            )
            for intIndex, strText in enumerate(listTexts)
        ]

    def config(self) -> dict:
        return {
            "strategy": self.name,
            "max_chars": self.intMaxChars,
            "prefer_paragraphs": self.boolPreferParagraphs,
        }

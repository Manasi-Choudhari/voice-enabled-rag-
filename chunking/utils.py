"""Text splitting utilities for chunking strategies."""

from __future__ import annotations

import re

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?。！？])\s+|\n{2,}"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalizeWhitespace(strText: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", strText).strip()


def splitSentences(strText: str) -> list[str]:
    strClean = normalizeWhitespace(strText)
    if not strClean:
        return []
    listParts = SENTENCE_BOUNDARY_PATTERN.split(strClean)
    return [part.strip() for part in listParts if part.strip()]


def splitParagraphs(strText: str) -> list[str]:
    listParagraphs = [p.strip() for p in re.split(r"\n{2,}", strText) if p.strip()]
    if listParagraphs:
        return listParagraphs
    return splitSentences(strText)


def windowByChars(
    strText: str,
    intChunkSize: int,
    intOverlap: int,
) -> list[str]:
    strClean = normalizeWhitespace(strText)
    if not strClean:
        return []
    if len(strClean) <= intChunkSize:
        return [strClean]

    listWindows: list[str] = []
    intStart = 0
    intStep = max(1, intChunkSize - intOverlap)
    while intStart < len(strClean):
        strWindow = strClean[intStart : intStart + intChunkSize]
        if strWindow.strip():
            listWindows.append(strWindow.strip())
        intStart += intStep
    return listWindows

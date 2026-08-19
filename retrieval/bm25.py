"""In-memory BM25 sparse retrieval for hybrid search."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class BM25Index:
    def __init__(
        self,
        floatK1: float = 1.5,
        floatB: float = 0.75,
    ) -> None:
        self.floatK1 = floatK1
        self.floatB = floatB
        self.listDocIds: list[str] = []
        self.listDocLengths: list[int] = []
        self.listDocTermFreqs: list[Counter] = []
        self.dictDocFreq: Counter = Counter()
        self.floatAvgDl: float = 0.0
        self.intNumDocs: int = 0

    def _tokenize(self, strText: str) -> list[str]:
        return [token.lower() for token in WORD_PATTERN.findall(strText)]

    def indexDocuments(self, listDocIds: list[str], listTexts: list[str]) -> None:
        self.listDocIds = listDocIds
        self.listTexts = listTexts
        self.dictIdToText = {
            listDocIds[intIdx]: listTexts[intIdx]
            for intIdx in range(len(listDocIds))
        }
        self.intNumDocs = len(listTexts)
        self.listDocTermFreqs = []
        self.listDocLengths = []
        self.dictDocFreq = Counter()

        for strText in listTexts:
            listTokens = self._tokenize(strText)
            counterTf = Counter(listTokens)
            self.listDocTermFreqs.append(counterTf)
            self.listDocLengths.append(len(listTokens))
            self.dictDocFreq.update(counterTf.keys())

        intTotalLength = sum(self.listDocLengths)
        self.floatAvgDl = intTotalLength / max(self.intNumDocs, 1)

    def search(self, strQuery: str, intTopK: int = 20) -> list[tuple[str, float]]:
        listQueryTokens = self._tokenize(strQuery)
        if not listQueryTokens:
            return []

        listScores: list[float] = [0.0] * self.intNumDocs
        for strToken in listQueryTokens:
            intDf = self.dictDocFreq.get(strToken, 0)
            if intDf == 0:
                continue
            floatIdf = math.log(
                (self.intNumDocs - intDf + 0.5) / (intDf + 0.5) + 1.0
            )
            for intDocIdx in range(self.intNumDocs):
                intTf = self.listDocTermFreqs[intDocIdx].get(strToken, 0)
                if intTf == 0:
                    continue
                intDl = self.listDocLengths[intDocIdx]
                floatNumerator = intTf * (self.floatK1 + 1)
                floatDenominator = intTf + self.floatK1 * (
                    1 - self.floatB + self.floatB * intDl / self.floatAvgDl
                )
                listScores[intDocIdx] += floatIdf * floatNumerator / floatDenominator

        listScoredDocs = [
            (self.listDocIds[intIdx], listScores[intIdx])
            for intIdx in range(self.intNumDocs)
            if listScores[intIdx] > 0
        ]
        listScoredDocs.sort(key=lambda x: x[1], reverse=True)
        return listScoredDocs[:intTopK]

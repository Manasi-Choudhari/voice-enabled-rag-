"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) + reciprocal rank fusion."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from retrieval.bm25 import BM25Index
from retrieval.config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_TOP_K,
    SCORE_THRESHOLD,
    TOP_K,
)
from retrieval.embeddings import embedQuery


_cachedClient: QdrantClient | None = None


def _getClient() -> QdrantClient:
    global _cachedClient
    if _cachedClient is None:
        _cachedClient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _cachedClient


def denseSearch(
    strQuery: str,
    intTopK: int = TOP_K,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    if client is None:
        client = _getClient()
    arrayQueryVec = embedQuery(strQuery)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=arrayQueryVec.tolist(),
        limit=intTopK,
        with_payload=True,
    )
    listResults: list[dict[str, Any]] = []
    for point in results.points:
        listResults.append({
            "chunk_id": point.payload.get("chunk_id", ""),
            "doc_id": point.payload.get("doc_id", ""),
            "text": point.payload.get("text", ""),
            "score": point.score,
            "query_type": point.payload.get("query_type", ""),
            "passage_index": point.payload.get("passage_index", 0),
            "source": "dense",
        })
    return listResults


def sparseSearch(
    strQuery: str,
    bm25Index: BM25Index,
    intTopK: int = BM25_TOP_K,
) -> list[dict[str, Any]]:
    listResults = bm25Index.search(strQuery, intTopK=intTopK)
    dictIdToText = getattr(bm25Index, "dictIdToText", {})
    return [
        {
            "chunk_id": strChunkId,
            "text": dictIdToText.get(strChunkId, ""),
            "score": floatScore,
            "source": "bm25",
        }
        for strChunkId, floatScore in listResults
    ]


def reciprocalRankFusion(
    listDenseResults: list[dict[str, Any]],
    listSparseResults: list[dict[str, Any]],
    intK: int = 60,
    floatDenseWeight: float = 0.7,
    floatSparseWeight: float = 0.3,
) -> list[dict[str, Any]]:
    """Combine dense and sparse results via weighted RRF."""
    dictScores: dict[str, float] = {}
    dictPayloads: dict[str, dict[str, Any]] = {}

    for intRank, dictResult in enumerate(listDenseResults):
        strId = dictResult["chunk_id"]
        dictScores[strId] = dictScores.get(strId, 0.0) + floatDenseWeight / (intK + intRank + 1)
        dictPayloads[strId] = dictResult

    for intRank, dictResult in enumerate(listSparseResults):
        strId = dictResult["chunk_id"]
        dictScores[strId] = dictScores.get(strId, 0.0) + floatSparseWeight / (intK + intRank + 1)
        if strId not in dictPayloads:
            dictPayloads[strId] = dictResult

    listFused = sorted(dictScores.items(), key=lambda x: x[1], reverse=True)
    listOutput: list[dict[str, Any]] = []
    for strId, floatRrfScore in listFused:
        dictItem = dict(dictPayloads.get(strId, {}))
        dictItem["rrf_score"] = floatRrfScore
        dictItem["chunk_id"] = strId
        listOutput.append(dictItem)
    return listOutput


def hybrid_search(
    strQuery: str,
    bm25Index: BM25Index,
    intTopK: int = RERANK_TOP_K,
    floatScoreThreshold: float = SCORE_THRESHOLD,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid dense+sparse retrieval with RRF, return top results."""
    listDense = denseSearch(strQuery, intTopK=TOP_K, client=client)
    listSparse = sparseSearch(strQuery, bm25Index, intTopK=BM25_TOP_K)
    listFused = reciprocalRankFusion(listDense, listSparse)

    listFiltered: list[dict[str, Any]] = []
    for dictResult in listFused:
        strText = (dictResult.get("text") or "").strip()
        if len(strText) < 80:
            continue
        listFiltered.append(dictResult)
        if len(listFiltered) >= intTopK:
            break
    return listFiltered

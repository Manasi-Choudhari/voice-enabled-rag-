#!/usr/bin/env python3
"""
Evaluate chunking strategies on MSMARCO-XI using recall@k and MRR.

Builds a global chunk corpus per strategy, embeds with sentence-transformers,
runs dense retrieval for held-out queries, and compares strategies.

Usage:
    python chunking/evaluate.py --data data/processed/msmarco_xi_hin_validation_limit500.parquet
    python chunking/evaluate.py --strategies fixed_size sentence_boundary semantic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking.base import ChunkingStrategy
from chunking.loader import (
    buildCorpusAndQueries,
    findLatestProcessedFile,
    loadProcessedDataset,
    saveEvalSummary,
)
from chunking.registry import getStrategy, listStrategies
from chunking.models import Chunk

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "chunking" / "results"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chunking strategies.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Processed parquet/jsonl path. Defaults to latest hin validation cache.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=f"Strategies to evaluate. Default: all ({', '.join(listStrategies())}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Sentence-transformers model for retrieval eval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for recall@k.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap eval queries for faster runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/CSV results.",
    )
    return parser.parse_args()


def configureStdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def embedTexts(
    model: SentenceTransformer,
    listTexts: list[str],
    strDesc: str,
) -> np.ndarray:
    if not listTexts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(
        listTexts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def retrieveTopK(
    arrayQueryEmb: np.ndarray,
    arrayCorpusEmb: np.ndarray,
    intK: int,
) -> np.ndarray:
    arrayScores = arrayCorpusEmb @ arrayQueryEmb
    intEffectiveK = min(intK, len(arrayScores))
    arrayTopIndices = np.argpartition(-arrayScores, intEffectiveK - 1)[:intEffectiveK]
    arrayTopIndices = arrayTopIndices[np.argsort(-arrayScores[arrayTopIndices])]
    return arrayTopIndices


def computeRecallAtK(
    listRetrievedDocIds: list[str],
    setRelevantDocIds: set[str],
    intK: int,
) -> float:
    if not setRelevantDocIds:
        return 0.0
    setTopK = set(listRetrievedDocIds[:intK])
    return 1.0 if setTopK & setRelevantDocIds else 0.0


def computeMrr(
    listRetrievedDocIds: list[str],
    setRelevantDocIds: set[str],
) -> float:
    for intRank, strDocId in enumerate(listRetrievedDocIds, start=1):
        if strDocId in setRelevantDocIds:
            return 1.0 / intRank
    return 0.0


def evaluateStrategy(
    strategy: ChunkingStrategy,
    listPassages: list,
    listQueries: list[dict[str, Any]],
    model: SentenceTransformer,
    listTopK: list[int],
) -> dict[str, Any]:
    print(f"\n--- Evaluating strategy: {strategy.name} ---")
    listChunks: list[Chunk] = strategy.chunkCorpus(listPassages)
    print(f"Chunks produced: {len(listChunks):,} from {len(listPassages):,} passages")

    listChunkTexts = [chunk.embeddingText() for chunk in listChunks]
    listChunkDocIds = [chunk.docId for chunk in listChunks]

    arrayCorpusEmb = embedTexts(model, listChunkTexts, f"Embed corpus ({strategy.name})")

    dictRecallSums = {intK: 0.0 for intK in listTopK}
    floatMrrSum = 0.0
    intQueryCount = 0

    listQueryTexts = [q["query_text"] for q in listQueries]
    arrayQueryEmb = embedTexts(model, listQueryTexts, f"Embed queries ({strategy.name})")

    intMaxK = max(listTopK)
    for intIndex, dictQuery in enumerate(tqdm(listQueries, desc=f"Retrieve ({strategy.name})")):
        arrayTopChunkIdx = retrieveTopK(arrayQueryEmb[intIndex], arrayCorpusEmb, intMaxK)
        listRetrievedDocIds = [listChunkDocIds[intChunkIdx] for intChunkIdx in arrayTopChunkIdx]

        setRelevant = dictQuery["relevant_doc_ids"]
        for intK in listTopK:
            dictRecallSums[intK] += computeRecallAtK(listRetrievedDocIds, setRelevant, intK)
        floatMrrSum += computeMrr(listRetrievedDocIds, setRelevant)
        intQueryCount += 1

    dictMetrics = {
        "strategy": strategy.name,
        "config": strategy.config(),
        "num_passages": len(listPassages),
        "num_chunks": len(listChunks),
        "num_queries": intQueryCount,
        "avg_chunks_per_passage": round(len(listChunks) / max(len(listPassages), 1), 3),
    }
    for intK in listTopK:
        dictMetrics[f"recall@{intK}"] = round(dictRecallSums[intK] / max(intQueryCount, 1), 4)
    dictMetrics["mrr"] = round(floatMrrSum / max(intQueryCount, 1), 4)
    return dictMetrics


def printComparisonTable(listResults: list[dict[str, Any]], listTopK: list[int]) -> None:
    listColumns = ["strategy", "num_chunks", "avg_chunks_per_passage", "mrr"]
    listColumns.extend([f"recall@{intK}" for intK in listTopK])

    print("\n" + "=" * 90)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 90)
    strHeader = " | ".join(f"{col:>22}" for col in listColumns)
    print(strHeader)
    print("-" * len(strHeader))

    for dictResult in sorted(listResults, key=lambda r: r["mrr"], reverse=True):
        listValues = [f"{dictResult.get(col, ''):>22}" for col in listColumns]
        print(" | ".join(listValues))


def main() -> int:
    configureStdout()
    args = parseArgs()

    pathData = args.data
    if pathData is None:
        pathData = findLatestProcessedFile(PROJECT_ROOT / "data" / "processed")
    print(f"Loading data from {pathData}")
    dfData = loadProcessedDataset(pathData)
    listPassages, listQueries = buildCorpusAndQueries(dfData)

    if args.max_queries is not None:
        listQueries = listQueries[: args.max_queries]

    print(f"Passages: {len(listPassages):,}")
    print(f"Eval queries (with gold labels): {len(listQueries):,}")

    listStrategyNames = args.strategies or listStrategies()
    print(f"Strategies: {', '.join(listStrategyNames)}")

    model = SentenceTransformer(args.model)
    listResults: list[dict[str, Any]] = []

    for strName in listStrategyNames:
        dictOptions: dict[str, Any] = {}
        if strName == "semantic":
            dictOptions["model"] = model
        strategy = getStrategy(strName, dictOptions)
        dictMetrics = evaluateStrategy(
            strategy,
            listPassages,
            listQueries,
            model,
            args.top_k,
        )
        listResults.append(dictMetrics)

    printComparisonTable(listResults, args.top_k)

    dictBest = max(listResults, key=lambda r: r["mrr"])
    dictSummary = {
        "data_path": str(pathData),
        "model": args.model,
        "top_k": args.top_k,
        "num_passages": len(listPassages),
        "num_eval_queries": len(listQueries),
        "results": listResults,
        "recommended_default": dictBest["strategy"],
        "recommended_mrr": dictBest["mrr"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pathJson = args.output_dir / "chunking_eval.json"
    saveEvalSummary(dictSummary, pathJson)

    dfResults = pd.DataFrame(listResults)
    pathCsv = args.output_dir / "chunking_eval.csv"
    dfResults.to_csv(pathCsv, index=False)

    print(f"\nRecommended default strategy: {dictBest['strategy']} (MRR={dictBest['mrr']})")
    print(f"Wrote {pathJson}")
    print(f"Wrote {pathCsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

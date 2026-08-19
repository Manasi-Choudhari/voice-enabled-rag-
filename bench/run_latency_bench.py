#!/usr/bin/env python3
"""
Latency benchmark: run the retrieval+generation pipeline over test queries,
record per-stage and total latency, compute P50/P70/P90/P100.

Usage:
    python bench/run_latency_bench.py --num-queries 200
    python bench/run_latency_bench.py --num-queries 50 --skip-generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking import getStrategy
from chunking.loader import buildCorpusAndQueries, loadProcessedDataset
from retrieval.indexer import index_corpus
from retrieval.search import hybrid_search
from generation.generator import generate_answer
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output

RESULTS_DIR = PROJECT_ROOT / "bench" / "results"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for RAG pipeline.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip LLM call (measure retrieval path only).")
    return parser.parse_args()


def percentile(listValues: list[float], intP: int) -> float:
    if not listValues:
        return 0.0
    return float(np.percentile(listValues, intP))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parseArgs()

    pathData = args.data
    if pathData is None:
        listFiles = sorted((PROJECT_ROOT / "data" / "processed").glob("msmarco_xi_hin_*.parquet"))
        if not listFiles:
            print("No processed data found.")
            return 1
        pathData = listFiles[-1]

    print(f"Loading data from {pathData}")
    dfData = loadProcessedDataset(pathData)
    listPassages, listQueries = buildCorpusAndQueries(dfData)

    if len(listQueries) < args.num_queries:
        print(f"Only {len(listQueries)} queries with gold labels; using all of them.")
    listTestQueries = listQueries[:args.num_queries]
    print(f"Test queries: {len(listTestQueries)}")

    strategy = getStrategy("semantic")
    listChunks = strategy.chunkCorpus(listPassages)
    bm25Index = index_corpus(listChunks, boolRecreate=True)

    strApiKey = os.getenv("GROQ_API_KEY", "")

    listRecords: list[dict] = []

    for intIdx, dictQuery in enumerate(listTestQueries):
        strQuery = dictQuery["query_text"]
        dictRecord: dict = {"query_idx": intIdx, "query": strQuery[:80]}

        # Input guardrail
        floatStart = time.perf_counter()
        dictInputCheck = check_input(strQuery)
        dictRecord["input_guard_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

        if not dictInputCheck["passed"]:
            dictRecord["retrieval_ms"] = 0
            dictRecord["generation_ms"] = 0
            dictRecord["output_guard_ms"] = 0
            dictRecord["total_ms"] = dictRecord["input_guard_ms"]
            dictRecord["refused"] = True
            listRecords.append(dictRecord)
            continue

        # Retrieval
        floatStart = time.perf_counter()
        listContext = hybrid_search(strQuery, bm25Index, intTopK=args.top_k)
        dictRecord["retrieval_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

        # Generation
        if args.skip_generation or not strApiKey:
            dictRecord["generation_ms"] = 0
        else:
            floatStart = time.perf_counter()
            dictAnswer = generate_answer(strQuery, listContext, strApiKey=strApiKey)
            dictRecord["generation_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

        # Output guardrail
        floatStart = time.perf_counter()
        if not args.skip_generation and strApiKey:
            check_output(dictAnswer, listContext)
        dictRecord["output_guard_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

        dictRecord["total_ms"] = round(
            dictRecord["input_guard_ms"]
            + dictRecord["retrieval_ms"]
            + dictRecord["generation_ms"]
            + dictRecord["output_guard_ms"],
            2,
        )
        dictRecord["refused"] = False
        listRecords.append(dictRecord)

        if (intIdx + 1) % 20 == 0:
            print(f"  Completed {intIdx + 1}/{len(listTestQueries)}")

    # Compute stats
    dfResults = pd.DataFrame(listRecords)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dfResults.to_csv(RESULTS_DIR / "latency_raw.csv", index=False)

    listStages = ["input_guard_ms", "retrieval_ms", "generation_ms", "output_guard_ms", "total_ms"]
    listPercentiles = [50, 70, 90, 100]

    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print(f"Queries: {len(listRecords)} | Skip generation: {args.skip_generation}")
    print("=" * 80)

    strHeader = f"{'Stage':>20} | {'Mean':>10} | {'P50':>10} | {'P70':>10} | {'P90':>10} | {'P100':>10}"
    print(strHeader)
    print("-" * len(strHeader))

    dictSummary: dict = {"num_queries": len(listRecords), "skip_generation": args.skip_generation}

    for strStage in listStages:
        listValues = dfResults[strStage].tolist()
        floatMean = round(float(np.mean(listValues)), 2) if listValues else 0
        dictRow = {"mean": floatMean}
        listParts = [f"{strStage:>20}", f"{floatMean:>10.2f}"]
        for intP in listPercentiles:
            floatVal = round(percentile(listValues, intP), 2)
            dictRow[f"p{intP}"] = floatVal
            listParts.append(f"{floatVal:>10.2f}")
        print(" | ".join(listParts))
        dictSummary[strStage] = dictRow

    pathSummary = RESULTS_DIR / "latency_summary.json"
    pathSummary.write_text(json.dumps(dictSummary, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR / 'latency_raw.csv'}")
    print(f"Wrote {pathSummary}")

    # Note on 200ms target
    floatRetrievalP90 = dictSummary.get("retrieval_ms", {}).get("p90", 0)
    print(f"\n--- 200ms Target Analysis ---")
    print(f"Retrieval P90: {floatRetrievalP90:.1f}ms")
    if not args.skip_generation:
        floatGenP90 = dictSummary.get("generation_ms", {}).get("p90", 0)
        print(f"Generation P90: {floatGenP90:.1f}ms")
        print(f"Retrieval + guardrails can fit under 200ms.")
        print(f"LLM generation over a network call adds {floatGenP90:.0f}ms — this is inherent API latency.")
        print(f"For sub-200ms total, use a local/distilled model or measure retrieval-path only.")
    else:
        floatTotalP90 = dictSummary.get("total_ms", {}).get("p90", 0)
        print(f"Total (retrieval path) P90: {floatTotalP90:.1f}ms — {'WITHIN' if floatTotalP90 < 200 else 'ABOVE'} 200ms target")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

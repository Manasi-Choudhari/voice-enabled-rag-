#!/usr/bin/env python3
"""Quick integration test: index 500-row sample and run a few queries."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking import getStrategy
from chunking.loader import buildCorpusAndQueries, loadProcessedDataset
from retrieval.indexer import index_corpus
from retrieval.search import hybrid_search


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pathData = PROJECT_ROOT / "data" / "processed" / "msmarco_xi_hin_validation_limit500.parquet"
    print(f"Loading data from {pathData}")
    dfData = loadProcessedDataset(pathData)
    listPassages, listQueries = buildCorpusAndQueries(dfData)
    print(f"Passages: {len(listPassages):,} | Eval queries: {len(listQueries):,}")

    strategy = getStrategy("semantic")
    listChunks = strategy.chunkCorpus(listPassages)
    print(f"Chunks: {len(listChunks):,}")

    bm25Index = index_corpus(listChunks, boolRecreate=True)

    print("\n--- Sample hybrid queries ---")
    for dictQuery in listQueries[:5]:
        strQuery = dictQuery["query_text"]
        listResults = hybrid_search(strQuery, bm25Index, intTopK=3)
        print(f"\nQ: {strQuery[:100]}")
        for intRank, dictResult in enumerate(listResults, 1):
            strText = dictResult.get("text", "")[:120]
            print(f"  [{intRank}] rrf={dictResult.get('rrf_score', 0):.4f} | {strText}")

    print("\nRetrieval test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

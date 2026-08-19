#!/usr/bin/env python3
"""
Prepare ai4bharat/MSMARCO-XI for the voice-enabled RAG pipeline.

Downloads the dataset, inspects schema, cleans records, and caches processed
Parquet + JSONL locally so re-runs are fast.

The HF repo stores one Parquet file per language under train/ and validation/.
For dev iteration we stream rows (no full 3.5 GB download) when --limit is set.

Usage (fast dev iteration):
    python data/prepare_dataset.py --config hi --split validation --limit 20000

Full language split (downloads entire Parquet for that language):
    python data/prepare_dataset.py --config hi --split train

Inspect schema only (streams a handful of rows):
    python data/prepare_dataset.py --config hi --split validation --inspect-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

DATASET_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
WHITESPACE_PATTERN = re.compile(r"\s+")

# HF repo uses 3-letter language prefixes in Parquet filenames.
LANGUAGE_FILES: dict[str, dict[str, str]] = {
    "asm": {"train": "train/asmtrain.parquet", "validation": "validation/asmval.parquet"},
    "ben": {"train": "train/bentrain.parquet", "validation": "validation/benval.parquet"},
    "guj": {"train": "train/gujtrain.parquet", "validation": "validation/gujval.parquet"},
    "hin": {"train": "train/hintrain.parquet", "validation": "validation/hinval.parquet"},
    "kan": {"train": "train/kantrain.parquet", "validation": "validation/kanval.parquet"},
    "mal": {"train": "train/maltrain.parquet", "validation": "validation/malval.parquet"},
    "mar": {"train": "train/martrain.parquet", "validation": "validation/marval.parquet"},
    "nep": {"train": "train/neptrain.parquet", "validation": "validation/nepval.parquet"},
    "ori": {"train": "train/oritrain.parquet", "validation": "validation/orival.parquet"},
    "pan": {"train": "train/pantrain.parquet", "validation": "validation/panval.parquet"},
    "san": {"train": "train/santrain.parquet", "validation": "validation/sanval.parquet"},
    "tam": {"train": "train/tamtrain.parquet", "validation": "validation/tamval.parquet"},
    "tel": {"train": None, "validation": "validation/telval.parquet"},
    "urd": {"train": "train/urdtrain.parquet", "validation": "validation/urdval.parquet"},
}

# Common 2-letter aliases -> 3-letter repo keys
LANGUAGE_ALIASES: dict[str, str] = {
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
    "pa": "pan",
    "or": "ori",
    "as": "asm",
    "ne": "nep",
    "ur": "urd",
    "sa": "san",
}


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, inspect, clean, and cache MSMARCO-XI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hi",
        help="Language code (hi/hin, bn/ben, ta/tam, ...). Use --list-configs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split. Prefer validation for faster downloads during dev.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process. When set, rows are streamed (fast, no full download).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cached processed files.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl", "both"],
        default="both",
        help="Output format for processed cache.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print available language configs and exit.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Load and print schema + samples without writing cache.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to print during inspection.",
    )
    return parser.parse_args()


def resolveLanguageKey(strConfig: str) -> str:
    strKey = strConfig.strip().lower()
    if strKey in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[strKey]
    if strKey in LANGUAGE_FILES:
        return strKey
    listAvailable = sorted(set(LANGUAGE_FILES.keys()) | set(LANGUAGE_ALIASES.keys()))
    raise ValueError(
        f"Unknown config {strConfig!r}. Available: {', '.join(listAvailable)}"
    )


def getParquetPath(strLanguageKey: str, strSplit: str) -> str:
    dictFiles = LANGUAGE_FILES[strLanguageKey]
    strPath = dictFiles.get(strSplit)
    if not strPath:
        raise ValueError(
            f"No {strSplit} split for language {strLanguageKey!r}. "
            f"Available splits: {[k for k, v in dictFiles.items() if v]}"
        )
    return strPath


def configureStdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalizeText(strValue: str | None) -> str | None:
    if strValue is None:
        return None
    strNormalized = unicodedata.normalize("NFKC", str(strValue))
    strNormalized = WHITESPACE_PATTERN.sub(" ", strNormalized).strip()
    return strNormalized if strNormalized else None


def flattenRow(dictRow: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested passages into tabular columns for easier downstream use."""
    dictPassages = dictRow.get("passages") or {}
    listIsSelected = dictPassages.get("is_selected") or []
    listEnglishPassages = dictPassages.get("English_passages") or []
    listTranslatedPassages = dictPassages.get("Translated_passages") or []
    dictMeta = dictRow.get("meta") or {}

    return {
        "query_id": dictRow.get("query_id"),
        "query_type": normalizeText(dictRow.get("query_type")),
        "source_lang": normalizeText(dictRow.get("source_lang")),
        "target_lang": normalizeText(dictRow.get("target_lang")),
        "query": normalizeText(dictRow.get("query")),
        "answer": normalizeText(dictRow.get("Answer")),
        "eng_query": normalizeText(dictRow.get("Eng_Query")),
        "eng_answer": normalizeText(dictRow.get("Eng_Answer")),
        "passage_is_selected": listIsSelected,
        "passage_english": [normalizeText(p) or "" for p in listEnglishPassages],
        "passage_translated": [normalizeText(p) or "" for p in listTranslatedPassages],
        "meta_model_name": dictMeta.get("model_name"),
        "meta_temperature": dictMeta.get("temperature"),
        "meta_max_tokens": dictMeta.get("max_tokens"),
        "meta_top_p": dictMeta.get("top_p"),
    }


def isValidRow(dictFlat: dict[str, Any]) -> bool:
    if dictFlat.get("query_id") is None:
        return False
    if not dictFlat.get("query"):
        return False
    listPassages = dictFlat.get("passage_translated") or dictFlat.get("passage_english") or []
    if not listPassages or not any(p.strip() for p in listPassages):
        return False
    return True


def iterRows(
    strLanguageKey: str,
    strSplit: str,
    intLimit: int | None,
) -> tuple[list[dict[str, Any]], Any | None]:
    strParquetPath = getParquetPath(strLanguageKey, strSplit)
    print(f"\nLoading {DATASET_ID}")
    print(f"  language : {strLanguageKey}")
    print(f"  split    : {strSplit}")
    print(f"  file     : {strParquetPath}")
    print(f"  limit    : {intLimit if intLimit is not None else 'none (full file)'}")

    if intLimit is not None:
        datasetStream = load_dataset(
            DATASET_ID,
            data_files={strSplit: strParquetPath},
            split=strSplit,
            streaming=True,
        )
        listRows: list[dict[str, Any]] = []
        for dictRow in datasetStream:
            listRows.append(dictRow)
            if len(listRows) >= intLimit:
                break
        print(f"Streamed {len(listRows):,} rows.")
        return listRows, None

    datasetSplit = load_dataset(
        DATASET_ID,
        data_files={strSplit: strParquetPath},
        split=strSplit,
    )
    print(f"Loaded {len(datasetSplit):,} rows.")
    return [datasetSplit[intIndex] for intIndex in range(len(datasetSplit))], datasetSplit


def printInspection(listRows: list[dict[str, Any]], intSampleRows: int, features: Any | None) -> None:
    print("\n" + "=" * 72)
    print("DATASET INSPECTION")
    print("=" * 72)
    print(f"Dataset ID   : {DATASET_ID}")
    print(f"Num rows     : {len(listRows):,}")
    if listRows:
        print(f"Column names : {sorted(listRows[0].keys())}")
    if features is not None:
        print(f"Features     : {features}")
    print("\n--- Sample rows (raw) ---")
    for intIndex, dictRow in enumerate(listRows[:intSampleRows]):
        print(f"\n[Row {intIndex}]")
        strJson = json.dumps(dictRow, indent=2, ensure_ascii=False, default=str)
        print(strJson[:5000])
        if len(strJson) > 5000:
            print("... (truncated)")


def cleanRows(listRows: list[dict[str, Any]]) -> pd.DataFrame:
    listFlatRows: list[dict[str, Any]] = []
    for dictRow in tqdm(listRows, desc="Cleaning rows"):
        dictFlat = flattenRow(dictRow)
        if isValidRow(dictFlat):
            listFlatRows.append(dictFlat)

    dfProcessed = pd.DataFrame(listFlatRows)
    intBeforeDedup = len(dfProcessed)
    dfProcessed = dfProcessed.drop_duplicates(subset=["query_id"], keep="first")
    intAfterDedup = len(dfProcessed)

    print("\n--- Cleaning summary ---")
    print(f"Raw rows loaded     : {len(listRows):,}")
    print(f"Rows after filtering: {intBeforeDedup:,}")
    print(f"Rows after dedup    : {intAfterDedup:,} (removed {intBeforeDedup - intAfterDedup:,} duplicate query_ids)")
    if not dfProcessed.empty:
        print(f"Columns             : {list(dfProcessed.columns)}")
        print(f"Query types         : {dfProcessed['query_type'].value_counts().head(10).to_dict()}")
        intSelectedCounts = dfProcessed["passage_is_selected"].apply(
            lambda vals: sum(1 for v in vals if v == 1) if isinstance(vals, list) else 0
        )
        print(f"Avg selected passages per query: {intSelectedCounts.mean():.2f}")
    return dfProcessed


def saveProcessed(
    dfProcessed: pd.DataFrame,
    pathOutputDir: Path,
    strLanguageKey: str,
    strSplit: str,
    intLimit: int | None,
    strFormat: str,
) -> None:
    pathOutputDir.mkdir(parents=True, exist_ok=True)
    strLimitTag = f"limit{intLimit}" if intLimit is not None else "full"
    strBaseName = f"msmarco_xi_{strLanguageKey}_{strSplit}_{strLimitTag}"

    dictMeta = {
        "dataset_id": DATASET_ID,
        "language": strLanguageKey,
        "split": strSplit,
        "limit": intLimit,
        "num_rows": len(dfProcessed),
        "columns": list(dfProcessed.columns),
    }

    pathMeta = pathOutputDir / f"{strBaseName}_meta.json"
    pathMeta.write_text(json.dumps(dictMeta, indent=2), encoding="utf-8")
    print(f"Wrote metadata -> {pathMeta}")

    if strFormat in ("parquet", "both"):
        pathParquet = pathOutputDir / f"{strBaseName}.parquet"
        dfProcessed.to_parquet(pathParquet, index=False)
        print(f"Wrote parquet  -> {pathParquet}")

    if strFormat in ("jsonl", "both"):
        pathJsonl = pathOutputDir / f"{strBaseName}.jsonl"
        dfProcessed.to_json(pathJsonl, orient="records", lines=True, force_ascii=False)
        print(f"Wrote jsonl    -> {pathJsonl}")


def listConfigs() -> None:
    print("Available language configs:")
    print("\nKey   Aliases   Train   Validation")
    print("-" * 50)
    for strKey in sorted(LANGUAGE_FILES.keys()):
        listAliases = [alias for alias, key in LANGUAGE_ALIASES.items() if key == strKey]
        dictFiles = LANGUAGE_FILES[strKey]
        print(
            f"{strKey:<5} {','.join(listAliases) or '-':<8} "
            f"{'yes' if dictFiles.get('train') else 'no':<7} "
            f"{'yes' if dictFiles.get('validation') else 'no'}"
        )


def main() -> int:
    configureStdout()
    args = parseArgs()

    if args.list_configs:
        listConfigs()
        return 0

    try:
        strLanguageKey = resolveLanguageKey(args.config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        listRows, datasetSplit = iterRows(strLanguageKey, args.split, args.limit)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    features = datasetSplit.features if datasetSplit is not None else None
    printInspection(listRows, args.sample_rows, features)

    if args.inspect_only:
        print("\n--inspect-only set; skipping clean/cache.")
        return 0

    dfProcessed = cleanRows(listRows)

    print("\n--- Processed sample rows ---")
    for _, seriesRow in dfProcessed.head(args.sample_rows).iterrows():
        dictSample = seriesRow.to_dict()
        for strKey in ("passage_english", "passage_translated", "passage_is_selected"):
            if strKey in dictSample and isinstance(dictSample[strKey], list):
                dictSample[strKey] = dictSample[strKey][:2]
        print(json.dumps(dictSample, indent=2, ensure_ascii=False, default=str))

    saveProcessed(
        dfProcessed,
        args.output_dir,
        strLanguageKey,
        args.split,
        args.limit,
        args.format,
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

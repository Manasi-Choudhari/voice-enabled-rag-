"""Load processed MSMARCO-XI cache into chunking-friendly structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from chunking.models import PassageRecord


def loadProcessedDataset(pathData: Path) -> pd.DataFrame:
    if pathData.suffix == ".parquet":
        return pd.read_parquet(pathData)
    if pathData.suffix == ".jsonl":
        return pd.read_json(pathData, lines=True)
    raise ValueError(f"Unsupported file type: {pathData}")


def findLatestProcessedFile(pathProcessedDir: Path, strLanguage: str = "hin") -> Path:
    listCandidates = sorted(pathProcessedDir.glob(f"msmarco_xi_{strLanguage}_*.parquet"))
    if not listCandidates:
        raise FileNotFoundError(f"No processed parquet found in {pathProcessedDir}")
    return listCandidates[-1]


def _toList(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def rowToPassageRecords(dictRow: dict[str, Any], boolUseTranslated: bool = True) -> list[PassageRecord]:
    strPassageKey = "passage_translated" if boolUseTranslated else "passage_english"
    strEngKey = "passage_english"
    listPassages = _toList(dictRow.get(strPassageKey))
    listEnglish = _toList(dictRow.get(strEngKey))
    listSelected = _toList(dictRow.get("passage_is_selected"))

    listRecords: list[PassageRecord] = []
    for intIndex, strText in enumerate(listPassages):
        if not strText or not str(strText).strip():
            continue
        boolSelected = False
        if intIndex < len(listSelected):
            boolSelected = int(listSelected[intIndex]) == 1
        strEng = listEnglish[intIndex] if intIndex < len(listEnglish) else None
        listRecords.append(
            PassageRecord(
                queryId=int(dictRow["query_id"]),
                passageIndex=intIndex,
                text=str(strText).strip(),
                queryType=dictRow.get("query_type"),
                isSelected=boolSelected,
                engText=strEng,
            )
        )
    return listRecords


def buildCorpusAndQueries(
    dfData: pd.DataFrame,
    boolUseTranslated: bool = True,
) -> tuple[list[PassageRecord], list[dict[str, Any]]]:
    listAllPassages: list[PassageRecord] = []
    listQueries: list[dict[str, Any]] = []

    for _, seriesRow in dfData.iterrows():
        dictRow = seriesRow.to_dict()
        listPassages = rowToPassageRecords(dictRow, boolUseTranslated=boolUseTranslated)
        listAllPassages.extend(listPassages)

        setRelevantDocIds = {
            passage.docId for passage in listPassages if passage.isSelected
        }
        if not setRelevantDocIds:
            continue

        listQueries.append(
            {
                "query_id": int(dictRow["query_id"]),
                "query_text": dictRow.get("query") or dictRow.get("eng_query") or "",
                "query_type": dictRow.get("query_type"),
                "relevant_doc_ids": setRelevantDocIds,
            }
        )

    return listAllPassages, listQueries


def saveEvalSummary(dictSummary: dict[str, Any], pathOutput: Path) -> None:
    pathOutput.parent.mkdir(parents=True, exist_ok=True)
    pathOutput.write_text(json.dumps(dictSummary, indent=2), encoding="utf-8")

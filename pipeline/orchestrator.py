"""Pipeline orchestrator: STT → guardrails → retrieval → generation → output guard."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pipeline.models import PipelineResponse

logger = logging.getLogger("pipeline")


def _configureLogging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def run_text_pipeline(
    strQuery: str,
    bm25Index: Any,
    strApiKey: str | None = None,
    intTopK: int = 5,
) -> PipelineResponse:
    """Run the RAG pipeline from a text query (post-STT)."""
    import sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from generation.generator import generate_answer
    from guardrails.input_guard import check_input
    from guardrails.output_guard import check_output
    from retrieval.search import hybrid_search

    _configureLogging()
    floatTotalStart = time.perf_counter()
    dictTimings: dict[str, float] = {}

    # Stage 1: Input guardrails
    floatStart = time.perf_counter()
    dictInputCheck = check_input(strQuery)
    dictTimings["input_guardrail_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

    if not dictInputCheck["passed"]:
        logger.info(f"Input refused: {dictInputCheck['category']} - {strQuery[:80]}")
        return PipelineResponse(
            query=strQuery,
            answer=dictInputCheck["reason"],
            refused=True,
            refusalReason=dictInputCheck["reason"],
            timings=dictTimings,
            totalLatencyMs=round((time.perf_counter() - floatTotalStart) * 1000, 2),
        )

    # Stage 2: Retrieval
    floatStart = time.perf_counter()
    listContext = hybrid_search(strQuery, bm25Index, intTopK=intTopK)
    dictTimings["retrieval_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)
    logger.info(f"Retrieved {len(listContext)} chunks in {dictTimings['retrieval_ms']:.1f}ms")

    # Stage 3: Generation
    floatStart = time.perf_counter()
    dictAnswer = generate_answer(strQuery, listContext, strApiKey=strApiKey)
    dictTimings["generation_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)
    logger.info(f"Generated answer in {dictTimings['generation_ms']:.1f}ms")

    # Stage 4: Output guardrails
    floatStart = time.perf_counter()
    dictOutputCheck = check_output(dictAnswer, listContext)
    dictTimings["output_guardrail_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)

    if not dictOutputCheck["passed"]:
        logger.info(f"Output refused: {dictOutputCheck['category']}")
        return PipelineResponse(
            query=strQuery,
            answer=dictOutputCheck["reason"],
            refused=True,
            refusalReason=dictOutputCheck["reason"],
            timings=dictTimings,
            totalLatencyMs=round((time.perf_counter() - floatTotalStart) * 1000, 2),
            retrievedChunks=listContext,
        )

    floatTotalMs = round((time.perf_counter() - floatTotalStart) * 1000, 2)
    dictTimings["total_ms"] = floatTotalMs

    logger.info(json.dumps({
        "event": "pipeline_complete",
        "query": strQuery[:100],
        "total_ms": floatTotalMs,
        "timings": dictTimings,
        "num_chunks": len(listContext),
        "confidence": dictAnswer.get("confidence", 0),
        "grounded": dictAnswer.get("grounded", False),
    }))

    return PipelineResponse(
        query=strQuery,
        answer=dictAnswer.get("answer", ""),
        sources=dictAnswer.get("sources", []),
        confidence=dictAnswer.get("confidence", 0.0),
        grounded=dictAnswer.get("grounded", False),
        timings=dictTimings,
        totalLatencyMs=floatTotalMs,
        retrievedChunks=listContext,
    )


def run_pipeline(
    audioPath: Path | None = None,
    audioBytes: bytes | None = None,
    strQuery: str | None = None,
    bm25Index: Any = None,
    strApiKey: str | None = None,
    strLanguage: str = "unknown",
    intTopK: int = 5,
    strAudioFormat: str = "wav",
) -> PipelineResponse:
    """Full pipeline: audio/text → STT → RAG pipeline."""
    import sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from stt import getSTTProvider

    _configureLogging()
    floatTotalStart = time.perf_counter()
    dictTimings: dict[str, float] = {}

    # STT stage (skip if text query provided directly)
    if strQuery is None:
        floatStart = time.perf_counter()
        sttProvider = getSTTProvider()

        if audioPath is not None:
            dictSTT = sttProvider.transcribe(audioPath, language=strLanguage)
        elif audioBytes is not None:
            dictSTT = sttProvider.transcribeBytes(
                audioBytes,
                language=strLanguage,
                strFormat=strAudioFormat,
            )
        else:
            return PipelineResponse(
                answer="No audio input or text query provided.",
                refused=True,
                refusalReason="no_input",
                timings=dictTimings,
                totalLatencyMs=round((time.perf_counter() - floatTotalStart) * 1000, 2),
            )

        dictTimings["stt_ms"] = round((time.perf_counter() - floatStart) * 1000, 2)
        strQuery = dictSTT.get("text", "")
        strSttError = dictSTT.get("error")
        logger.info(
            f"STT: '{strQuery[:80]}' in {dictTimings['stt_ms']:.1f}ms error={strSttError}"
        )

        if not strQuery.strip():
            strReason = strSttError or "empty_transcript"
            return PipelineResponse(
                answer=(
                    "I couldn't understand the audio. Please try again. "
                    f"({strReason})"
                ),
                refused=True,
                refusalReason=strReason,
                timings=dictTimings,
                totalLatencyMs=round((time.perf_counter() - floatTotalStart) * 1000, 2),
            )

    # Run text pipeline
    response = run_text_pipeline(strQuery, bm25Index, strApiKey=strApiKey, intTopK=intTopK)

    # Merge STT timing if present
    if "stt_ms" in dictTimings:
        response.timings["stt_ms"] = dictTimings["stt_ms"]
        response.totalLatencyMs = round((time.perf_counter() - floatTotalStart) * 1000, 2)
        response.timings["total_ms"] = response.totalLatencyMs

    return response

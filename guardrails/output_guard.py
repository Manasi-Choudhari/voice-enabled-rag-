"""Output guardrails: groundedness check + low-confidence refusal."""

from __future__ import annotations

from typing import Any

LOW_CONFIDENCE_THRESHOLD = 0.3
MIN_RETRIEVAL_SCORE = 0.01


def _checkGroundedness(dictAnswer: dict[str, Any]) -> dict[str, Any]:
    """Verify the answer claims grounding in sources."""
    if not dictAnswer.get("grounded", False):
        return {
            "passed": False,
            "reason": "The generated answer is not sufficiently grounded in the retrieved context.",
            "category": "ungrounded",
        }

    listSources = dictAnswer.get("sources", [])
    if not listSources:
        return {
            "passed": False,
            "reason": "The answer does not cite any sources from the retrieved context.",
            "category": "no_citations",
        }

    return {"passed": True, "reason": None, "category": None}


def _checkConfidence(dictAnswer: dict[str, Any]) -> dict[str, Any]:
    """Refuse if confidence is too low."""
    floatConfidence = dictAnswer.get("confidence", 0.0)
    if floatConfidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "passed": False,
            "reason": "I don't have enough information to answer that confidently.",
            "category": "low_confidence",
        }
    return {"passed": True, "reason": None, "category": None}


def _checkRetrievalQuality(listContext: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse if retrieval scores are too low."""
    if not listContext:
        return {
            "passed": False,
            "reason": "No relevant passages were found for your question.",
            "category": "no_retrieval",
        }

    floatTopScore = max(
        ctx.get("rrf_score", 0.0) or ctx.get("score", 0.0)
        for ctx in listContext
    )
    if floatTopScore < MIN_RETRIEVAL_SCORE:
        return {
            "passed": False,
            "reason": "The retrieved passages are not relevant enough to answer your question reliably.",
            "category": "low_retrieval_score",
        }

    return {"passed": True, "reason": None, "category": None}


def check_output(
    dictAnswer: dict[str, Any],
    listContext: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all output guardrails.

    Returns:
        dict with keys: passed (bool), reason (str|None), category (str|None)
    """
    dictRetrievalCheck = _checkRetrievalQuality(listContext)
    if not dictRetrievalCheck["passed"]:
        return dictRetrievalCheck

    dictGroundCheck = _checkGroundedness(dictAnswer)
    if not dictGroundCheck["passed"]:
        return dictGroundCheck

    dictConfidenceCheck = _checkConfidence(dictAnswer)
    if not dictConfidenceCheck["passed"]:
        return dictConfidenceCheck

    return {"passed": True, "reason": None, "category": None}

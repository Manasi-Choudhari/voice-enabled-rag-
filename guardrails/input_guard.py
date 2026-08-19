"""Input guardrails: off-topic detection + unsafe content filtering."""

from __future__ import annotations

import re
from typing import Any

UNSAFE_PATTERNS = [
    re.compile(r"\b(kill|murder|bomb|attack|shoot|stab|suicide|self.harm)\b", re.IGNORECASE),
    re.compile(r"\b(hack|exploit|crack\s+password|sql\s*inject)\b", re.IGNORECASE),
    re.compile(r"\b(porn|xxx|nude|naked|sex\s+with)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+make\s+(a\s+)?bomb|how\s+to\s+kill)\b", re.IGNORECASE),
]

OFFTOPIC_SIGNALS = [
    re.compile(r"\b(write\s+(me\s+)?(a\s+)?(poem|song|story|essay|code))\b", re.IGNORECASE),
    re.compile(r"\b(tell\s+me\s+a\s+joke|sing|play\s+a\s+game)\b", re.IGNORECASE),
    re.compile(r"\b(what\s+is\s+your\s+name|who\s+made\s+you|are\s+you\s+ai)\b", re.IGNORECASE),
    re.compile(r"\b(translate\s+this|summarize\s+this\s+article)\b", re.IGNORECASE),
]


def _isUnsafe(strQuery: str) -> bool:
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(strQuery):
            return True
    return False


def _isOffTopic(strQuery: str) -> bool:
    for pattern in OFFTOPIC_SIGNALS:
        if pattern.search(strQuery):
            return True
    if len(strQuery.strip()) < 3:
        return True
    return False


def check_input(strQuery: str) -> dict[str, Any]:
    """Check query for safety and relevance.

    Returns:
        dict with keys: passed (bool), reason (str|None), category (str|None)
    """
    if _isUnsafe(strQuery):
        return {
            "passed": False,
            "reason": "Query contains unsafe or harmful content. I cannot process this request.",
            "category": "unsafe",
        }

    if _isOffTopic(strQuery):
        return {
            "passed": False,
            "reason": "This question is outside the scope of my knowledge base. I can only answer factual questions from the dataset.",
            "category": "off_topic",
        }

    return {"passed": True, "reason": None, "category": None}

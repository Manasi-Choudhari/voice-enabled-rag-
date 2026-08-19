"""LLM-based answer generation with structured JSON output via Groq."""

from __future__ import annotations

import json
import time
from typing import Any

from groq import Groq

from generation.config import (
    GROQ_API_KEY,
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
)

SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer ONLY from the provided context passages. Do not use any prior knowledge.

Rules:
1. If the context does not contain enough information to answer, set "grounded" to false and say so.
2. Always cite which source(s) support your answer using their IDs.
3. Respond ONLY with valid JSON matching this exact schema:

{
  "answer": "string - the answer text",
  "sources": ["list of source chunk_ids that support the answer"],
  "confidence": 0.0 to 1.0,
  "grounded": true or false
}

Do not include any text outside the JSON object."""


def _buildUserPrompt(strQuery: str, listContext: list[dict[str, Any]]) -> str:
    listParts = ["Context passages:\n"]
    for intIdx, dictChunk in enumerate(listContext):
        strChunkId = dictChunk.get("chunk_id", f"source_{intIdx}")
        strText = dictChunk.get("text", "")
        listParts.append(f"[{strChunkId}]: {strText}\n")
    listParts.append(f"\nQuestion: {strQuery}\n\nAnswer as JSON:")
    return "\n".join(listParts)


def _parseResponse(strResponse: str) -> dict[str, Any]:
    strClean = strResponse.strip()
    if strClean.startswith("```"):
        listLines = strClean.split("\n")
        listLines = [l for l in listLines if not l.strip().startswith("```")]
        strClean = "\n".join(listLines)

    dictParsed = json.loads(strClean)

    dictValidated = {
        "answer": str(dictParsed.get("answer", "")),
        "sources": list(dictParsed.get("sources", [])),
        "confidence": float(dictParsed.get("confidence", 0.0)),
        "grounded": bool(dictParsed.get("grounded", False)),
    }
    return dictValidated


def generate_answer(
    strQuery: str,
    listContext: list[dict[str, Any]],
    strApiKey: str | None = None,
) -> dict[str, Any]:
    """Generate a grounded answer from retrieved context chunks.

    Returns dict with keys: answer, sources, confidence, grounded, latency_ms, raw_response.
    """
    strKey = strApiKey or GROQ_API_KEY
    if not strKey:
        return {
            "answer": "LLM API key not configured.",
            "sources": [],
            "confidence": 0.0,
            "grounded": False,
            "latency_ms": 0,
            "error": "missing_api_key",
        }

    client = Groq(api_key=strKey)
    strUserPrompt = _buildUserPrompt(strQuery, listContext)
    strRawResponse = ""

    for intAttempt in range(MAX_RETRIES + 1):
        try:
            floatStart = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": strUserPrompt},
                ],
            )
            floatLatencyMs = (time.perf_counter() - floatStart) * 1000

            strRawResponse = response.choices[0].message.content or ""
            dictResult = _parseResponse(strRawResponse)
            dictResult["latency_ms"] = round(floatLatencyMs, 1)
            dictResult["raw_response"] = strRawResponse
            return dictResult

        except json.JSONDecodeError:
            if intAttempt < MAX_RETRIES:
                continue
            return {
                "answer": "Failed to parse LLM response as valid JSON.",
                "sources": [],
                "confidence": 0.0,
                "grounded": False,
                "latency_ms": 0,
                "error": "json_parse_failure",
                "raw_response": strRawResponse,
            }
        except Exception as exc:
            if intAttempt < MAX_RETRIES:
                time.sleep(1.0 * (intAttempt + 1))
                continue
            return {
                "answer": f"LLM call failed: {exc}",
                "sources": [],
                "confidence": 0.0,
                "grounded": False,
                "latency_ms": 0,
                "error": str(exc),
            }

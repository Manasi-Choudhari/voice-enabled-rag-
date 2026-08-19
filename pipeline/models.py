"""Pydantic models for pipeline stage inputs/outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class STTResult(BaseModel):
    text: str = ""
    language: str = "hi-IN"
    confidence: float = 0.0
    latencyMs: float = 0.0
    error: str | None = None


class GuardrailResult(BaseModel):
    passed: bool = True
    reason: str | None = None
    category: str | None = None


class RetrievalResult(BaseModel):
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    latencyMs: float = 0.0


class GenerationResult(BaseModel):
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    latencyMs: float = 0.0
    error: str | None = None


class PipelineResponse(BaseModel):
    query: str = ""
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    refused: bool = False
    refusalReason: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    totalLatencyMs: float = 0.0
    retrievedChunks: list[dict[str, Any]] = Field(default_factory=list)

#!/usr/bin/env python3
"""FastAPI backend for voice-enabled RAG pipeline."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking import getStrategy
from chunking.loader import buildCorpusAndQueries, loadProcessedDataset
from pipeline.orchestrator import run_pipeline, run_text_pipeline
from retrieval.bm25 import BM25Index
from retrieval.indexer import index_corpus

app = FastAPI(title="Voice-Enabled RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_bm25Index: BM25Index | None = None
_boolReady = False


class TextQueryRequest(BaseModel):
    query: str
    topK: int = 5


@app.on_event("startup")
async def startup() -> None:
    global _bm25Index, _boolReady

    pathData = PROJECT_ROOT / "data" / "processed"
    listFiles = sorted(pathData.glob("msmarco_xi_hin_*.parquet"))
    if not listFiles:
        print("WARNING: No processed data found. Run data/prepare_dataset.py first.")
        return

    pathLatest = listFiles[-1]
    print(f"Loading data from {pathLatest}")
    dfData = loadProcessedDataset(pathLatest)
    listPassages, _ = buildCorpusAndQueries(dfData, boolUseTranslated=False)
    print(f"Passages: {len(listPassages):,}")

    strategy = getStrategy("sentence_boundary", {"intMaxChars": 1500})
    listChunks = strategy.chunkCorpus(listPassages)
    print(f"Chunks: {len(listChunks):,}")

    _bm25Index = index_corpus(listChunks, boolRecreate=True)
    _boolReady = True
    print("Pipeline ready.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ready" if _boolReady else "loading", "ready": _boolReady}


@app.post("/api/query")
async def textQuery(request: TextQueryRequest) -> dict:
    if not _boolReady or _bm25Index is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    response = run_text_pipeline(
        request.query,
        _bm25Index,
        strApiKey=os.getenv("GROQ_API_KEY"),
        intTopK=request.topK,
    )
    return response.model_dump()


@app.post("/api/voice")
async def voiceQuery(
    audio: UploadFile = File(...),
    language: str = Form("unknown"),
    topK: int = Form(5),
) -> dict:
    if not _boolReady or _bm25Index is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    audioBytes = await audio.read()
    if not audioBytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    strFormat = "wav"
    if audio.filename:
        strExt = Path(audio.filename).suffix.lstrip(".")
        if strExt in ("webm", "mp3", "wav", "ogg", "m4a"):
            strFormat = strExt

    response = run_pipeline(
        audioBytes=audioBytes,
        bm25Index=_bm25Index,
        strApiKey=os.getenv("GROQ_API_KEY"),
        strLanguage=language,
        intTopK=topK,
        strAudioFormat=strFormat,
    )
    return response.model_dump()


# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serveIndex() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

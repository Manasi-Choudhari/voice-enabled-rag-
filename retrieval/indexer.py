"""Index corpus chunks into Qdrant + build BM25 index."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from retrieval.bm25 import BM25Index
from retrieval.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_HOST,
    QDRANT_PORT,
)
from retrieval.embeddings import embedTexts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking import Chunk


def getQdrantClient() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensureCollection(client: QdrantClient) -> None:
    listCollections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in listCollections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")


def index_corpus(
    listChunks: list[Chunk],
    boolRecreate: bool = False,
) -> BM25Index:
    """Index chunks into Qdrant and build in-memory BM25 index. Returns BM25Index."""
    client = getQdrantClient()

    if boolRecreate:
        listCollections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in listCollections:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")

    ensureCollection(client)

    listChunks = [
        chunk for chunk in listChunks if len((chunk.text or "").strip()) >= 80
    ]
    listTexts = [chunk.embeddingText() for chunk in listChunks]
    listDocIds = [chunk.chunkId for chunk in listChunks]

    print(f"Embedding {len(listTexts):,} chunks...")
    intBatchSize = 256
    for intStart in tqdm(range(0, len(listTexts), intBatchSize), desc="Indexing"):
        intEnd = min(intStart + intBatchSize, len(listTexts))
        listBatchTexts = listTexts[intStart:intEnd]
        listBatchChunks = listChunks[intStart:intEnd]
        arrayEmbeddings = embedTexts(listBatchTexts)

        listPoints: list[PointStruct] = []
        for intIdx, chunk in enumerate(listBatchChunks):
            listPoints.append(
                PointStruct(
                    id=intStart + intIdx,
                    vector=arrayEmbeddings[intIdx].tolist(),
                    payload={
                        "chunk_id": chunk.chunkId,
                        "doc_id": chunk.docId,
                        "text": chunk.text,
                        "query_id": chunk.queryId,
                        "query_type": chunk.queryType or "",
                        "passage_index": chunk.passageIndex,
                        "chunk_index": chunk.chunkIndex,
                        "parent_chunk_id": chunk.parentChunkId or "",
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=listPoints)

    print(f"Indexed {len(listChunks):,} chunks in Qdrant.")

    print("Building BM25 index...")
    bm25Index = BM25Index()
    bm25Index.indexDocuments(listDocIds, listTexts)
    print(f"BM25 index built: {bm25Index.intNumDocs:,} docs.")

    return bm25Index

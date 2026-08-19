"""Retrieval configuration."""

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "rag_passages"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 10
BM25_TOP_K = 20
RERANK_TOP_K = 5
SCORE_THRESHOLD = 0.25

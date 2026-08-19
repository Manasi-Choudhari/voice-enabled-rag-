"""Default chunking configuration for the RAG pipeline."""

DEFAULT_STRATEGY = "semantic"

STRATEGY_CONFIGS = {
    "fixed_size": {"intChunkSize": 512, "intOverlap": 64},
    "sentence_boundary": {"intMaxChars": 512, "boolPreferParagraphs": True},
    "semantic": {
        "strModelName": "sentence-transformers/all-MiniLM-L6-v2",
        "floatSimilarityThreshold": 0.55,
        "intMaxSentencesPerChunk": 8,
    },
    "metadata_aware": {"intMaxChars": 512},
    "hierarchical": {"intChildSize": 256, "intChildOverlap": 32},
}

# factory.py  (Step 6)
# One function that builds a retriever by name: "bm25", "dense" or "hybrid".

from src.config import load_config
from src.retrieval.bm25_retriever import BM25Search
from src.retrieval.dense_retriever import DenseSearch
from src.retrieval.hybrid_retriever import HybridSearch
from src.storage import sqlite_store, vector_store


def build_retriever(name, chunk_size, conn, embeddings):
    """Create a retriever for one chunk size."""
    if name == "bm25":
        return BM25Search(sqlite_store.get_chunks(conn, chunk_size))

    if name == "dense":
        return DenseSearch(vector_store.get_vector_store(chunk_size, embeddings))

    if name == "hybrid":
        bm25 = BM25Search(sqlite_store.get_chunks(conn, chunk_size))
        dense = DenseSearch(vector_store.get_vector_store(chunk_size, embeddings))
        rrf_k = load_config()["retrieval"]["rrf_k"]
        return HybridSearch(bm25, dense, rrf_k=rrf_k)

    raise ValueError("Unknown retriever: " + name + "  (use bm25, dense or hybrid)")
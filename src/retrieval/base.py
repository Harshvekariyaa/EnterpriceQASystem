# base.py  (Step 6)
# Shared helpers. EVERY retriever returns results in the SAME format,
# so the rest of the project can swap retrievers freely.
#
# A retriever has one method:   retrieve(question, k)  ->  list of result dictionaries

import re


def tokenize(text):
    """Split text into lowercase words and numbers. 'Rs. 5,000?' -> ['rs', '5', '000']"""
    return re.findall(r"[a-z0-9]+", text.lower())


def make_result(rank, chunk_id, text, metadata, score, retriever_name):
    """Build one result dictionary."""
    return {
        "rank": rank,                          # 1 = best
        "chunk_id": chunk_id,
        "text": text,
        "score": score,                        # meaning depends on the retriever
        "file_name": metadata.get("file_name"),
        "section": metadata.get("section"),
        "page": metadata.get("page"),
        "doc_id": metadata.get("doc_id"),
        "chunk_size": metadata.get("chunk_size"),
        "retriever": retriever_name,
    }
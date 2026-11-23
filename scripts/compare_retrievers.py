# compare_retrievers.py  (Step 6)
# Run:   python scripts/compare_retrievers.py
# Or:    python scripts/compare_retrievers.py "your own question here"
# Shows the top 3 chunks from BM25, dense and hybrid for the same question.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.retrieval.factory import build_retriever
from src.storage import sqlite_store, vector_store

config = load_config()
chunk_size = config["ingestion"]["default_chunk_size"]       # 80

if len(sys.argv) > 1:
    questions = [sys.argv[1]]
else:
    questions = [
        "How many sick leave days do I get?",             # meaning question
        "Can I use SMS codes for login?",                 # exact keyword: SMS
        "Who approves claims above 25000 rupees?",        # number written differently
    ]

conn = sqlite_store.connect(str(resolve_path(config["paths"]["sqlite_path"])))
embeddings = vector_store.get_embeddings()

retrievers = {}
for name in ["bm25", "dense", "hybrid"]:
    retrievers[name] = build_retriever(name, chunk_size, conn, embeddings)

for question in questions:
    print("\n" + "=" * 90)
    print("QUESTION:", question, "  (chunk size", chunk_size, ")")
    print("=" * 90)
    for name, retriever in retrievers.items():
        print("\n  [" + name.upper() + "]")
        for r in retriever.retrieve(question, k=3):
            snippet = r["text"].replace("\n", " ")[:70]
            print("   ", r["rank"], "| score:", r["score"], "|", r["file_name"],
                  "|", r["section"], "|", snippet)
conn.close()
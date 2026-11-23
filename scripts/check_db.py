# check_db.py  (Step 5)
# Run:   python scripts/check_db.py
# Looks inside the databases and tries 2 searches, to prove everything is stored.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.storage import sqlite_store, vector_store

config = load_config()
conn = sqlite_store.connect(str(resolve_path(config["paths"]["sqlite_path"])))

print("---- SQLite tables ----")
for table in ["documents", "chunks", "cache", "eval_runs"]:
    print(table, ":", sqlite_store.count_rows(conn, table), "rows")

print()
print("---- Chunks per size ----")
for row in conn.execute("SELECT chunk_size, COUNT(*) AS n FROM chunks GROUP BY chunk_size"):
    print("chunk size", row["chunk_size"], ":", row["n"], "chunks")

print()
print("---- Test searches in ChromaDB ----")
embeddings = vector_store.get_embeddings()
questions = ["How many sick leave days do I get?", "What is the password length rule?"]
for size in config["ingestion"]["chunk_sizes_words"]:
    store = vector_store.get_vector_store(size, embeddings)
    for question in questions:
        # Chroma returns a DISTANCE (small = close). similarity = 1 - distance
        results = store.similarity_search_with_score(question, k=1)
        document, distance = results[0]
        print("[size", size, "]", question)
        print("   similarity:", round(1 - distance, 3), "|", document.metadata["file_name"],
              "|", document.metadata["section"], "| page", document.metadata["page"])
        print("   text:", document.page_content[:90], "...")
conn.close()
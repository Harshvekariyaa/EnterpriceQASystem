# run_ingest.py  (Step 5)
# Run from the project folder:   python scripts/run_ingest.py
# Run it TWICE. The second time it must add nothing.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.ingestion.pipeline import load_json_list, sync_chunks, sync_documents
from src.logger import get_logger
from src.storage import sqlite_store, vector_store

log = get_logger("run_ingest")

config = load_config()
pdf_folder = str(resolve_path(config["paths"]["pdf_dir"]))
raw_folder = str(resolve_path(config["paths"]["raw_dir"]))
processed_folder = str(resolve_path(config["paths"]["processed_dir"]))
db_path = str(resolve_path(config["paths"]["sqlite_path"]))
kb_version = config["project"]["kb_version"]
chunk_sizes = config["ingestion"]["chunk_sizes_words"]

# Optional:  python scripts/run_ingest.py --reset   deletes both databases first.
# Needed when chunk TEXT is unchanged but its labels (section / page) changed.
if "--reset" in sys.argv:
    import shutil
    if os.path.exists(db_path):
        os.remove(db_path)
    chroma_folder = str(resolve_path(config["paths"]["chroma_dir"]))
    if os.path.isdir(chroma_folder):
        shutil.rmtree(chroma_folder)
    print("Both databases were deleted. Building them again from scratch...\n")
# 1. Open the database and create tables

conn = sqlite_store.connect(db_path)
sqlite_store.create_tables(conn)

# 2. Documents (from Step 3)
documents = load_json_list(raw_folder, prefix="")
doc_report = sync_documents(conn, documents, pdf_folder, kb_version)
print("DOCUMENTS ->", doc_report)

# 3. Chunks (from Step 4), one chunk size at a time
embeddings = vector_store.get_embeddings()
for size in chunk_sizes:
    chunks = load_json_list(processed_folder, prefix="chunks_" + str(size))
    chunks = chunks[0] if len(chunks) > 0 else []        # the file holds a list of chunks
    store = vector_store.get_vector_store(size, embeddings)
    report = sync_chunks(conn, size, chunks, store)
    print("CHUNKS size", size, "->", report)

# 4. Final totals
print()
print("Rows in SQLite  ->",
      "documents:", sqlite_store.count_rows(conn, "documents"),
      "| chunks:", sqlite_store.count_rows(conn, "chunks"))
log.info("Ingestion finished. kb_version=%s", kb_version)
conn.close()
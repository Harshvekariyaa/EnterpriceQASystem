# pipeline.py  (Step 5 - part 3)
# Job: put documents and chunks into SQLite + ChromaDB WITHOUT making duplicates.
#
# The trick (called "sync"):
#   - Every chunk has an id (its hash fingerprint).
#   - We compare the ids we WANT with the ids already STORED.
#       want but not stored  -> add it
#       stored but not wanted -> remove it (old data)
#       in both              -> do nothing
#   So running the pipeline again changes nothing.

import hashlib
import json
import os

from src.storage import sqlite_store, vector_store


def file_hash(path):
    """Fingerprint of a PDF file. If the PDF changes, the hash changes."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()


def load_json_list(folder, prefix):
    """Load every .json file in a folder whose name starts with `prefix`."""
    items = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json") and name.startswith(prefix):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                items.append(json.load(f))
    return items


def sync_documents(conn, documents, pdf_folder, kb_version):
    """Save document details. Returns how many are new / updated / unchanged."""
    report = {"new": 0, "updated": 0, "unchanged": 0}
    for doc in documents:
        new_hash = file_hash(os.path.join(pdf_folder, doc["file_name"]))
        old_hash = sqlite_store.get_saved_hash(conn, doc["file_name"])

        if old_hash is None:
            report["new"] += 1
        elif old_hash != new_hash:
            report["updated"] += 1
        else:
            report["unchanged"] += 1

        sqlite_store.save_document(conn, doc, new_hash, kb_version)   # insert or update
    return report


def sync_chunks(conn, chunk_size, chunks, store):
    """Make SQLite and ChromaDB contain exactly the given chunks."""
    wanted_ids = set()
    for c in chunks:
        wanted_ids.add(c["chunk_id"])

    # ---- SQLite ----
    stored_ids = sqlite_store.get_chunk_ids(conn, chunk_size)
    sql_new = [c for c in chunks if c["chunk_id"] not in stored_ids]
    sql_old = stored_ids - wanted_ids
    sqlite_store.insert_chunks(conn, sql_new)
    sqlite_store.delete_chunks(conn, chunk_size, sql_old)

    # ---- ChromaDB (embeddings are made ONLY for the new chunks) ----
    chroma_ids = vector_store.get_all_ids(store)
    chroma_new = [c for c in chunks if c["chunk_id"] not in chroma_ids]
    chroma_old = chroma_ids - wanted_ids
    vector_store.add_chunks(store, chroma_new)
    vector_store.remove_ids(store, chroma_old)

    return {
        "total_wanted": len(chunks),
        "sqlite_added": len(sql_new), "sqlite_removed": len(sql_old),
        "chroma_added": len(chroma_new), "chroma_removed": len(chroma_old),
    }
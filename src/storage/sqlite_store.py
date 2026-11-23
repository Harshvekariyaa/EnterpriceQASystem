# sqlite_store.py  (Step 5 - part 1)
# Job: create the SQLite database and read/write rows.
# SQLite = a whole database in ONE file (data/db/rag.sqlite). No installation needed.

import os
import sqlite3
from datetime import datetime


def connect(db_path):
    """Open (or create) the database file."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row            # lets us use row["name"]
    conn.execute("PRAGMA foreign_keys = ON")  # make SQLite check links between tables
    return conn


def create_tables(conn):
    """Create the 4 tables (only if they do not exist yet)."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        file_name      TEXT PRIMARY KEY,
        doc_id         TEXT,
        title          TEXT,
        department     TEXT,
        version        TEXT,
        effective_date TEXT,
        num_pages      INTEGER,
        file_hash      TEXT,
        kb_version     TEXT,
        ingested_at    TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    TEXT,
        chunk_size  INTEGER,
        doc_id      TEXT,
        file_name   TEXT,
        title       TEXT,
        section     TEXT,
        page        INTEGER,
        chunk_index INTEGER,
        word_count  INTEGER,
        text        TEXT,
        PRIMARY KEY (chunk_id, chunk_size),
        FOREIGN KEY (file_name) REFERENCES documents (file_name)
    );

    CREATE TABLE IF NOT EXISTS cache (
        query_hash      TEXT PRIMARY KEY,
        query_text      TEXT,
        query_embedding TEXT,
        answer          TEXT,
        citations       TEXT,
        kb_version      TEXT,
        hit_count       INTEGER DEFAULT 0,
        created_at      TEXT
    );

    CREATE TABLE IF NOT EXISTS eval_runs (
        run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_name TEXT,
        config_json     TEXT,
        metrics_json    TEXT,
        created_at      TEXT
    );
    """)
    conn.commit()


# ---------------- documents ----------------

def get_saved_hash(conn, file_name):
    """Return the saved file hash of a document, or None if it is not stored yet."""
    row = conn.execute(
        "SELECT file_hash FROM documents WHERE file_name = ?", (file_name,)
    ).fetchone()
    if row is None:
        return None
    return row["file_hash"]


def save_document(conn, doc, file_hash, kb_version):
    """Insert a document. If it already exists, update it (no duplicate row)."""
    conn.execute("""
        INSERT INTO documents (file_name, doc_id, title, department, version,
                               effective_date, num_pages, file_hash, kb_version, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_name) DO UPDATE SET
            doc_id = excluded.doc_id, title = excluded.title,
            department = excluded.department, version = excluded.version,
            effective_date = excluded.effective_date, num_pages = excluded.num_pages,
            file_hash = excluded.file_hash, kb_version = excluded.kb_version,
            ingested_at = excluded.ingested_at
    """, (doc["file_name"], doc["doc_id"], doc["title"], doc["department"],
          doc["version"], doc["effective_date"], doc["num_pages"], file_hash,
          kb_version, datetime.now().isoformat(timespec="seconds")))
    conn.commit()


# ---------------- chunks ----------------

def get_chunk_ids(conn, chunk_size):
    """All chunk ids stored for one chunk size (returned as a set)."""
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE chunk_size = ?", (chunk_size,)
    ).fetchall()
    ids = set()
    for row in rows:
        ids.add(row["chunk_id"])
    return ids


def insert_chunks(conn, chunks):
    """Insert chunks. 'INSERT OR IGNORE' = if the chunk already exists, do nothing."""
    rows = []
    for c in chunks:
        rows.append((c["chunk_id"], c["chunk_size"], c["doc_id"], c["file_name"],
                     c["title"], c["section"], c["page"], c["chunk_index"],
                     c["word_count"], c["text"]))
    conn.executemany("""
        INSERT OR IGNORE INTO chunks
        (chunk_id, chunk_size, doc_id, file_name, title, section, page,
         chunk_index, word_count, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def delete_chunks(conn, chunk_size, chunk_ids):
    """Delete chunks that no longer exist in the knowledge base."""
    rows = []
    for chunk_id in chunk_ids:
        rows.append((chunk_id, chunk_size))
    conn.executemany("DELETE FROM chunks WHERE chunk_id = ? AND chunk_size = ?", rows)
    conn.commit()


def count_rows(conn, table):
    """Number of rows in a table (table names come from our own code, not from users)."""
    return conn.execute("SELECT COUNT(*) AS n FROM " + table).fetchone()["n"]


def get_chunks(conn, chunk_size):
    """Return all chunks of one size as a list of dictionaries (used to build BM25)."""
    rows = conn.execute(
        "SELECT * FROM chunks WHERE chunk_size = ? ORDER BY file_name, chunk_index",
        (chunk_size,),
    ).fetchall()
    chunks = []
    for row in rows:
        chunks.append(dict(row))
    return chunks

import json


def save_eval_run(conn, experiment_name, config, metrics):
    """Save one experiment result in the eval_runs table.
    If the same experiment name already exists, the old row is replaced (latest result only)."""
    conn.execute("DELETE FROM eval_runs WHERE experiment_name = ?", (experiment_name,))
    conn.execute(
        "INSERT INTO eval_runs (experiment_name, config_json, metrics_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (experiment_name, json.dumps(config), json.dumps(metrics),
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
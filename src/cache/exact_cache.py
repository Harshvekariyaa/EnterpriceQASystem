# exact_cache.py  (Step 11 - part 1)
# Job: remember answers by the EXACT question (after tidying it) and give them back instantly.
# Both caches share the SQLite table "cache", which was created in Step 5.

import hashlib
import json
import re
from datetime import datetime

from src.storage import sqlite_store


def normalize_question(question):
    """Make two spellings of the same question look identical.
    'How many sick days?'  and  'how many   SICK days'  ->  'how many sick days'
    Numbers are protected: '5,000' stays '5000' and '3.5' stays '3 5' (never '35'),
    so questions about different amounts can never collide."""
    text = str(question).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)      # 5,000 -> 5000
    text = re.sub(r"[^\w\s]", " ", text)            # punctuation -> space
    return " ".join(text.split())                   # collapse spaces


def make_key(question, kb_version):
    """The cache key. kb_version is INSIDE the key, so changing the PDFs and bumping
    kb_version in config.yaml makes every old entry unfindable."""
    raw = kb_version + "|" + normalize_question(question)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExactCache:
    def __init__(self, db_path, kb_version):
        self.db_path = db_path
        self.kb_version = kb_version

    def get(self, question):
        """Return {'answer', 'citations'} or None."""
        key = make_key(question, self.kb_version)
        conn = sqlite_store.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT answer, citations FROM cache WHERE query_hash = ? AND kb_version = ?",
                (key, self.kb_version),
            ).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?", (key,))
            conn.commit()
            return {"answer": row["answer"], "citations": json.loads(row["citations"])}
        finally:
            conn.close()

    def put(self, question, answer, citations, embedding=None):
        """Save an answer. The embedding is stored too, so the semantic cache can use it."""
        key = make_key(question, self.kb_version)
        conn = sqlite_store.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO cache (query_hash, query_text, query_embedding, answer, citations,
                                   kb_version, hit_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    answer = excluded.answer, citations = excluded.citations,
                    query_embedding = excluded.query_embedding, created_at = excluded.created_at
            """, (key, question, json.dumps(embedding) if embedding is not None else None,
                  answer, json.dumps(citations), self.kb_version,
                  datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        finally:
            conn.close()

    def clear_old_versions(self):
        """Delete entries made for an older knowledge base. Returns how many were removed."""
        conn = sqlite_store.connect(self.db_path)
        try:
            cursor = conn.execute("DELETE FROM cache WHERE kb_version != ?", (self.kb_version,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
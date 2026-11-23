# semantic_cache.py  (Step 11 - part 2)
# Job: reuse an answer when a NEW question MEANS the same as a cached one.

import json
import re

import numpy as np

from src.storage import sqlite_store

# Groups of words that CHANGE the answer. Inside one group the words are alternatives:
# "probation" and "confirmation" are different situations, "sick" and "maternity" are different leaves.
# Rule: if BOTH questions mention a word from the same group and the words differ -> no cache hit.
# (If only one question mentions the group we allow it, because a paraphrase often drops a word.)
# Extend these groups when you meet a wrong hit.
TERM_GROUPS = {
    "leave_type": {"sick": "sick", "maternity": "maternity", "paternity": "paternity",
                   "annual": "annual", "casual": "casual", "earned": "earned", "marriage": "marriage"},
    "stage":      {"probation": "probation", "confirmation": "confirmed", "confirmed": "confirmed"},
    "direction":  {"after": "after", "before": "before", "longer": "longer", "shorter": "shorter",
                   "more": "more", "less": "less", "above": "above", "below": "below",
                   "over": "over", "under": "under", "minimum": "min", "maximum": "max"},
    "scope":      {"domestic": "domestic", "international": "international"},
    "role":       {"manager": "manager", "managers": "manager", "senior": "senior", "junior": "junior",
                   "director": "director", "intern": "intern", "interns": "intern"},
    "family":     {"parent": "parent", "parents": "parent", "spouse": "spouse",
                   "child": "child", "children": "child"},
    "work_mode":  {"remote": "remote", "hybrid": "hybrid", "office": "office", "onsite": "onsite"},
    "expense":    {"laptop": "laptop", "mobile": "mobile", "internet": "internet", "meal": "meal",
                   "meals": "meal", "hotel": "hotel", "flight": "flight", "taxi": "taxi", "car": "car"},
    "period":     {"monthly": "monthly", "daily": "daily", "yearly": "yearly", "weekly": "weekly"},
}


def extract_numbers(text):
    """All numbers in a question as a set of strings: 'Rs. 5,000 in 14 days' -> {'5000', '14'}."""
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text))
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def extract_terms(text):
    """{'leave_type': {'sick'}, 'scope': {'domestic'}} - the key words found, grouped."""
    words = re.findall(r"[a-z]+", str(text).lower())
    found = {}
    for group, mapping in TERM_GROUPS.items():
        hits = {mapping[w] for w in words if w in mapping}
        if hits:
            found[group] = hits
    return found


def facts_match(question_a, question_b):
    """Safety 2. True only if the numbers are identical and no key-word group conflicts."""
    if extract_numbers(question_a) != extract_numbers(question_b):
        return False
    terms_a = extract_terms(question_a)
    terms_b = extract_terms(question_b)
    for group in terms_a:
        if group in terms_b and terms_a[group] != terms_b[group]:
            return False
    return True


def cosine(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class SemanticCache:
    def __init__(self, db_path, kb_version, embed_fn, threshold):
        self.db_path = db_path
        self.kb_version = kb_version
        self.embed_fn = embed_fn          # a function: text -> list of numbers
        self.threshold = threshold

    def get(self, question, question_embedding=None):
        """Return {'answer', 'citations', 'similarity', 'matched_question'} or None."""
        if question_embedding is None:
            question_embedding = self.embed_fn(question)

        conn = sqlite_store.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT query_hash, query_text, query_embedding, answer, citations "
                "FROM cache WHERE kb_version = ? AND query_embedding IS NOT NULL",
                (self.kb_version,),
            ).fetchall()

            best = None
            best_similarity = -1.0
            for row in rows:
                similarity = cosine(question_embedding, json.loads(row["query_embedding"]))
                if similarity < self.threshold:
                    continue                                    # safety 1: high threshold
                if not facts_match(question, row["query_text"]):
                    continue                                    # safety 2: numbers / key terms
                if similarity > best_similarity:
                    best, best_similarity = row, similarity

            if best is None:
                return None
            conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                         (best["query_hash"],))
            conn.commit()
            return {"answer": best["answer"], "citations": json.loads(best["citations"]),
                    "similarity": round(best_similarity, 4), "matched_question": best["query_text"]}
        finally:
            conn.close()
# dedup.py  (Step 4 - part 3)
# Job: remove chunks that have exactly the same text.
# Idea: make a "fingerprint" (hash) of each text. Same text = same fingerprint.

import hashlib


def make_chunk_id(text):
    """Fingerprint of a chunk. We lower-case the text and fix spaces first,
    so 'Hello  World' and 'hello world' count as the same."""
    cleaned = " ".join(text.lower().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


def remove_duplicates(chunks):
    """Keep the first copy of every text. Returns (unique_chunks, number_removed)."""
    seen_ids = set()          # a set remembers ids we already saw
    unique_chunks = []
    removed = 0

    for chunk in chunks:
        chunk_id = make_chunk_id(chunk["text"])
        if chunk_id in seen_ids:
            removed = removed + 1          # duplicate -> skip it
        else:
            seen_ids.add(chunk_id)
            chunk["chunk_id"] = chunk_id
            unique_chunks.append(chunk)
    return unique_chunks, removed
# test_ingestion.py  (Step 13)
# Chunk sizes, overlap, duplicate removal, and the key P0 rule:
# RUNNING INGESTION TWICE MUST ADD NOTHING.
import pytest

from src.ingestion import pipeline as ingest
from src.ingestion.dedup import make_chunk_id, remove_duplicates
from src.storage import sqlite_store


# ---------------- chunking ----------------

def make_blocks(word_count):
    text = " ".join("w" + str(i) for i in range(1, word_count + 1))
    return [{"text": text, "markers": [(0, "1. Test Section", 1)]}]


DOCUMENT = {"doc_id": "HR-TEST-01", "file_name": "test.pdf", "title": "Test"}


def test_chunks_are_never_longer_than_the_chunk_size():
    pytest.importorskip("langchain_text_splitters")
    from src.ingestion.chunker import make_chunks
    chunks = make_chunks(DOCUMENT, make_blocks(300), chunk_size=80, overlap=15)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["word_count"] <= 80
        assert chunk["chunk_size"] == 80


def test_neighbouring_chunks_overlap():
    pytest.importorskip("langchain_text_splitters")
    from src.ingestion.chunker import make_chunks
    chunks = make_chunks(DOCUMENT, make_blocks(300), chunk_size=80, overlap=15)
    for first, second in zip(chunks, chunks[1:]):
        shared = set(first["text"].split()) & set(second["text"].split())
        assert 1 <= len(shared) <= 15


def test_bigger_chunk_size_gives_fewer_chunks():
    pytest.importorskip("langchain_text_splitters")
    from src.ingestion.chunker import make_chunks
    small = make_chunks(DOCUMENT, make_blocks(300), chunk_size=80, overlap=15)
    big = make_chunks(DOCUMENT, make_blocks(300), chunk_size=160, overlap=15)
    assert len(big) < len(small)


def test_chunks_keep_their_section_page_and_order():
    pytest.importorskip("langchain_text_splitters")
    from src.ingestion.chunker import make_chunks
    chunks = make_chunks(DOCUMENT, make_blocks(200), chunk_size=80, overlap=15)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk["section"] == "1. Test Section"
        assert chunk["page"] == 1
        assert chunk["file_name"] == "test.pdf"


# ---------------- de-duplication ----------------

def test_identical_text_gets_the_same_id_even_with_other_case_and_spacing():
    assert make_chunk_id("Confidential   Notice") == make_chunk_id("confidential notice")
    assert len(make_chunk_id("anything")) == 16


def test_different_text_gets_a_different_id():
    assert make_chunk_id("12 days of sick leave") != make_chunk_id("26 weeks of maternity leave")


def test_duplicates_are_removed_and_counted():
    chunks = [
        {"text": "This document is confidential."},
        {"text": "Employees get 12 days of sick leave."},
        {"text": "THIS  document is   CONFIDENTIAL."},       # same text, other spelling
    ]
    unique, removed = remove_duplicates(chunks)
    assert removed == 1
    assert len(unique) == 2
    assert unique[0]["text"] == "This document is confidential."     # the FIRST copy is kept
    assert "chunk_id" in unique[0]


# ---------------- idempotent ingestion ----------------

class FakeVectorStore:
    """Pretends to be ChromaDB. Only remembers which chunk ids it holds."""
    def __init__(self):
        self.ids = set()


def use_fake_vector_functions(monkeypatch):
    monkeypatch.setattr(ingest.vector_store, "get_all_ids", lambda store: set(store.ids))
    monkeypatch.setattr(ingest.vector_store, "add_chunks",
                        lambda store, chunks: store.ids.update(c["chunk_id"] for c in chunks))
    monkeypatch.setattr(ingest.vector_store, "remove_ids",
                        lambda store, ids: store.ids.difference_update(ids))


def make_db(tmp_path):
    conn = sqlite_store.connect(str(tmp_path / "test.sqlite"))
    sqlite_store.create_tables(conn)
    return conn


def make_doc(file_name="a.pdf"):
    return {"file_name": file_name, "doc_id": "HR-A-01", "title": "A", "department": "HR",
            "version": "1.0", "effective_date": "2025-01-01", "num_pages": 1}


def make_stored_chunk(chunk_id, text):
    return {"chunk_id": chunk_id, "chunk_size": 80, "doc_id": "HR-A-01", "file_name": "a.pdf",
            "title": "A", "section": "1. Intro", "page": 1, "chunk_index": 0,
            "word_count": len(text.split()), "text": text}


def test_running_ingestion_twice_adds_nothing(tmp_path, monkeypatch):
    use_fake_vector_functions(monkeypatch)
    conn = make_db(tmp_path)
    store = FakeVectorStore()
    (tmp_path / "a.pdf").write_bytes(b"pretend this is a pdf")
    chunks = [make_stored_chunk("id1", "first chunk text"), make_stored_chunk("id2", "second chunk text")]

    # first run
    docs_1 = ingest.sync_documents(conn, [make_doc()], str(tmp_path), "v1")
    chunks_1 = ingest.sync_chunks(conn, 80, chunks, store)
    assert docs_1 == {"new": 1, "updated": 0, "unchanged": 0}
    assert chunks_1["sqlite_added"] == 2 and chunks_1["chroma_added"] == 2

    # second run: exactly the same input
    docs_2 = ingest.sync_documents(conn, [make_doc()], str(tmp_path), "v1")
    chunks_2 = ingest.sync_chunks(conn, 80, chunks, store)
    assert docs_2 == {"new": 0, "updated": 0, "unchanged": 1}
    assert chunks_2["sqlite_added"] == 0 and chunks_2["chroma_added"] == 0
    assert chunks_2["sqlite_removed"] == 0 and chunks_2["chroma_removed"] == 0

    assert sqlite_store.count_rows(conn, "documents") == 1
    assert sqlite_store.count_rows(conn, "chunks") == 2
    assert len(store.ids) == 2
    conn.close()


def test_a_changed_pdf_is_detected_and_old_chunks_are_removed(tmp_path, monkeypatch):
    use_fake_vector_functions(monkeypatch)
    conn = make_db(tmp_path)
    store = FakeVectorStore()
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"version one")
    ingest.sync_documents(conn, [make_doc()], str(tmp_path), "v1")
    ingest.sync_chunks(conn, 80, [make_stored_chunk("id1", "old text")], store)

    pdf.write_bytes(b"version two")                                  # the file changes
    docs = ingest.sync_documents(conn, [make_doc()], str(tmp_path), "v1")
    result = ingest.sync_chunks(conn, 80, [make_stored_chunk("id9", "new text")], store)

    assert docs["updated"] == 1
    assert result["sqlite_added"] == 1 and result["sqlite_removed"] == 1
    assert result["chroma_added"] == 1 and result["chroma_removed"] == 1
    assert store.ids == {"id9"}
    conn.close()


def test_file_hash_changes_when_the_file_changes(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"one")
    first = ingest.file_hash(str(pdf))
    pdf.write_bytes(b"two")
    assert ingest.file_hash(str(pdf)) != first
    assert len(first) == 64
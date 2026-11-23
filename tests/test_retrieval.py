# test_retrieval.py  (Step 13)
# BM25, dense and hybrid on a tiny sample corpus.
import pytest

from helpers import make_chunk
from src.retrieval.base import make_result, tokenize
from src.retrieval.dense_retriever import DenseSearch
from src.retrieval.hybrid_retriever import HybridSearch


def test_tokenize_lowercases_and_splits_numbers():
    assert tokenize("Rs. 5,000?") == ["rs", "5", "000"]


# ---------------- BM25 (real, on a 5-chunk corpus) ----------------

SAMPLE_TEXTS = [
    "Maternity leave is 26 weeks of paid leave for eligible employees.",
    "Employees receive 12 days of paid sick leave per year with a medical certificate.",
    "Laptop screens lock automatically after 5 minutes of inactivity.",
    "Meal claims are limited to Rs. 1,000 per day on business duty.",
    "Hotel stays during international travel are capped per night.",
]


def sample_corpus():
    corpus = []
    for number, text in enumerate(SAMPLE_TEXTS, start=1):
        chunk = make_chunk(number, text=text)
        chunk["chunk_id"] = "c" + str(number)
        corpus.append(chunk)
    return corpus


def test_bm25_puts_the_chunk_with_the_rare_words_first():
    pytest.importorskip("langchain_community")
    pytest.importorskip("rank_bm25")
    from src.retrieval.bm25_retriever import BM25Search
    search = BM25Search(sample_corpus())
    assert search.retrieve("How many weeks of maternity leave?", k=3)[0]["chunk_id"] == "c1"
    assert search.retrieve("When do laptop screens lock?", k=3)[0]["chunk_id"] == "c3"


def test_bm25_respects_k_and_returns_the_common_result_format():
    pytest.importorskip("langchain_community")
    pytest.importorskip("rank_bm25")
    from src.retrieval.bm25_retriever import BM25Search
    results = BM25Search(sample_corpus()).retrieve("paid leave", k=2)
    assert len(results) == 2
    assert [r["rank"] for r in results] == [1, 2]
    for key in ["chunk_id", "text", "score", "file_name", "section", "page", "doc_id", "retriever"]:
        assert key in results[0]
    assert results[0]["retriever"] == "bm25"


# ---------------- dense (fake vector store) ----------------

class FakeDoc:
    def __init__(self, text, chunk_id):
        self.page_content = text
        self.metadata = {"chunk_id": chunk_id, "file_name": "01_Leave_Policy.pdf",
                         "section": "3. Sick Leave", "page": 1, "doc_id": "HR-LEAVE-01", "chunk_size": 80}


class FakeVectorStore:
    """Chroma returns (document, DISTANCE) pairs. Small distance = close."""
    def similarity_search_with_score(self, question, k=5):
        pairs = [(FakeDoc("sick leave text", "a"), 0.10), (FakeDoc("other text", "b"), 0.60)]
        return pairs[:k]


def test_dense_turns_distance_into_similarity_and_keeps_the_order():
    results = DenseSearch(FakeVectorStore()).retrieve("sick days?", k=5)
    assert [r["chunk_id"] for r in results] == ["a", "b"]
    assert results[0]["score"] == pytest.approx(0.90)          # 1 - 0.10
    assert results[1]["score"] == pytest.approx(0.40)
    assert results[0]["retriever"] == "dense"
    assert results[0]["rank"] == 1


# ---------------- hybrid (RRF) ----------------

class FixedSearch:
    def __init__(self, chunk_ids):
        self.chunk_ids = chunk_ids
        self.asked_for = None

    def retrieve(self, question, k=5):
        self.asked_for = k
        meta = {"file_name": "f.pdf", "section": "s", "page": 1, "doc_id": "d", "chunk_size": 80}
        return [make_result(rank, cid, "text " + cid, meta, None, "fixed")
                for rank, cid in enumerate(self.chunk_ids, start=1)][:k]


def test_hybrid_ranks_a_chunk_found_by_both_searches_first():
    bm25 = FixedSearch(["A", "B", "C"])
    dense = FixedSearch(["B", "C", "D"])
    results = HybridSearch(bm25, dense, rrf_k=60).retrieve("question", k=4)
    assert [r["chunk_id"] for r in results] == ["B", "C", "A", "D"]


def test_hybrid_score_is_the_sum_of_reciprocal_ranks():
    bm25 = FixedSearch(["A", "B", "C"])
    dense = FixedSearch(["B", "C", "D"])
    results = HybridSearch(bm25, dense, rrf_k=60).retrieve("question", k=4)
    # B is rank 2 in BM25 and rank 1 in dense:  1/(60+2) + 1/(60+1)
    assert results[0]["score"] == pytest.approx(round(1 / 62 + 1 / 61, 5))
    # A is rank 1 in BM25 only:  1/(60+1)
    assert results[2]["score"] == pytest.approx(round(1 / 61, 5))


def test_hybrid_respects_k_and_asks_each_search_for_more_candidates():
    bm25 = FixedSearch(["A", "B", "C"])
    dense = FixedSearch(["B", "C", "D"])
    results = HybridSearch(bm25, dense, candidates=20).retrieve("question", k=2)
    assert len(results) == 2
    assert bm25.asked_for == 20 and dense.asked_for == 20
    assert results[0]["retriever"] == "hybrid"
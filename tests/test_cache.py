# test_cache.py  (Step 13)
# Exact hit, semantic hit, no hit for different numbers, KB-version invalidation.
# The embedding model is replaced by a dictionary of tiny hand-made vectors, so
# every similarity in these tests is known in advance.
import pytest

from helpers import FakePipeline
from src.cache.cache_manager import CachedRAG
from src.cache.exact_cache import make_key, normalize_question
from src.cache.semantic_cache import cosine, extract_numbers, facts_match
from src.storage import sqlite_store

Q_SICK = "How many sick leave days do I get?"
Q_SICK_PARAPHRASE = "How many days of sick leave per year am I entitled to?"
Q_MATERNITY = "How many maternity leave days do I get?"
Q_CLAIM_LOW = "Who approves claims up to Rs. 5000?"
Q_CLAIM_HIGH = "Who approves claims between Rs. 5001 and Rs. 25000?"
Q_HOTEL = "What is the hotel limit per night?"

VECTORS = {
    Q_SICK: [1.0, 0.0, 0.0],
    Q_SICK_PARAPHRASE: [0.98, 0.20, 0.0],        # cosine with Q_SICK = 0.98  (very close)
    Q_MATERNITY: [1.0, 0.0, 0.0],                # SAME vector as Q_SICK on purpose: only the guard can stop it
    Q_CLAIM_LOW: [0.0, 1.0, 0.0],
    Q_CLAIM_HIGH: [0.0, 1.0, 0.0],               # identical vector, different numbers
    Q_HOTEL: [0.0, 0.0, 1.0],                    # nothing in common with the others
}


def fake_embed(text):
    return VECTORS.get(text, [0.5, 0.5, 0.5])


def make_cache(tmp_path, pipeline=None, kb_version="v1", threshold=0.90):
    db_path = str(tmp_path / "cache.sqlite")
    conn = sqlite_store.connect(db_path)
    sqlite_store.create_tables(conn)
    conn.close()
    if pipeline is None:
        pipeline = FakePipeline()
    return CachedRAG(pipeline, embed_fn=fake_embed, db_path=db_path,
                     kb_version=kb_version, threshold=threshold)


# ---------------- small helper functions ----------------

def test_normalize_makes_spellings_equal():
    assert normalize_question("How many sick days?") == normalize_question("  how MANY sick   days ")


def test_normalize_keeps_different_numbers_different():
    assert normalize_question("Rs. 5,000") == normalize_question("rs 5000")
    assert normalize_question("limit 3.5") != normalize_question("limit 35")


def test_cache_key_depends_on_the_kb_version():
    assert make_key(Q_SICK, "v1") != make_key(Q_SICK, "v2")
    assert make_key(Q_SICK, "v1") == make_key(Q_SICK.lower(), "v1")


def test_extract_numbers():
    assert extract_numbers("between Rs. 5,001 and Rs. 25,000 within 14 days") == {"5001", "25000", "14"}


def test_facts_match_rules():
    assert facts_match(Q_SICK, Q_SICK_PARAPHRASE) is True
    assert facts_match(Q_CLAIM_LOW, Q_CLAIM_HIGH) is False                       # numbers differ
    assert facts_match(Q_SICK, Q_MATERNITY) is False                             # leave type differs
    assert facts_match("notice period during probation", "notice period after confirmation") is False
    assert facts_match("hotel limit domestic", "hotel limit international") is False


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0


# ---------------- the cache as a whole ----------------

def test_exact_hit(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline)
    first = cache.ask(Q_SICK)
    second = cache.ask("how many SICK leave days do i get")        # other spelling
    assert first["cache"] == "none"
    assert second["cache"] == "exact"
    assert second["answer"] == first["answer"]
    assert second["citations"] == first["citations"]
    assert pipeline.calls == 1                                      # the pipeline ran only once


def test_semantic_hit_for_a_paraphrase(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    result = cache.ask(Q_SICK_PARAPHRASE)
    assert result["cache"] == "semantic"
    assert result["cache_similarity"] >= 0.90
    assert result["refused"] is False
    assert pipeline.calls == 1


def test_no_hit_when_the_numbers_differ_even_if_the_vectors_are_identical(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_CLAIM_LOW)
    result = cache.ask(Q_CLAIM_HIGH)
    assert result["cache"] == "none"
    assert pipeline.calls == 2


def test_no_hit_when_a_key_term_differs_even_if_the_vectors_are_identical(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    assert cache.ask(Q_MATERNITY)["cache"] == "none"
    assert pipeline.calls == 2


def test_no_hit_below_the_threshold(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    assert cache.ask(Q_HOTEL)["cache"] == "none"


def test_a_higher_threshold_turns_a_semantic_hit_into_a_miss(tmp_path):
    pipeline = FakePipeline()
    cache = make_cache(tmp_path, pipeline, threshold=0.99)           # the paraphrase has 0.98
    cache.ask(Q_SICK)
    assert cache.ask(Q_SICK_PARAPHRASE)["cache"] == "none"


def test_changing_the_kb_version_invalidates_old_entries(tmp_path):
    pipeline = FakePipeline()
    old_kb = make_cache(tmp_path, pipeline, kb_version="v1")
    old_kb.ask(Q_SICK)
    assert old_kb.ask(Q_SICK)["cache"] == "exact"

    new_kb = make_cache(tmp_path, pipeline, kb_version="v2")         # same database file, new version
    assert new_kb.ask(Q_SICK)["cache"] == "none"                     # not exact ...
    assert new_kb.ask(Q_SICK_PARAPHRASE)["cache"] in ("none", "semantic")
    assert pipeline.calls == 2


def test_old_version_entries_can_be_deleted(tmp_path):
    cache_v1 = make_cache(tmp_path, kb_version="v1")
    cache_v1.ask(Q_SICK)
    cache_v2 = make_cache(tmp_path, kb_version="v2")
    assert cache_v2.exact.clear_old_versions() == 1


def test_refusals_are_not_cached(tmp_path):
    pipeline = FakePipeline(refused=True, citations=[])
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    assert cache.ask(Q_SICK)["cache"] == "none"
    assert pipeline.calls == 2


def test_answers_with_citation_warnings_are_not_cached(tmp_path):
    pipeline = FakePipeline(warnings=["The answer cites passages that do not exist: [7]"])
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    assert cache.ask(Q_SICK)["cache"] == "none"


def test_answers_without_citations_are_not_cached(tmp_path):
    pipeline = FakePipeline(citations=[])
    cache = make_cache(tmp_path, pipeline)
    cache.ask(Q_SICK)
    assert cache.ask(Q_SICK)["cache"] == "none"


def test_an_empty_question_is_rejected_before_any_cache_lookup(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(ValueError):
        cache.ask("   ")
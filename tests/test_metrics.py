# test_metrics.py  (Step 13)
# The metrics are checked against answers we calculated BY HAND.
import pytest

from src.evaluation import metrics


def test_recall_is_one_when_a_relevant_chunk_is_in_the_top_k():
    assert metrics.recall_at_k([0, 1, 0, 1], 3) == 1.0


def test_recall_is_zero_when_the_relevant_chunk_is_outside_the_top_k():
    assert metrics.recall_at_k([0, 0, 0, 1], 3) == 0.0


def test_reciprocal_rank_first_second_third_and_missing():
    assert metrics.reciprocal_rank([1, 0, 0]) == 1.0
    assert metrics.reciprocal_rank([0, 1, 0, 1]) == 0.5
    assert metrics.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([0, 0, 0]) == 0.0


def test_reciprocal_rank_ignores_results_after_k():
    assert metrics.reciprocal_rank([0, 0, 1], k=2) == 0.0


def test_dcg_discounts_lower_ranks():
    # rank 1 counts fully, rank 2 counts 1/log2(3)
    assert metrics.dcg_at_k([1, 1], 2) == pytest.approx(1 + 1 / 1.5849625, abs=1e-6)


def test_ndcg_worked_example_from_the_code_comments():
    # DCG = 0.6309 + 0.4307 = 1.0616, ideal = 1.6309, nDCG = 0.651
    assert metrics.ndcg_at_k([0, 1, 0, 1], total_relevant=2, k=4) == pytest.approx(0.651, abs=0.001)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_for_nothing_found():
    assert metrics.ndcg_at_k([1, 0, 0], total_relevant=1, k=3) == 1.0
    assert metrics.ndcg_at_k([0, 0, 0], total_relevant=1, k=3) == 0.0


def test_ndcg_with_no_relevant_chunks_in_the_knowledge_base_is_zero():
    assert metrics.ndcg_at_k([0, 0], total_relevant=0, k=2) == 0.0


def test_mean():
    assert metrics.mean([1, 2, 3]) == 2.0
    assert metrics.mean([]) == 0.0


def test_percent_agreement():
    assert metrics.percent_agreement([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75
    assert metrics.percent_agreement([], []) is None


def test_cohen_kappa_worked_example():
    # observed 4/6, expected 0.5  ->  kappa = 1/3
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 0, 0, 1, 1]
    assert metrics.cohen_kappa(a, b) == pytest.approx(1 / 3, abs=1e-6)


def test_cohen_kappa_edge_cases():
    assert metrics.cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0      # perfect agreement
    assert metrics.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0      # one label only, same on both sides
    assert metrics.cohen_kappa([1, 1], [0, 0]) == pytest.approx(0.0)   # total disagreement
    assert metrics.cohen_kappa([], []) is None
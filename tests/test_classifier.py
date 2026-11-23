# test_classifier.py  (Step 13)
# The trained model itself is checked in Step 9 (precision / recall / F1).
# Here we test the CODE AROUND it, with the model replaced by a fake probability.
import pytest

from helpers import make_pipeline
from src.classifier.predict import ScopeClassifier
from src.rag.prompt import NOT_FOUND_MESSAGE


def make_checker(monkeypatch, probabilities, threshold=0.5):
    """A ScopeClassifier whose model is replaced by a dictionary: question -> probability."""
    checker = object.__new__(ScopeClassifier)            # skip __init__, which loads the real model
    checker.threshold = threshold
    monkeypatch.setattr(ScopeClassifier, "probability_in_scope",
                        lambda self, question: probabilities[question])
    return checker


def test_returns_label_and_confidence(monkeypatch):
    checker = make_checker(monkeypatch, {"sick days?": 0.93, "cricket?": 0.04})
    assert checker("sick days?") == (True, 0.93)
    assert checker("cricket?") == (False, 0.04)


def test_threshold_boundary_counts_as_in_scope(monkeypatch):
    checker = make_checker(monkeypatch, {"exactly": 0.5, "just below": 0.49})
    assert checker("exactly")[0]
    assert not checker("just below")[0]


def test_a_stricter_threshold_changes_the_decision(monkeypatch):
    checker = make_checker(monkeypatch, {"borderline": 0.6}, threshold=0.8)
    assert not checker("borderline")[0]


def test_missing_model_folder_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        ScopeClassifier(model_dir=str(tmp_path))
    assert "train_classifier" in str(error.value)


def test_pipeline_refuses_early_when_classifier_says_out_of_scope():
    pipeline = make_pipeline(scope_checker=lambda question: (False, 0.02))
    result = pipeline.ask("Who won the cricket world cup?")
    assert result["refused"] is True
    assert result["refusal_reason"] == "out_of_scope_classifier"
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert pipeline.retriever.calls == 0          # nothing was retrieved ...
    assert pipeline.generator.calls == 0          # ... and the LLM was never called


def test_pipeline_continues_when_classifier_says_in_scope():
    pipeline = make_pipeline(scope_checker=lambda question: (True, 0.97))
    result = pipeline.ask("How many sick leave days do I get?")
    assert result["refused"] is False
    assert pipeline.generator.calls == 1
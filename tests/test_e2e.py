# test_e2e.py  (Step 13)
# End to end: question in -> answer with citations (or a refusal) out.
# The real pipeline, cache and API code run; only the LLM, the search results and the
# classifier are fakes.
import pytest

from helpers import FakeRetriever, make_chunk, make_pipeline
from src.cache.cache_manager import CachedRAG
from src.rag.generator import GenerationError
from src.rag.pipeline import clean_answer, clean_question, parse_citations
from src.rag.prompt import NOT_FOUND_MESSAGE, build_messages
from src.storage import sqlite_store


# ---------------- answer with citations ----------------

def test_in_scope_question_gets_an_answer_with_a_correct_citation():
    pipeline = make_pipeline(reply="Employees get 12 days of paid sick leave [1].")
    result = pipeline.ask("How many sick leave days do I get?")
    assert result["refused"] is False
    assert "12 days" in result["answer"]
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["number"] == 1
    assert citation["file_name"] == "01_Leave_Policy.pdf"
    assert citation["section"] == "3. Sick Leave"
    assert citation["page"] == 1
    assert result["citation_warnings"] == []


def test_the_prompt_contains_numbered_passages_and_the_question():
    messages = build_messages("How many sick days?", [make_chunk(1), make_chunk(2, text="Second passage.")])
    system_text, human_text = messages[0][1], messages[1][1]
    assert NOT_FOUND_MESSAGE in system_text
    assert "[1] Source: 01_Leave_Policy.pdf" in human_text
    assert "[2] Source:" in human_text
    assert "Question: How many sick days?" in human_text


def test_a_second_citation_points_to_the_right_chunk():
    pipeline = make_pipeline(reply="12 days [1]. A medical certificate is needed [2].")
    result = pipeline.ask("sick leave rules?")
    assert [c["number"] for c in result["citations"]] == [1, 2]
    assert result["citations"][1]["page"] == 2


# ---------------- refusals ----------------

def test_out_of_scope_question_is_refused_without_calling_the_llm():
    pipeline = make_pipeline(scope_checker=lambda question: (False, 0.01))
    result = pipeline.ask("Who won the cricket world cup?")
    assert result["refused"] is True
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["citations"] == []
    assert pipeline.generator.calls == 0


def test_low_similarity_is_refused_without_calling_the_llm():
    pipeline = make_pipeline(gate_score=0.10)           # min_similarity is 0.25
    result = pipeline.ask("What is the capital of France?")
    assert result["refused"] is True
    assert result["refusal_reason"] == "low_similarity"
    assert pipeline.generator.calls == 0


def test_when_the_model_says_it_does_not_know_the_result_is_a_refusal():
    pipeline = make_pipeline(reply=NOT_FOUND_MESSAGE)
    result = pipeline.ask("What is the bonus for senior managers?")
    assert result["refused"] is True
    assert result["refusal_reason"] == "model_said_unknown"
    assert result["citations"] == []


# ---------------- citation checks ----------------

def test_a_citation_to_a_missing_passage_is_flagged():
    pipeline = make_pipeline(reply="12 days [7].")
    result = pipeline.ask("sick leave?")
    assert result["citations"] == []
    assert any("do not exist" in w for w in result["citation_warnings"])


def test_an_answer_without_any_citation_is_flagged():
    pipeline = make_pipeline(reply="12 days.")
    result = pipeline.ask("sick leave?")
    assert "The answer has no citation." in result["citation_warnings"]


def test_repeated_citation_markers_are_cleaned():
    assert clean_answer("[1] [1] You get 12 days.") == "[1] You get 12 days."


def test_parse_citations_reports_invalid_numbers():
    citations, invalid = parse_citations("a [1] b [3]", [make_chunk(1), make_chunk(2)])
    assert [c["number"] for c in citations] == [1]
    assert invalid == [3]


# ---------------- input checks ----------------

def test_empty_and_too_long_questions_are_rejected():
    with pytest.raises(ValueError):
        clean_question("   ")
    with pytest.raises(ValueError):
        clean_question("x" * 501)
    assert clean_question("  How   many   days?  ") == "How many days?"


# ---------------- pipeline + cache together ----------------

def make_cached(tmp_path, pipeline):
    db_path = str(tmp_path / "e2e.sqlite")
    conn = sqlite_store.connect(db_path)
    sqlite_store.create_tables(conn)
    conn.close()
    return CachedRAG(pipeline, embed_fn=lambda text: [1.0, 0.0], db_path=db_path,
                     kb_version="v1", threshold=0.90)


def test_the_second_identical_question_is_served_from_cache(tmp_path):
    pipeline = make_pipeline()
    cached = make_cached(tmp_path, pipeline)
    first = cached.ask("How many sick leave days do I get?")
    second = cached.ask("how many sick leave days do i get")
    assert first["cache"] == "none" and second["cache"] == "exact"
    assert second["answer"] == first["answer"]
    assert second["citations"][0]["file_name"] == "01_Leave_Policy.pdf"
    assert pipeline.generator.calls == 1


def test_a_refused_question_is_asked_again_not_cached(tmp_path):
    pipeline = make_pipeline(scope_checker=lambda question: (False, 0.01))
    cached = make_cached(tmp_path, pipeline)
    assert cached.ask("Who won the cricket world cup?")["cache"] == "none"
    assert cached.ask("Who won the cricket world cup?")["cache"] == "none"


# ---------------- the web API ----------------

class FakeRag:
    def __init__(self, error=None):
        self.error = error

    def ask(self, question):
        if self.error is not None:
            raise self.error
        return {"question": question, "answer": "12 days [1]", "refused": False,
                "refusal_reason": None,
                "citations": [{"number": 1, "file_name": "01_Leave_Policy.pdf",
                               "section": "3. Sick Leave", "page": 1, "snippet": "12 days of paid sick leave"}],
                "citation_warnings": [], "cache": "none", "cache_similarity": None}


def make_client(monkeypatch, rag):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.main import app
    monkeypatch.setattr(app.state, "rag", rag, raising=False)
    return TestClient(app)          # used WITHOUT "with", so the real models are not loaded


def test_api_ask_returns_answer_citations_and_cache_status(monkeypatch):
    client = make_client(monkeypatch, FakeRag())
    reply = client.post("/ask", json={"question": "How many sick leave days do I get?"})
    assert reply.status_code == 200
    data = reply.json()
    assert data["answer"] == "12 days [1]"
    assert data["refused"] is False
    assert data["citations"][0]["file_name"] == "01_Leave_Policy.pdf"
    assert data["cache"] == "none"
    assert data["response_time_seconds"] >= 0


def test_api_rejects_an_empty_question(monkeypatch):
    client = make_client(monkeypatch, FakeRag())
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_turns_a_bad_question_into_http_400(monkeypatch):
    client = make_client(monkeypatch, FakeRag(error=ValueError("The question is empty.")))
    assert client.post("/ask", json={"question": "hello"}).status_code == 400


def test_api_turns_an_ollama_failure_into_http_503(monkeypatch):
    client = make_client(monkeypatch, FakeRag(error=GenerationError("Ollama is not running")))
    reply = client.post("/ask", json={"question": "hello"})
    assert reply.status_code == 503
    assert "Ollama" in reply.json()["detail"]


def test_api_health_reports_ok_and_degraded(monkeypatch):
    client = make_client(monkeypatch, FakeRag())
    monkeypatch.setattr("src.api.main.check_ollama",
                        lambda: {"reachable": True, "model": "m", "model_available": True})
    assert client.get("/health").json()["status"] == "ok"
    monkeypatch.setattr("src.api.main.check_ollama",
                        lambda: {"reachable": False, "model": "m", "model_available": False})
    assert client.get("/health").json()["status"] == "degraded"
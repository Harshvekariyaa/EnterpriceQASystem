# test_error_analysis.py  (Step 14)
# Checks the error-analysis rules on tiny hand-made examples (no LLM, no database needed).

from src.evaluation import error_analysis as ea


def chunk(chunk_id, text, file_name="01_Leave_Policy.pdf", section="3. Sick Leave"):
    return {"chunk_id": chunk_id, "text": text, "file_name": file_name, "section": section, "page": 1}


def question(answerable="yes"):
    return {"question_id": "Q1", "question": "How many sick days?", "expected_answer": "12 days",
            "source_file": "01_Leave_Policy.pdf", "supporting_passage": "12 days of paid sick leave",
            "answerable": answerable}


def record(answer, retrieved, citations=None, refused=False, reason=None, warnings=None):
    return {"answer": answer, "refused": refused, "refusal_reason": reason, "retrieved": retrieved,
            "citations": citations or [], "citation_warnings": warnings or [], "top_similarity": 0.6}


GOOD = chunk("c1", "Employees get 12 days of paid sick leave.")
OTHER = chunk("c2", "Casual leave is 8 days.", section="4. Casual Leave")


def test_has_content_ignores_markers():
    assert ea.has_content("12 days [1]") is True
    assert ea.has_content("[1]") is False
    assert ea.has_content("[1] [2]") is False
    assert ea.has_content("[04_Expense.pdf | Section | Page") is False     # marker cut off by max_tokens


def test_good_answer_has_no_problem():
    r = record("12 days [1]", [GOOD], citations=[{"chunk_id": "c1"}])
    d = ea.diagnose(question(), r)
    assert d["primary_cause"] is None and d["penalty"] == 0


def test_empty_answer_is_a_generation_failure():
    r = record("[1]", [GOOD], citations=[{"chunk_id": "c1"}])
    d = ea.diagnose(question(), r)
    assert d["primary_cause"] == "empty_answer"
    assert d["evidence"]["gold_rank"] == 1          # retrieval was fine


def test_answering_an_unanswerable_question():
    r = record("It is 15 percent [1]", [OTHER], citations=[{"chunk_id": "c2"}])
    assert ea.diagnose(question("no"), r)["primary_cause"] == "answered_unanswerable"


def test_false_refusal():
    r = record("I could not find this in the company documents.", [], refused=True, reason="out_of_scope_classifier")
    d = ea.diagnose(question(), r)
    assert d["primary_cause"] == "false_refusal" and d["penalty"] == 4


def test_retrieval_miss():
    r = record("Casual leave is 8 days [1]", [OTHER], citations=[{"chunk_id": "c2"}])
    assert "retrieval_miss" in ea.diagnose(question(), r)["causes"]


def test_citing_extra_passages_is_flagged():
    r = record("12 days [1] [2]", [GOOD, OTHER], citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])
    assert ea.diagnose(question(), r)["primary_cause"] == "bad_citations"


def test_invalid_citation_number_is_flagged():
    r = record("12 days [1] [5]", [GOOD], citations=[{"chunk_id": "c1"}],
               warnings=["The answer cites passages that do not exist: [5]"])
    assert ea.diagnose(question(), r)["primary_cause"] == "invalid_citation_number"


def test_human_labels_add_penalty():
    r = record("20 days [1]", [GOOD], citations=[{"chunk_id": "c1"}])
    human = {"faithful": 0, "relevant": 1, "correct": 0, "citation_ok": 1}
    d = ea.diagnose(question(), r, human)
    assert d["primary_cause"] == "unsupported_answer" and d["penalty"] == 5      # 3 incorrect + 2 unsupported


def test_pick_worst_prefers_different_causes():
    rows = [{"question_id": "A", "penalty": 5, "failure_category": "empty_answer"},
            {"question_id": "B", "penalty": 5, "failure_category": "empty_answer"},
            {"question_id": "C", "penalty": 4, "failure_category": "false_refusal"},
            {"question_id": "D", "penalty": 1, "failure_category": "bad_citations"}]
    chosen = [r["question_id"] for r in ea.pick_worst(rows, 3)]
    assert chosen == ["A", "C", "D"]            # B is the same cause as A, so D comes first

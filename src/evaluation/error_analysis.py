# error_analysis.py  (Step 14 - part 1)
# Job: look at every answer, find what went wrong, name the CAUSE, and pick the 5 worst
# cases (one per different cause where possible, so the report shows variety).
#
# It works in two modes:
#   - WITHOUT human labels: uses automatic checks (refusals, empty answers, citations).
#   - WITH human labels (from qa_human_labels.csv): adds your scores to the penalty.

import os
import re

from src.evaluation.answer_eval import cited_chunks
from src.evaluation.retrieval_eval import normalize

# When one answer has several problems, the FIRST cause in this list is its main cause.
CAUSE_ORDER = ["answered_unanswerable", "false_refusal", "retrieval_miss", "empty_answer",
               "unsupported_answer", "bad_citations", "invalid_citation_number"]


# ------------------------------------------------------------------ small helpers
def strip_markers(answer):
    """Remove citation markers like [1] or [04_Expense.pdf | Section | Page: 1].
    Also removes a marker that was cut off at the end (the model ran out of tokens)."""
    text = re.sub(r"\[[^\]]*\]", " ", answer)
    text = re.sub(r"\[[^\]]*$", " ", text)
    return text.strip()


def has_content(answer):
    """True if the answer contains real text, not only citation markers."""
    return re.search(r"[A-Za-z0-9]", strip_markers(answer)) is not None


def gold_chunks(question, record):
    """The retrieved chunks that contain the gold (supporting) passage, with their rank (1, 2, 3...)."""
    gold = normalize(question["supporting_passage"])
    found = []
    for rank, chunk in enumerate(record["retrieved"], start=1):
        if gold != "" and gold in normalize(chunk["text"]):
            found.append((rank, chunk))
    return found


# ------------------------------------------------------------------ 1. diagnose one answer
def diagnose(question, record, human=None):
    """Return a dictionary: causes (list), primary_cause, penalty (higher = worse), flags, evidence."""
    answerable = question["answerable"] == "yes"
    causes, flags, penalty = [], [], 0
    evidence = {"gold_rank": None, "cited": [], "cited_with_gold": 0}

    if not answerable and not record["refused"]:
        causes.append("answered_unanswerable")
        flags.append("answered a question that the documents cannot answer")
        penalty = penalty + 4

    if answerable and record["refused"]:
        causes.append("false_refusal")
        flags.append("false refusal (" + str(record["refusal_reason"]) + ")")
        penalty = penalty + 4

    if not record["refused"]:
        chunks = cited_chunks(record)
        evidence["cited"] = [c["file_name"] + " / " + str(c["section"]) for c in chunks]

        # Was the supporting text even retrieved? (only meaningful for answerable questions)
        if answerable:
            found = gold_chunks(question, record)
            if len(found) == 0:
                causes.append("retrieval_miss")
                flags.append("supporting passage was NOT in the retrieved context")
                penalty = penalty + 3
            else:
                evidence["gold_rank"] = found[0][0]
                gold_ids = [c["chunk_id"] for rank, c in found]
                evidence["cited_with_gold"] = sum(1 for c in chunks if c["chunk_id"] in gold_ids)

                wrong_source = [c for c in chunks if c["file_name"] != question["source_file"]]
                not_gold = [c for c in chunks if c["chunk_id"] not in gold_ids]
                if len(wrong_source) > 0:
                    causes.append("bad_citations")
                    flags.append("cites a passage from the wrong PDF")
                    penalty = penalty + 1
                elif len(not_gold) > 0 and len(chunks) > 1:
                    causes.append("bad_citations")
                    flags.append("cites passages that do not contain the supporting text")
                    penalty = penalty + 1

        # Is there any real sentence in the answer?
        if not has_content(record["answer"]):
            causes.append("empty_answer")
            flags.append("answer is only citation markers, no sentence")
            if human is None:
                penalty = penalty + 3

        if any("do not exist" in w for w in record.get("citation_warnings", [])):
            causes.append("invalid_citation_number")
            flags.append("cites a passage number that does not exist")
            penalty = penalty + 1

        # Your hand labels (only when the sheet is filled in)
        if human is not None and answerable:
            if human["correct"] == 0:
                penalty = penalty + 3
                flags.append("human: incorrect")
            if human["faithful"] < 1:
                penalty = penalty + (2 if human["faithful"] == 0 else 1)
                flags.append("human: unsupported claims")
                if has_content(record["answer"]):
                    causes.append("unsupported_answer")
            if human["relevant"] == 0:
                penalty = penalty + 2
                flags.append("human: not relevant")
            if human["citation_ok"] == 0:
                penalty = penalty + 1
                flags.append("human: citation does not support the answer")

    primary = None
    for cause in CAUSE_ORDER:
        if cause in causes:
            primary = cause
            break
    return {"question_id": question["question_id"], "causes": causes, "primary_cause": primary,
            "penalty": penalty, "flags": flags, "evidence": evidence}


# ------------------------------------------------------------------ 2. explain the cause
def explain(cause, question, record, diagnosis, context):
    """Return (reason_for_failure, proposed_improvement) in plain words, using real numbers."""
    evidence = diagnosis["evidence"]
    answer_text = " ".join(record["answer"].split())
    similarity = record.get("top_similarity")
    gate = context.get("min_similarity")

    if cause == "answered_unanswerable":
        reason = ("The documents do not contain this. The similarity gate let it through (top similarity "
                  + (str(round(similarity, 2)) if similarity is not None else "n/a")
                  + ", gate " + str(gate) + "), the model then invented an answer and cited "
                  + (", ".join(evidence["cited"]) if evidence["cited"] else "a passage") + ".")
        low = context.get("lowest_answerable_similarity")
        fix = ("Add a grounding check after generation: every number in the answer must appear in the cited "
               "chunk, otherwise refuse. Raising min_similarity alone is not enough")
        if low is not None and similarity is not None and similarity >= low:
            fix = fix + (": this question scored " + str(round(similarity, 2)) + ", above some answerable "
                         "questions (lowest " + str(round(low, 2)) + ").")
        else:
            fix = fix + "."
        return reason, fix

    if cause == "false_refusal":
        reason = ("Refused by '" + str(record["refusal_reason"]) + "' before the model was asked, although the "
                  "answer exists in " + question["source_file"] + ".")
        if record["refusal_reason"] == "out_of_scope_classifier":
            span = context.get("classifier_miss_range")
            if span is not None:
                reason = reason + (" The classifier's own test-set misses scored only " + str(span[0]) + " to "
                                   + str(span[1]) + " (threshold 0.5), so these are borderline calls.")
        fix = ("Refuse early only when the classifier is confident (for example probability below 0.3). In the grey "
               "zone 0.3 to 0.5 let retrieval and the similarity gate decide. Choose the cut-off on the validation "
               "set, then confirm on the test set. Add more short 'what time / when' in-scope questions to training.")
        return reason, fix

    if cause == "retrieval_miss":
        reason = "The supporting passage was not among the chunks given to the model, so it could not answer."
        fix = "Raise top_k / max_context_chunks, or compare hybrid fusion weights; check the chunk boundaries."
        return reason, fix

    if cause == "empty_answer":
        reason = ("Retrieval worked (supporting text at rank " + str(evidence["gold_rank"]) + " of "
                  + str(len(record["retrieved"])) + ") but the model wrote no sentence, only: '"
                  + answer_text[:70] + ("..." if len(answer_text) > 70 else "") + "'. This is a generation failure, "
                  "not a retrieval failure.")
        fix = ("Use a larger generator (llama3.2:3b) and add one worked example to the prompt "
               "('Q: ... A: You get 12 days of paid sick leave [1].'). Add an empty-answer guard in RAGPipeline: "
               "if no text is left after removing markers, retry once, then show the top cited sentence.")
        return reason, fix

    if cause == "unsupported_answer":
        reason = "The answer contains claims that the cited text does not support (marked in your hand labels)."
        fix = "Add the grounding check (numbers in the answer must appear in the cited chunk) and shorten the answer."
        return reason, fix

    if cause == "bad_citations":
        reason = ("The answer cites " + str(len(evidence["cited"])) + " passages but only "
                  + str(evidence["cited_with_gold"]) + (" contains" if evidence["cited_with_gold"] == 1 else " contain")
                  + " the supporting text: " + "; ".join(evidence["cited"]) + ".")
        fix = ("Tell the model to cite only the one passage it used, and filter citations after generation: keep a "
               "cited chunk only if it shares key words or numbers with the answer.")
        return reason, fix

    if cause == "invalid_citation_number":
        reason = "The model cited a passage number that does not exist, and the marker stays in the shown answer."
        fix = "In parse_citations, also remove invalid [n] markers from the answer text (today they are only flagged)."
        return reason, fix

    return "", ""


# ------------------------------------------------------------------ 3. pick the worst five
def analyse_all(questions, runs, humans, context):
    """Diagnose every question. Returns a list of result rows (all questions with at least one problem)."""
    rows = []
    for q in questions:
        qid = q["question_id"]
        human = humans.get(qid) if humans else None
        d = diagnose(q, runs[qid], human)
        if d["primary_cause"] is None and d["penalty"] == 0:
            continue
        record = runs[qid]
        reason, fix = explain(d["primary_cause"], q, record, d, context)
        context_text = " || ".join(c["file_name"] + " / " + str(c["section"]) + ": "
                                   + c["text"].replace("\n", " ")[:160] for c in record["retrieved"])
        rows.append({"question_id": qid, "penalty": d["penalty"], "failure_category": d["primary_cause"],
                     "all_causes": "; ".join(d["causes"]), "question": q["question"],
                     "expected_answer": q["expected_answer"], "actual_answer": " ".join(record["answer"].split()),
                     "retrieved_context": context_text, "reason_for_failure": reason,
                     "proposed_improvement": fix, "automatic_flags": "; ".join(d["flags"])})
    return rows


def pick_worst(rows, n=5):
    """Take the n worst rows, but first one per DIFFERENT main cause (worst of each), then fill up."""
    ranked = sorted(rows, key=lambda r: (-r["penalty"], r["question_id"]))
    chosen, seen = [], []
    for row in ranked:
        if row["failure_category"] not in seen and len(chosen) < n:
            chosen.append(row)
            seen.append(row["failure_category"])
    for row in ranked:
        if row not in chosen and len(chosen) < n:
            chosen.append(row)
    chosen.sort(key=lambda r: (-r["penalty"], r["question_id"]))
    return chosen


def cause_counts(rows, total_questions):
    """How often each main cause happens (used for the summary table)."""
    counts = {}
    for row in rows:
        counts[row["failure_category"]] = counts.get(row["failure_category"], 0) + 1
    out = []
    for cause in CAUSE_ORDER:
        if cause in counts:
            out.append({"failure_category": cause, "questions": counts[cause], "out_of": total_questions})
    return out


# ------------------------------------------------------------------ 4. are the saved answers out of date?
WATCHED_FILES = ["config.yaml", "src/ingestion/chunker.py", "src/ingestion/cleaner.py", "src/rag/pipeline.py",
                 "src/rag/prompt.py", "src/rag/generator.py", "src/retrieval/hybrid_retriever.py",
                 "data/db/rag.sqlite"]


def stale_files(root, runs_path):
    """Code or data files that were changed AFTER the saved answers were generated."""
    runs_time = os.path.getmtime(runs_path)
    newer = []
    for name in WATCHED_FILES:
        path = os.path.join(root, name)
        if os.path.exists(path) and os.path.getmtime(path) > runs_time + 1:
            newer.append(name)
    return newer

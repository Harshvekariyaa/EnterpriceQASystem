# answer_eval.py  (Step 10 - part 2)
# Job: run the pipeline on the evaluation questions, make the human label sheet,
# run the LLM judge, and compute every metric (with numerator and denominator).

import csv
import json
import os

from src.evaluation import metrics
from src.evaluation.retrieval_eval import normalize
from src.rag.prompt import build_context

SHEET_COLUMNS = ["question_id", "question", "expected_answer", "gold_passage", "answer",
                 "cited_sources", "cited_text", "retrieved_context",
                 "human_faithful", "human_relevant", "human_correct", "human_citation_ok", "notes"]
LABEL_FIELDS = {"human_faithful": ("faithful", [0, 0.5, 1]),
                "human_relevant": ("relevant", [0, 1]),
                "human_correct": ("correct", [0, 1]),
                "human_citation_ok": ("citation_ok", [0, 1])}


# ------------------------------------------------------------------ files
def load_questions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:     # utf-8-sig also reads Excel files
        return list(csv.DictReader(f))


def load_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into {question_id: record}."""
    records = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[record["question_id"]] = record
    return records


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ 1. generate answers
def generate_answers(pipeline, questions, runs_path, say=print):
    """Ask the pipeline every question. Already finished questions are skipped,
    so you can stop and restart. A failed question is NOT saved, so it is retried next time."""
    done = load_jsonl(runs_path)
    for q in questions:
        if q["question_id"] in done:
            continue
        say("asking " + q["question_id"] + ": " + q["question"])
        try:
            result = pipeline.ask(q["question"])
        except Exception as error:
            say("   FAILED (will be retried next run): " + str(error))
            continue
        record = dict(result)
        record["question_id"] = q["question_id"]
        record["answerable"] = q["answerable"]
        append_jsonl(runs_path, record)
        say("   " + ("REFUSED (" + str(result["refusal_reason"]) + ")" if result["refused"]
                     else "answered in " + str(result["latency_seconds"]) + " s"))


# ------------------------------------------------------------------ 2. human label sheet
def cited_chunks(record):
    """The chunks that the answer cites, as a list of dictionaries."""
    by_id = {}
    for chunk in record["retrieved"]:
        by_id[chunk["chunk_id"]] = chunk
    result = []
    for citation in record["citations"]:
        if citation["chunk_id"] in by_id:
            result.append(by_id[citation["chunk_id"]])
    return result


def make_label_sheet(questions, runs, sheet_path):
    """Write the CSV the human fills in. Only ANSWERED questions need labels."""
    rows = []
    for q in questions:
        record = runs[q["question_id"]]
        if record["refused"]:
            continue
        chunks = cited_chunks(record)
        rows.append({
            "question_id": q["question_id"], "question": q["question"],
            "expected_answer": q["expected_answer"], "gold_passage": q["supporting_passage"],
            "answer": record["answer"],
            "cited_sources": "; ".join(c["file_name"] + " / " + str(c["section"]) for c in chunks),
            "cited_text": "\n---\n".join(c["text"] for c in chunks),
            "retrieved_context": build_context(record["retrieved"]),
            "human_faithful": "", "human_relevant": "", "human_correct": "",
            "human_citation_ok": "", "notes": "",
        })
    with open(sheet_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_sheet(rows, sheet_path):
    with open(sheet_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_label_line(line):
    """Turn '1 0.5 1 0' into [1, 0.5, 1, 0] and check each value is allowed. Raises ValueError."""
    parts = line.replace(",", ".").split()
    if len(parts) != 4:
        raise ValueError("type exactly 4 numbers, for example:  1 1 1 1")
    values = []
    for part, (column, (name, allowed)) in zip(parts, LABEL_FIELDS.items()):
        try:
            value = float(part)
        except ValueError:
            raise ValueError("'" + part + "' is not a number")
        if value not in allowed:
            raise ValueError(name + " must be one of " + str(allowed) + ", not " + part)
        values.append(value)
    return values


def label_interactively(sheet_path, input_fn=input, say=print):
    """Show each unlabelled answer in the terminal and ask for the 4 labels.
    The sheet is saved after EVERY answer, so you can quit (q) and continue later."""
    with open(sheet_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    todo = [r for r in rows if any((r[c] or "").strip() == "" for c in LABEL_FIELDS)]
    say("Answers still to label: " + str(len(todo)) + " of " + str(len(rows)))
    say("Type 4 numbers:  faithful(1/0.5/0)  relevant(1/0)  correct(1/0)  citation_ok(1/0)")
    say("Example: 1 1 1 1     |   s = skip   |   q = save and quit\n")

    for number, row in enumerate(todo, start=1):
        say("=" * 90)
        say("[" + str(number) + "/" + str(len(todo)) + "]  " + row["question_id"])
        say("QUESTION  : " + row["question"])
        say("EXPECTED  : " + row["expected_answer"])
        say("ANSWER    : " + row["answer"])
        say("CITED     : " + (row["cited_sources"] or "(no citation)"))
        say("CITED TEXT: " + (row["cited_text"] or "(none)").replace("\n", " "))
        say("--- text the model was shown ---")
        say(row["retrieved_context"])
        say("--------------------------------")
        while True:
            line = input_fn("faithful relevant correct citation_ok > ").strip().lower()
            if line == "q":
                write_sheet(rows, sheet_path)
                say("Saved. Run the same command again to continue.")
                return False
            if line == "s":
                break
            try:
                values = parse_label_line(line)
            except ValueError as error:
                say("  not accepted: " + str(error))
                continue
            for column, value in zip(LABEL_FIELDS, values):
                row[column] = str(int(value)) if value == int(value) else str(value)
            write_sheet(rows, sheet_path)          # save after every answer
            break
    write_sheet(rows, sheet_path)
    say("\nAll answers handled. Your labels are saved in " + sheet_path)
    return True


def read_human_labels(sheet_path):
    """Read the filled sheet. Stops with a clear message if any label is missing or invalid."""
    humans = {}
    problems = []
    with open(sheet_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels = {}
            for column, (name, allowed) in LABEL_FIELDS.items():
                text = (row[column] or "").strip().replace(",", ".")
                try:
                    value = float(text)
                except ValueError:
                    problems.append(row["question_id"] + ": '" + column + "' is empty or not a number")
                    continue
                if value not in allowed:
                    problems.append(row["question_id"] + ": '" + column + "' must be one of " + str(allowed))
                    continue
                labels[name] = value
            if len(labels) == len(LABEL_FIELDS):
                humans[row["question_id"]] = labels
    if problems:
        raise ValueError("The label sheet is not complete:\n  " + "\n  ".join(problems))
    return humans


# ------------------------------------------------------------------ 3. LLM judge
def run_judge(judge, questions, runs, judge_path, say=print):
    """Let the LLM grade every answered question. Resumable, like generate_answers."""
    done = load_jsonl(judge_path)
    for q in questions:
        record = runs[q["question_id"]]
        if record["refused"] or q["question_id"] in done:
            continue
        say("judging " + q["question_id"])
        context = build_context(record["retrieved"])
        faithful, raw_f = judge.faithfulness(context, record["answer"])
        relevant, raw_r = judge.relevance(q["question"], record["answer"])
        correct, raw_c = judge.correctness(q["question"], q["expected_answer"], record["answer"])
        chunks = cited_chunks(record)
        if len(chunks) == 0:
            citation_ok, raw_ci = 0.0, "(no citation: scored 0 without asking the judge)"
        else:
            citation_ok, raw_ci = judge.citation(record["answer"], "\n---\n".join(c["text"] for c in chunks))
        append_jsonl(judge_path, {
            "question_id": q["question_id"],
            "faithful": faithful, "relevant": relevant, "correct": correct, "citation_ok": citation_ok,
            "raw": {"faithful": raw_f, "relevant": raw_r, "correct": raw_c, "citation_ok": raw_ci}})


# ------------------------------------------------------------------ 4. metrics
def ratio(numerator, denominator):
    return None if denominator == 0 else round(numerator / denominator, 3)


def metric_row(name, numerator, denominator, how):
    return {"metric": name, "value": ratio(numerator, denominator), "numerator": round(numerator, 3),
            "denominator": denominator, "how_measured": how}


def compute_summary(questions, runs, labels, who):
    """Every metric, with numerator and denominator. `labels` = human or judge labels."""
    answerable = [q for q in questions if q["answerable"] == "yes"]
    unanswerable = [q for q in questions if q["answerable"] != "yes"]
    answered = [q for q in questions if not runs[q["question_id"]]["refused"]]
    scored = [q for q in answered if labels.get(q["question_id"]) is not None]

    def values(field):
        out = []
        for q in scored:
            value = labels[q["question_id"]].get(field)
            if value is not None:
                out.append(value)
        return out

    rows = []
    if who == "human":
        refused_ok = sum(1 for q in unanswerable if runs[q["question_id"]]["refused"])
        refused_bad = sum(1 for q in answerable if runs[q["question_id"]]["refused"])
        rows.append(metric_row("Correct refusal rate", refused_ok, len(unanswerable),
                               "unanswerable questions that were refused / all unanswerable questions"))
        rows.append(metric_row("False refusal rate", refused_bad, len(answerable),
                               "answerable questions that were refused / all answerable questions"))
        rows.append(metric_row("Answers without any citation",
                               sum(1 for q in answered if len(runs[q["question_id"]]["citations"]) == 0),
                               len(answered), "answered questions with 0 citations / answered questions"))

    f, r, c, ci = values("faithful"), values("relevant"), values("correct"), values("citation_ok")
    rows.append(metric_row("Faithfulness (" + who + ")", sum(f), len(f),
                           "mean score: 1 = fully supported, 0.5 = partly, 0 = not supported; over answered questions"))
    rows.append(metric_row("Hallucination rate (" + who + ")", sum(1 for x in f if x < 1), len(f),
                           "answers with at least one unsupported claim (faithfulness < 1) / answered questions"))
    rows.append(metric_row("Answer relevance (" + who + ")", sum(r), len(r),
                           "answers that address the question / answered questions"))
    rows.append(metric_row("Answer correctness (" + who + ")", sum(c), len(c),
                           "answers matching the expected answer / answered questions"))
    rows.append(metric_row("Citation correctness (" + who + ")", sum(ci), len(ci),
                           "answers whose cited passages support them (no citation = 0) / answered questions"))

    if who == "human":
        total = source_ok = passage_ok = 0
        for q in answerable:
            record = runs[q["question_id"]]
            if record["refused"]:
                continue
            gold = normalize(q["supporting_passage"])
            for chunk in cited_chunks(record):
                total = total + 1
                if chunk["file_name"] == q["source_file"]:
                    source_ok = source_ok + 1
                if gold in normalize(chunk["text"]):
                    passage_ok = passage_ok + 1
        rows.append(metric_row("Citation: right source file (automatic)", source_ok, total,
                               "citations pointing to the expected PDF / all citations on answered answerable questions"))
        rows.append(metric_row("Citation: contains gold passage (automatic)", passage_ok, total,
                               "citations whose chunk contains the gold passage text / same denominator"))
    return rows


def compute_agreement(humans, judge):
    """Judge versus human, per criterion. Returns (agreement_rows, disagreement_rows)."""
    agreement, disagreements = [], []
    for field in ["faithful", "relevant", "correct", "citation_ok"]:
        pairs = []
        unparsed = 0
        for qid in humans:
            if qid not in judge:
                continue
            judge_value = judge[qid].get(field)
            if judge_value is None:
                unparsed = unparsed + 1
                continue
            pairs.append((qid, humans[qid][field], judge_value))
        h = [p[1] for p in pairs]
        j = [p[2] for p in pairs]
        # kappa is computed on a YES/NO version (for faithfulness: "fully supported" or not)
        hb = [1 if x == 1 else 0 for x in h]
        jb = [1 if x == 1 else 0 for x in j]
        kappa = metrics.cohen_kappa(hb, jb)
        agree = metrics.percent_agreement(h, j)
        agreement.append({
            "criterion": field, "n_compared": len(pairs), "judge_unparsed": unparsed,
            "percent_agreement": None if agree is None else round(agree, 3),
            "cohen_kappa": None if kappa is None else round(kappa, 3),
            "human_mean": None if not h else round(sum(h) / len(h), 3),
            "judge_mean": None if not j else round(sum(j) / len(j), 3),
            "judge_minus_human": None if not h else round(sum(j) / len(j) - sum(h) / len(h), 3)})
        for qid, hv, jv in pairs:
            if hv != jv:
                disagreements.append({"question_id": qid, "criterion": field, "human": hv, "judge": jv})
    return agreement, disagreements


def find_worst(questions, runs, humans, n=5):
    """Rank answers from worst to best with a simple penalty score."""
    ranked = []
    for q in questions:
        qid = q["question_id"]
        record = runs[qid]
        score, flags = 0, []
        if q["answerable"] == "yes" and record["refused"]:
            score, flags = score + 4, flags + ["false refusal (" + str(record["refusal_reason"]) + ")"]
        if q["answerable"] != "yes" and not record["refused"]:
            score, flags = score + 4, flags + ["answered a question that the documents cannot answer"]
        if not record["refused"] and qid in humans:
            h = humans[qid]
            if h["correct"] == 0:
                score, flags = score + 3, flags + ["incorrect"]
            if h["faithful"] < 1:
                score, flags = score + (2 if h["faithful"] == 0 else 1), flags + ["unsupported claims"]
            if h["relevant"] == 0:
                score, flags = score + 2, flags + ["not relevant"]
            if h["citation_ok"] == 0:
                score, flags = score + 1, flags + ["citation does not support the answer"]
        if score > 0:
            context = " || ".join(c["file_name"] + " / " + str(c["section"]) + ": " + c["text"].replace("\n", " ")[:200]
                                  for c in record["retrieved"])
            ranked.append({"question_id": qid, "penalty": score, "question": q["question"],
                           "expected_answer": q["expected_answer"], "actual_answer": record["answer"],
                           "retrieved_context": context, "automatic_flags": "; ".join(flags),
                           "reason_for_failure": "", "proposed_improvement": ""})
    ranked.sort(key=lambda r: (-r["penalty"], r["question_id"]))
    return ranked[:n]
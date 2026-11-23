# make_report.py  (Step 14)
# Run:  python scripts/make_report.py
# Builds reports/one_page_report.pdf from the CSV files that your earlier steps wrote.
# Four tables:  A retrieval   B classifier   C answer quality   D five error cases.
# Nothing is typed in by hand: every number comes from a file in reports/ or data/eval/.
# If your hand labels are missing, table C says so and the PDF is marked DRAFT.

import csv
import datetime
import os
import re
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import load_config, resolve_path
from src.evaluation import answer_eval as ae
from src.evaluation import error_analysis as ea

config = load_config()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = str(resolve_path(config["paths"]["eval_dir"]))
reports = str(resolve_path(config["paths"]["reports_dir"]))
runs_path = os.path.join(reports, "qa_runs.jsonl")
judge_path = os.path.join(reports, "qa_judge_runs.jsonl")
sheet_path = os.path.join(eval_dir, "qa_human_labels.csv")
out_path = os.path.join(reports, "one_page_report.pdf")


# ------------------------------------------------------------------ reading files
def read_csv(name):
    path = os.path.join(reports, name)
    if not os.path.exists(path):
        print("ERROR: missing", path)
        print("Run the step that creates it first (see the message above), then run this script again.")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def percent(value):
    return "n/a" if value is None else str(round(value * 100, 1)) + "%"


def fraction(row):
    return str(round(float(row["numerator"]), 1)).rstrip("0").rstrip(".") + " / " + str(row["denominator"])


# ------------------------------------------------------------------ the data for each table
retrieval = read_csv("retrieval_results.csv")
classifier = read_csv("classifier_results.csv")
worst = read_csv("error_analysis.csv")        # made by scripts/error_analysis.py

questions = ae.load_questions(os.path.join(eval_dir, "qa_eval.csv"))
runs = ae.load_jsonl(runs_path)
questions = [q for q in questions if q["question_id"] in runs]
judge = ae.load_jsonl(judge_path)

humans = None
if os.path.exists(sheet_path):
    try:
        humans = ae.read_human_labels(sheet_path)
    except ValueError:
        humans = None
stale = ea.stale_files(root, runs_path)
draft_reasons = []
if humans is None:
    draft_reasons.append("human labels are not filled in")
if stale:
    draft_reasons.append("saved answers are older than the code (" + ", ".join(stale[:3]) + ")")

# ---- Table C rows: (metric, value, n / d, basis)
c_rows = []
answered = [q for q in questions if not runs[q["question_id"]]["refused"]]
auto = ae.compute_summary(questions, runs, humans or {}, "human")
for r in auto:
    if humans is None and "(human)" in r["metric"]:
        continue                                   # no labels yet, so these are empty
    if r["value"] is None:
        continue
    name = r["metric"].replace(" (human)", "")
    c_rows.append((name, percent(r["value"]), fraction(r), "human" if "(human)" in r["metric"] else "automatic"))

no_sentence = sum(1 for q in answered if not ea.has_content(runs[q["question_id"]]["answer"]))
c_rows.append(("Answers with no sentence (only [n] markers)", percent(no_sentence / len(answered)),
               str(no_sentence) + " / " + str(len(answered)), "automatic"))

if len(judge) > 0:
    for r in ae.compute_summary(questions, runs, judge, "judge"):
        if r["value"] is not None:
            c_rows.append((r["metric"].replace(" (judge)", ""), percent(r["value"]), fraction(r), "LLM judge"))

agreement_text = "Judge vs human agreement: PENDING (fill in data/eval/qa_human_labels.csv, then run answer_eval.py report)."
if humans is not None and len(judge) > 0:
    agreement, disagreements = ae.compute_agreement(humans, judge)
    parts = []
    for a in agreement:
        parts.append(a["criterion"] + " " + percent(a["percent_agreement"]) + " (kappa " + str(a["cohen_kappa"]) + ")")
    agreement_text = "Judge vs human agreement: " + "; ".join(parts) + "."


# ------------------------------------------------------------------ Table B: confusion matrix from P / R / support
def confusion(model):
    """Rebuild the 2x2 confusion matrix (test split) from precision, recall and support."""
    rows = {r["class"]: r for r in classifier if r["model"] == model and r["split"] == "test"}
    pos, neg = rows["in_scope"], rows["out_of_scope"]
    tp = round(float(pos["recall"]) * int(pos["support"]))
    fn = int(pos["support"]) - tp
    tn = round(float(neg["recall"]) * int(neg["support"]))
    fp = int(neg["support"]) - tn
    # sanity check against the stored precision, so a wrong rebuild never reaches the report
    if tp + fp > 0 and abs(tp / (tp + fp) - float(pos["precision"])) > 0.01:
        return None
    return [["", "pred in", "pred out"], ["actual in", str(tp), str(fn)], ["actual out", str(fp), str(tn)]]


# ------------------------------------------------------------------ styles
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=6.6, leading=8)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=6.2, leading=7.4)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=6.1, leading=7.2)
HEAD = ParagraphStyle("head", parent=BODY, fontName="Helvetica-Bold", fontSize=8.6, leading=10,
                      textColor=colors.HexColor("#1f3a5f"), spaceBefore=5, spaceAfter=2)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=12.5, leading=15)
RED = ParagraphStyle("red", parent=BODY, fontName="Helvetica-Bold", textColor=colors.HexColor("#b00020"))


def p(text, style=CELL):
    return Paragraph(escape(str(text)), style)


def clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def grid(data, col_widths, header=True, highlight_rows=()):
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 6.6),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa5b1")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for row in highlight_rows:
        style += [("BACKGROUND", (0, row), (-1, row), colors.HexColor("#e9f5e9")),
                  ("FONTNAME", (0, row), (-1, row), "Helvetica-Bold")]
    table.setStyle(TableStyle(style))
    return table


# ------------------------------------------------------------------ build the page
width = A4[0] - 24 * mm
story = []
story.append(Paragraph(escape(config["project"]["name"]) + ": one-page experiment report", TITLE))
story.append(Paragraph(escape("Knowledge base " + config["project"]["kb_version"] + " (10 PDFs)  |  embeddings: "
                              + config["embedding"]["model_name"].split("/")[-1] + "  |  generator: "
                              + config["llm"]["model"] + "  |  judge: " + config["judge"]["model"]
                              + "  |  default: " + config["retrieval"]["best_retriever"] + ", "
                              + str(config["ingestion"]["default_chunk_size"]) + "-word chunks  |  generated "
                              + datetime.date.today().isoformat()), SMALL))
if draft_reasons:
    story.append(Paragraph("DRAFT: " + escape("; ".join(draft_reasons)) + ".", RED))

# A
story.append(Paragraph("A. Retrieval experiments (40 questions; relevant = chunk contains the gold phrase)", HEAD))
best_mrr = max(float(r["MRR"]) for r in retrieval)
data = [["Configuration", "Chunk size", "Retriever", "Recall@5", "MRR", "nDCG@5"]]
best_rows = []
for index, r in enumerate(retrieval, start=1):
    data.append([r["Configuration"], r["Chunk Size"], r["Retriever"], r["Recall@5"], r["MRR"], r["nDCG@5"]])
    if float(r["MRR"]) == best_mrr:
        best_rows.append(index)
story.append(grid(data, [width * f for f in (0.22, 0.14, 0.18, 0.15, 0.15, 0.16)], highlight_rows=best_rows))
story.append(Paragraph("Green = best MRR. The pipeline uses " + escape(config["retrieval"]["best_retriever"]) + " retrieval with "
                       + str(config["ingestion"]["default_chunk_size"]) + "-word chunks (config.yaml). "
                       "Limit: 10 short PDFs, so Recall@5 is near the ceiling and MRR is the more useful number.", SMALL))

# B
test_size = sum(int(r["support"]) for r in classifier if r["split"] == "test" and r["model"] == classifier[0]["model"])
story.append(Paragraph("B. Scope classifier (test split, " + str(test_size) + " questions; in-scope = positive class)", HEAD))
b_data = [["Model", "Class", "Precision", "Recall", "F1", "Support"]]
for r in classifier:
    if r["split"] == "test":
        b_data.append([r["model"], r["class"], r["precision"], r["recall"], r["f1"], r["support"]])
left = grid(b_data, [width * f for f in (0.18, 0.12, 0.08, 0.07, 0.07, 0.07)])
matrices = []
for model in ["baseline_logreg", "finetuned_minilm"]:
    m = confusion(model)
    if m is not None:
        matrices.append([Paragraph(model, SMALL), grid(m, [17 * mm, 14 * mm, 14 * mm])])
right = Table([[x] for pair in matrices for x in pair], colWidths=[48 * mm]) if matrices else Spacer(1, 1)
layout = Table([[left, right]], colWidths=[width * 0.62, width * 0.38])
layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(layout)
errors_path = os.path.join(reports, "classifier_errors.csv")
error_note = ""
if os.path.exists(errors_path):
    with open(errors_path, newline="", encoding="utf-8-sig") as f:
        misses = list(csv.DictReader(f))
    if misses:
        probs = [float(m["probability_in_scope"]) for m in misses]
        error_note = (str(len(misses)) + " test mistakes, e.g. '" + misses[0]["question"] + "' (in-scope probability "
                      + str(misses[0]["probability_in_scope"]) + "; threshold " + str(config["classifier"]["threshold"])
                      + "). Mistake probabilities range " + str(min(probs)) + " to " + str(max(probs)) + ". ")
story.append(Paragraph(escape(error_note) + "Small dataset written by us; see reports/classifier_errors.csv.", SMALL))

# C
story.append(Paragraph("C. Answer quality (" + str(len(questions)) + " questions: "
                       + str(sum(1 for q in questions if q["answerable"] == "yes")) + " answerable, "
                       + str(sum(1 for q in questions if q["answerable"] != "yes")) + " unanswerable; "
                       + str(len(answered)) + " answered)", HEAD))
c_data = [["Metric", "Value", "Calculation", "Basis"]]
for metric, value, calc, basis in c_rows:
    c_data.append([p(metric), value, calc, basis])
story.append(grid(c_data, [width * f for f in (0.50, 0.12, 0.16, 0.22)]))
story.append(Paragraph(escape(agreement_text) + " The judge is a small local model, so treat its scores as indicative until "
                       "they are compared with human labels.", SMALL))

# D
story.append(Paragraph("D. Five error cases (automatic analysis: reports/error_analysis.csv)", HEAD))
d_data = [["ID", "Question and what the system answered", "Why it failed", "Proposed fix"]]
for w in worst:
    d_data.append([w["question_id"],
                   p("Q: " + clip(w["question"], 80) + "  A: " + clip(w["actual_answer"], 90)),
                   p("[" + w["failure_category"] + "] " + clip(w["reason_for_failure"], 260)),
                   p(clip(w["proposed_improvement"], 260))])
story.append(grid(d_data, [width * f for f in (0.05, 0.27, 0.34, 0.34)]))

# Limitations
test_line = ""
test_file = os.path.join(reports, "test_results.txt")
if os.path.exists(test_file):
    with open(test_file, encoding="utf-8", errors="replace") as f:
        matches = re.findall(r"(\d+) passed(?:, (\d+) failed)?", f.read())
    if matches:
        test_line = " Tests: " + matches[-1][0] + " passed" + (", " + matches[-1][1] + " failed." if matches[-1][1] else ".")
story.append(Paragraph("Limitations: small self-written evaluation sets; one person labels the answers; the judge is a "
                       "small local model; results apply to these 10 PDFs only." + escape(test_line), SMALL))

doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm,
                        topMargin=10 * mm, bottomMargin=9 * mm, title="One-page experiment report")
doc.build(story)
print("Report written:", out_path)
if draft_reasons:
    print("It is marked DRAFT because:", "; ".join(draft_reasons))

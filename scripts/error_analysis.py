# error_analysis.py  (Step 14)
# Run:  python scripts/error_analysis.py
# Reads the saved answers (reports/qa_runs.jsonl) and, if you have filled it in, your hand labels.
# Writes:  reports/error_analysis.csv   (the 5 worst cases, with reason and proposed fix)
#          reports/error_causes.csv     (how often each cause happens)

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.evaluation import answer_eval as ae
from src.evaluation import error_analysis as ea

config = load_config()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = str(resolve_path(config["paths"]["eval_dir"]))
reports = str(resolve_path(config["paths"]["reports_dir"]))
runs_path = os.path.join(reports, "qa_runs.jsonl")
sheet_path = os.path.join(eval_dir, "qa_human_labels.csv")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classifier_miss_range():
    """Lowest and highest in-scope probability among the classifier's test-set mistakes."""
    path = os.path.join(reports, "classifier_errors.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        values = [float(r["probability_in_scope"]) for r in csv.DictReader(f) if r["real_label"] == "1"]
    return (min(values), max(values)) if values else None


if not os.path.exists(runs_path):
    print("ERROR: no saved answers. Run first:  python scripts/answer_eval.py generate")
    sys.exit(1)

questions = ae.load_questions(os.path.join(eval_dir, "qa_eval.csv"))
runs = ae.load_jsonl(runs_path)
questions = [q for q in questions if q["question_id"] in runs]

# Hand labels are optional here. We use them only if the WHOLE sheet is filled in.
humans = None
if os.path.exists(sheet_path):
    try:
        humans = ae.read_human_labels(sheet_path)
    except ValueError:
        humans = None

answerable_sims = [r["top_similarity"] for q in questions for r in [runs[q["question_id"]]]
                   if q["answerable"] == "yes" and r["top_similarity"] is not None]
context = {"min_similarity": config["rag"]["min_similarity"],
           "lowest_answerable_similarity": min(answerable_sims) if answerable_sims else None,
           "classifier_miss_range": classifier_miss_range()}

rows = ea.analyse_all(questions, runs, humans, context)
worst = ea.pick_worst(rows, 5)
causes = ea.cause_counts(rows, len(questions))
write_csv(os.path.join(reports, "error_analysis.csv"), worst)
write_csv(os.path.join(reports, "error_causes.csv"), causes)

print("=" * 100)
print("ERROR ANALYSIS  (" + str(len(questions)) + " questions, " + str(len(rows)) + " with at least one problem)")
print("Mode:", "automatic checks + your hand labels" if humans else "AUTOMATIC CHECKS ONLY (hand labels are not complete)")
newer = ea.stale_files(root, runs_path)
if newer:
    print("\nWARNING: these files changed AFTER the saved answers were generated:")
    for name in newer:
        print("   -", name)
    print("The answers may not match your current code. Re-run:  python scripts/answer_eval.py generate")
print("=" * 100)

print("\nMAIN CAUSES")
for c in causes:
    print("  " + c["failure_category"].ljust(26) + str(c["questions"]) + " of " + str(c["out_of"]))

print("\nTHE 5 WORST CASES")
for number, w in enumerate(worst, start=1):
    print("\n" + "-" * 100)
    print(str(number) + ". " + w["question_id"] + "  [" + w["failure_category"] + "]  penalty " + str(w["penalty"]))
    print("QUESTION   :", w["question"])
    print("EXPECTED   :", w["expected_answer"])
    print("ACTUAL     :", w["actual_answer"][:300])
    print("CONTEXT    :", w["retrieved_context"][:300])
    print("REASON     :", w["reason_for_failure"])
    print("IMPROVEMENT:", w["proposed_improvement"])

print("\nSaved: reports/error_analysis.csv and reports/error_causes.csv")
print("Read each REASON yourself. They are automatic, so correct them if your own look at the answer differs.")

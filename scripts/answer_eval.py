# answer_eval.py  (Step 10)
# Four commands, run in THIS order:
#   python scripts/answer_eval.py generate   # 1. ask the pipeline all 40 questions (slow, resumable)
#   python scripts/answer_eval.py sheet      # 2. make the sheet you fill in BY HAND
#   python scripts/answer_eval.py judge      # 3. let the LLM judge grade the same answers (slow, resumable)
#   python scripts/answer_eval.py report     # 4. compute all metrics and compare judge with you

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.evaluation import answer_eval as ae

config = load_config()
eval_dir = str(resolve_path(config["paths"]["eval_dir"]))
reports = str(resolve_path(config["paths"]["reports_dir"]))
os.makedirs(reports, exist_ok=True)

qa_path = os.path.join(eval_dir, "qa_eval.csv")
runs_path = os.path.join(reports, "qa_runs.jsonl")
sheet_path = os.path.join(eval_dir, "qa_human_labels.csv")
judge_path = os.path.join(reports, "qa_judge_runs.jsonl")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def need(path, message):
    if not os.path.exists(path):
        print("ERROR: file not found ->", path)
        print(message)
        sys.exit(1)


def all_runs(questions):
    runs = ae.load_jsonl(runs_path)
    missing = [q["question_id"] for q in questions if q["question_id"] not in runs]
    if missing:
        print("ERROR: no answer yet for", len(missing), "questions:", missing[:8], "...")
        print("Run first:  python scripts/answer_eval.py generate")
        sys.exit(1)
    return runs


command = sys.argv[1] if len(sys.argv) > 1 else ""
questions = ae.load_questions(qa_path)

if command == "generate":
    from src.rag.pipeline import build_pipeline
    ae.generate_answers(build_pipeline(), questions, runs_path)
    print("\nDone. Answers saved in", runs_path)

elif command == "sheet":
    runs = all_runs(questions)
    if os.path.exists(sheet_path) and "--force" not in sys.argv:
        print("STOP: the label sheet already exists:", sheet_path)
        print("It may contain your hand labels. Use  --force  only if you really want to overwrite it.")
        sys.exit(1)
    count = ae.make_label_sheet(questions, runs, sheet_path)
    print("Label sheet written:", sheet_path, "(" + str(count) + " answers to label)")
    print("Open it in Excel and fill the 4 human_* columns BEFORE you run the judge.")

elif command == "label":
    need(sheet_path, "Run first:  python scripts/answer_eval.py sheet")
    ae.label_interactively(sheet_path)

elif command == "judge":
    from src.evaluation.llm_judge import LLMJudge
    from src.rag.generator import OllamaGenerator
    runs = all_runs(questions)
    settings = config["judge"]
    print("Judge model:", settings["model"])
    judge = LLMJudge(OllamaGenerator(model=settings["model"], max_tokens=settings["max_tokens"]))
    ae.run_judge(judge, questions, runs, judge_path)
    print("\nDone. Judge results saved in", judge_path)

elif command == "report":
    runs = all_runs(questions)
    need(sheet_path, "Run:  python scripts/answer_eval.py sheet   and fill it in first.")
    need(judge_path, "Run:  python scripts/answer_eval.py judge")
    try:
        humans = ae.read_human_labels(sheet_path)
    except ValueError as error:
        print(error)
        sys.exit(1)
    answered_ids = [q["question_id"] for q in questions if not runs[q["question_id"]]["refused"]]
    missing = [qid for qid in answered_ids if qid not in humans]
    if missing:
        print("ERROR: these answered questions have no human label (regenerate the sheet?):", missing)
        sys.exit(1)
    judge = ae.load_jsonl(judge_path)

    human_rows = ae.compute_summary(questions, runs, humans, "human")
    judge_rows = ae.compute_summary(questions, runs, judge, "judge")
    agreement, disagreements = ae.compute_agreement(humans, judge)
    worst = ae.find_worst(questions, runs, humans, 5)

    write_csv(os.path.join(reports, "answer_quality_summary.csv"), human_rows + judge_rows)
    write_csv(os.path.join(reports, "judge_agreement.csv"), agreement)
    write_csv(os.path.join(reports, "judge_disagreements.csv"), disagreements)
    write_csv(os.path.join(reports, "worst_answers.csv"), worst)

    print("\n" + "=" * 100)
    print("ANSWER QUALITY (human labels = ground truth)")
    print("=" * 100)
    for r in human_rows:
        print(r["metric"].ljust(46), str(r["value"]).ljust(7), "(" + str(r["numerator"]) + " / " + str(r["denominator"]) + ")")
    print("\nSame metrics according to the LLM judge:")
    for r in judge_rows:
        print(r["metric"].ljust(46), str(r["value"]).ljust(7), "(" + str(r["numerator"]) + " / " + str(r["denominator"]) + ")")
    print("\n" + "=" * 100)
    print("JUDGE vs HUMAN")
    print("=" * 100)
    print("criterion".ljust(14), "n".ljust(4), "agree".ljust(8), "kappa".ljust(8), "human".ljust(8), "judge".ljust(8), "judge-human")
    for a in agreement:
        print(a["criterion"].ljust(14), str(a["n_compared"]).ljust(4), str(a["percent_agreement"]).ljust(8),
              str(a["cohen_kappa"]).ljust(8), str(a["human_mean"]).ljust(8), str(a["judge_mean"]).ljust(8),
              a["judge_minus_human"], ("  (" + str(a["judge_unparsed"]) + " unparsed)") if a["judge_unparsed"] else "")
    print("\nDisagreements:", len(disagreements), "(see reports/judge_disagreements.csv)")
    print("\n5 WORST ANSWERS (see reports/worst_answers.csv):")
    for w in worst:
        print("  ", w["question_id"], "| penalty", w["penalty"], "|", w["question"], "|", w["automatic_flags"])
else:
    print("Commands:  generate  ->  sheet  ->  (fill the sheet by hand)  ->  judge  ->  report")
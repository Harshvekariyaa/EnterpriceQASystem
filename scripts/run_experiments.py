# run_experiments.py  (Step 7)
# Run:   python scripts/run_experiments.py
# Runs 2 chunk sizes x 3 retrievers = 6 experiments and saves the results.

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.evaluation import metrics
from src.evaluation.retrieval_eval import evaluate_retriever, load_questions, summarize
from src.retrieval.factory import build_retriever
from src.storage import sqlite_store, vector_store

config = load_config()
chunk_sizes = config["ingestion"]["chunk_sizes_words"]          # [80, 160]
k = config["retrieval"]["top_k"]                                # 5
retriever_names = ["bm25", "dense", "hybrid"]

eval_folder = str(resolve_path(config["paths"]["eval_dir"]))
reports_folder = str(resolve_path(config["paths"]["reports_dir"]))
os.makedirs(reports_folder, exist_ok=True)

conn = sqlite_store.connect(str(resolve_path(config["paths"]["sqlite_path"])))
embeddings = vector_store.get_embeddings()
questions = load_questions(os.path.join(eval_folder, "retrieval_eval.csv"))
print("Test questions:", len(questions), "| k =", k)

summary_rows = []       # one row per experiment
question_rows = []      # one row per (experiment, question)  -> used for error analysis
type_rows = []          # results split by question type
experiment_number = 0

for size in chunk_sizes:
    all_chunks = sqlite_store.get_chunks(conn, size)
    for name in retriever_names:
        experiment_number = experiment_number + 1
        label = "Experiment " + str(experiment_number)
        print("\n" + label + ": chunk size", size, "| retriever", name)

        retriever = build_retriever(name, size, conn, embeddings)
        rows = evaluate_retriever(retriever, questions, all_chunks, k)
        result = summarize(rows)

        summary_rows.append({
            "Configuration": label, "Chunk Size": size, "Retriever": name,
            "Recall@" + str(k): round(result["recall"], 3),
            "MRR": round(result["mrr"], 3),
            "nDCG@" + str(k): round(result["ndcg"], 3),
            "Questions": result["num_questions"],
        })
        for r in rows:
            row = dict(r)
            row["configuration"] = label
            row["chunk_size"] = size
            row["retriever"] = name
            question_rows.append(row)

        for qtype in ["keyword", "paraphrase", "number"]:
            subset = [r for r in rows if r["question_type"] == qtype]
            type_rows.append({
                "Configuration": label, "Chunk Size": size, "Retriever": name,
                "Question Type": qtype, "Questions": len(subset),
                "Recall@" + str(k): round(metrics.mean([r["recall"] for r in subset]), 3),
                "MRR": round(metrics.mean([r["rr"] for r in subset]), 3),
            })

        sqlite_store.save_eval_run(conn, "retrieval_" + label.replace(" ", "_"),
                                   {"chunk_size": size, "retriever": name, "k": k}, result)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_csv(os.path.join(reports_folder, "retrieval_results.csv"), summary_rows)
write_csv(os.path.join(reports_folder, "retrieval_per_question.csv"), question_rows)
write_csv(os.path.join(reports_folder, "retrieval_by_type.csv"), type_rows)

# ---- print the main table ----
print("\n" + "=" * 86)
print("RETRIEVAL RESULTS   (Recall@" + str(k) + ", MRR and nDCG@" + str(k) + " are averages over the questions)")
print("=" * 86)
header = ["Configuration", "Chunk Size", "Retriever", "Recall@" + str(k), "MRR", "nDCG@" + str(k), "Questions"]
print("".join(h.ljust(15) for h in header))
for row in summary_rows:
    print("".join(str(row[h]).ljust(15) for h in header))

best = sorted(summary_rows, key=lambda r: (r["MRR"], r["nDCG@" + str(k)]), reverse=True)[0]
print("\nBest by MRR (tie -> nDCG):", best["Configuration"], "| chunk size", best["Chunk Size"],
      "| retriever", best["Retriever"])
print("Saved in:", reports_folder)
conn.close()
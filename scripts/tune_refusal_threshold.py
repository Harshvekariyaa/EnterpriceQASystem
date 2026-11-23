# tune_refusal_threshold.py  (Step 8)
# Run:   python scripts/tune_refusal_threshold.py
# Finds a good value for `min_similarity` in config.yaml using DATA, not guessing.
# It compares the best-chunk similarity of questions that ARE answerable (our 40 test
# questions) with questions that are NOT answerable (written below).

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.evaluation.retrieval_eval import load_questions
from src.retrieval.factory import build_retriever
from src.storage import sqlite_store, vector_store

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the cricket world cup?",
    "How do I bake a chocolate cake?",
    "How do I reverse a list in Python?",
    "What is the price of Bitcoin today?",
    "Tell me a joke.",
    "What is the weather in Mumbai?",
    "Explain how photosynthesis works.",
    "What is the best smartphone to buy?",
    "How do I invest in mutual funds?",
    "Translate hello into French.",
    "What is machine learning?",
    "Which team won the football match yesterday?",
    # near-misses: they SOUND like company questions, but the documents do not answer them
    "What is the leave policy at Google?",
    "Who is the CEO of NovaTech?",
    "How many employees does NovaTech have?",
    "What is the salary of a software engineer at NovaTech?",
    "What is the dress code at NovaTech?",
    "Does NovaTech offer stock options?",
    "What is the cafeteria menu today?",
]

config = load_config()
chunk_size = config["ingestion"]["default_chunk_size"]
conn = sqlite_store.connect(str(resolve_path(config["paths"]["sqlite_path"])))
embeddings = vector_store.get_embeddings()
dense = build_retriever("dense", chunk_size, conn, embeddings)
conn.close()


def best_similarity(question):
    results = dense.retrieve(question, k=1)
    return results[0]["score"] if results else 0.0


in_scope = []
for q in load_questions(os.path.join(str(resolve_path(config["paths"]["eval_dir"])), "retrieval_eval.csv")):
    in_scope.append((best_similarity(q["question"]), q["question"]))
out_scope = []
for question in OUT_OF_SCOPE:
    out_scope.append((best_similarity(question), question))
in_scope.sort()
out_scope.sort(reverse=True)

print("Chunk size:", chunk_size)
print("\nLOWEST similarity among answerable questions (risk: wrongly refused):")
for score, question in in_scope[:5]:
    print("  ", score, "|", question)
print("\nHIGHEST similarity among unanswerable questions (risk: wrongly answered):")
for score, question in out_scope[:5]:
    print("  ", score, "|", question)

print("\nthreshold | answerable kept | unanswerable refused | accuracy")
rows = []
best = None
threshold = 0.20
while threshold <= 0.80:
    kept = sum(1 for s, q in in_scope if s >= threshold)
    refused = sum(1 for s, q in out_scope if s < threshold)
    accuracy = (kept + refused) / (len(in_scope) + len(out_scope))
    rows.append({"threshold": round(threshold, 2), "answerable_kept": kept,
                 "unanswerable_refused": refused, "accuracy": round(accuracy, 3)})
    print("  ", round(threshold, 2), "  |  ", kept, "/", len(in_scope), "  |  ",
          refused, "/", len(out_scope), "  |  ", round(accuracy, 3))
    if best is None or accuracy > best["accuracy"]:        # tie -> keep the LOWER threshold
        best = rows[-1]
    threshold = threshold + 0.02

reports = str(resolve_path(config["paths"]["reports_dir"]))
os.makedirs(reports, exist_ok=True)
with open(os.path.join(reports, "refusal_threshold.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nBest threshold:", best["threshold"], "(accuracy", best["accuracy"], ")")
print("Put this in config.yaml ->  rag: min_similarity:", best["threshold"])
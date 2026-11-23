# retrieval_eval.py  (Step 7)
# Job: run a retriever on all test questions and score the results.

import csv

from src.evaluation import metrics


def normalize(text):
    """lowercase + single spaces, so 'Rs.  5,000\n' and 'rs. 5,000' compare equal."""
    return " ".join(text.lower().split())


def load_questions(csv_path):
    """Read data/eval/retrieval_eval.csv into a list of dictionaries."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_relevant_chunks(all_chunks, gold_phrase):
    """How many chunks (in the whole knowledge base) contain the gold phrase?"""
    gold = normalize(gold_phrase)
    count = 0
    for chunk in all_chunks:
        if gold in normalize(chunk["text"]):
            count = count + 1
    return count


def evaluate_retriever(retriever, questions, all_chunks, k):
    """Score one retriever. Returns one row (dictionary) per question."""
    rows = []
    for q in questions:
        gold = normalize(q["gold_phrase"])
        total_relevant = count_relevant_chunks(all_chunks, q["gold_phrase"])
        if total_relevant == 0:
            # The gold phrase is not inside ANY chunk (the chunking cut it). Skip and warn.
            print("  WARNING: gold phrase not found in any chunk ->", q["question_id"])
            continue

        results = retriever.retrieve(q["question"], k=k)

        relevances = []                     # 1 if the chunk contains the gold phrase
        for r in results:
            if gold in normalize(r["text"]):
                relevances.append(1)
            else:
                relevances.append(0)

        first_relevant_rank = 0
        for position, rel in enumerate(relevances, start=1):
            if rel == 1:
                first_relevant_rank = position
                break

        rows.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "first_relevant_rank": first_relevant_rank,        # 0 = not found in top k
            "recall": metrics.recall_at_k(relevances, k),
            "rr": metrics.reciprocal_rank(relevances, k),
            "ndcg": metrics.ndcg_at_k(relevances, total_relevant, k),
            "top1_file": results[0]["file_name"] if results else "",
            "top1_section": results[0]["section"] if results else "",
            "expected_file": q["source_file"],
        })
    return rows


def summarize(rows):
    """Average the per-question scores."""
    return {
        "recall": metrics.mean([r["recall"] for r in rows]),
        "mrr": metrics.mean([r["rr"] for r in rows]),
        "ndcg": metrics.mean([r["ndcg"] for r in rows]),
        "num_questions": len(rows),
    }
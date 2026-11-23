# tune_cache_threshold.py  (Step 11)
# Job: choose the semantic-cache threshold by EXPERIMENT, not by guessing.
# Run:  python scripts/tune_cache_threshold.py
# Needs: data/eval/cache_pairs.csv  (question_a, question_b, label: 1 = same meaning, 0 = different)
# Creates: reports/cache_threshold_results.csv and reports/cache_threshold_curve.png

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window needed
import matplotlib.pyplot as plt

from src.cache.semantic_cache import cosine, facts_match
from src.config import load_config, resolve_path


def score(pairs, similarities, threshold, use_guard):
    """Count hits at one threshold. A 'hit' = the cache would reuse the old answer."""
    tp = fp = fn = tn = 0
    for pair, similarity in zip(pairs, similarities):
        hit = similarity >= threshold
        if use_guard and hit:
            hit = facts_match(pair["question_a"], pair["question_b"])
        same = pair["label"] == 1
        if hit and same:
            tp += 1
        elif hit and not same:
            fp += 1                         # WRONG answer would be reused  <- the dangerous one
        elif (not hit) and same:
            fn += 1                         # a missed saving, harmless
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def choose(rows):
    """Best recall among thresholds with ZERO wrong hits. If none exists, the best precision."""
    safe = [r for r in rows if r["fp"] == 0]
    if len(safe) > 0:
        return max(safe, key=lambda r: (r["recall"], -r["threshold"]))
    return max(rows, key=lambda r: (r["precision"], r["recall"]))


def main():
    config = load_config()
    pairs_path = resolve_path(config["paths"]["eval_dir"]) / "cache_pairs.csv"
    with open(pairs_path, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    for pair in pairs:
        pair["label"] = int(pair["label"])

    from src.storage import vector_store
    embeddings = vector_store.get_embeddings()
    vectors_a = embeddings.embed_documents([p["question_a"] for p in pairs])
    vectors_b = embeddings.embed_documents([p["question_b"] for p in pairs])
    similarities = [cosine(a, b) for a, b in zip(vectors_a, vectors_b)]

    same = [s for s, p in zip(similarities, pairs) if p["label"] == 1]
    different = [s for s, p in zip(similarities, pairs) if p["label"] == 0]
    print("Pairs: %d same-meaning, %d different-meaning" % (len(same), len(different)))
    print("Similarity of SAME pairs      : min %.3f  avg %.3f" % (min(same), sum(same) / len(same)))
    print("Similarity of DIFFERENT pairs : max %.3f  avg %.3f" % (max(different), sum(different) / len(different)))

    thresholds = [round(0.70 + 0.01 * i, 2) for i in range(30)]          # 0.70 ... 0.99
    results = {"embedding_only": [], "with_guard": []}
    for name, use_guard in (("embedding_only", False), ("with_guard", True)):
        for t in thresholds:
            row = score(pairs, similarities, t, use_guard)
            row["threshold"] = t
            results[name].append(row)

    out_dir = resolve_path(config["paths"]["reports_dir"])
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "cache_threshold_results.csv", "w", newline="", encoding="utf-8") as f:
        columns = ["mode", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name in results:
            for row in results[name]:
                writer.writerow({"mode": name, **row})

    plt.figure(figsize=(8, 5))
    for name, style in (("embedding_only", "--"), ("with_guard", "-")):
        plt.plot(thresholds, [r["precision"] for r in results[name]], style, color="tab:red",
                 label="precision (" + name + ")")
        plt.plot(thresholds, [r["recall"] for r in results[name]], style, color="tab:blue",
                 label="recall (" + name + ")")
    best = choose(results["with_guard"])
    plt.axvline(best["threshold"], color="green", linestyle=":", label="chosen = %.2f" % best["threshold"])
    plt.xlabel("similarity threshold")
    plt.ylabel("score")
    plt.title("Semantic cache threshold")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cache_threshold_curve.png", dpi=150)

    print("\nthreshold | mode            | wrong hits | precision | recall")
    for name in results:
        for row in results[name]:
            if row["threshold"] in (0.75, 0.80, 0.85, 0.90, 0.95):
                print("   %.2f   | %-15s |     %2d     |   %.2f    |  %.2f"
                      % (row["threshold"], name, row["fp"], row["precision"], row["recall"]))

    print("\nCHOSEN (with guard): threshold %.2f -> precision %.2f, recall %.2f, wrong hits %d"
          % (best["threshold"], best["precision"], best["recall"], best["fp"]))
    print("Put this in config.yaml:  cache: semantic_threshold: %.2f" % best["threshold"])
    print("Saved: reports/cache_threshold_results.csv and reports/cache_threshold_curve.png")


if __name__ == "__main__":
    main()
# train_classifier.py  (Step 9)
# Run:   python scripts/train_classifier.py
# Quick run without the transformer:   python scripts/train_classifier.py --baseline-only
# Also search the transformer settings: python scripts/train_classifier.py --grid-finetune

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")          # draw pictures without opening a window
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from src.classifier import data, train
from src.config import load_config, resolve_path
from src.logger import get_logger

log = get_logger("train_classifier")
config = load_config()
settings = config["classifier"]
seed = config["project"]["random_seed"]
threshold = settings["threshold"]
reports = str(resolve_path(config["paths"]["reports_dir"]))
model_dir = str(resolve_path(settings["model_dir"]))
os.makedirs(reports, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
baseline_only = "--baseline-only" in sys.argv
grid_finetune = "--grid-finetune" in sys.argv

# ---------------------------------------------------------------- 1. data
dataset_path = os.path.join(str(resolve_path(config["paths"]["eval_dir"])), "classifier_dataset.csv")
if not os.path.exists(dataset_path):
    print("ERROR: file not found ->", dataset_path)
    print("Run this first:  python scripts/build_classifier_data.py")
    sys.exit(1)
rows = data.load_csv(dataset_path)
part = {}
for split in ["train", "validation", "test"]:
    part[split] = [r for r in rows if r["split"] == split]
texts = {s: [r["question"] for r in part[s]] for s in part}
labels = {s: [r["label"] for r in part[s]] for s in part}
print("train:", len(texts["train"]), "| validation:", len(texts["validation"]), "| test:", len(texts["test"]))

train.set_seed(seed)
embedding_name = config["embedding"]["model_name"]
results = {}        # model name -> {"validation": ..., "test": ..., "test_probs": ...}

# ---------------------------------------------------------------- 2. Model A: baseline + GridSearchCV
print("\nModel A: embeddings + Logistic Regression, tuned with GridSearchCV")
vectors = {s: train.embed_texts(texts[s], embedding_name) for s in part}
groups = [r["group_id"] for r in part["train"]]
grid = settings["baseline_grid"]
n_settings = 1
for values in grid.values():
    n_settings = n_settings * len(values)
print("  trying", n_settings, "settings x", settings["cv_folds"], "folds =", n_settings * settings["cv_folds"], "fits")

baseline, best_params, best_cv_score, grid_rows = train.tune_baseline(
    vectors["train"], labels["train"], groups, grid, settings["cv_folds"], seed)
print("  best settings:", best_params, "| cross-validated macro-F1:", round(best_cv_score, 3))
for row in grid_rows:
    if row["params"] == str({"C": 1.0, "class_weight": None}) or row["params"] == str({"C": 1, "class_weight": None}):
        print("  (the untuned default C=1 had cross-validated macro-F1:", row["cv_macro_f1_mean"], ")")

with open(os.path.join(reports, "classifier_gridsearch.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
    writer.writeheader()
    writer.writerows(grid_rows)

probs = {s: train.baseline_probabilities(baseline, vectors[s]) for s in ["validation", "test"]}
results["baseline_logreg"] = {
    "validation": train.evaluate(labels["validation"], probs["validation"], threshold),
    "test": train.evaluate(labels["test"], probs["test"], threshold),
    "test_probs": probs["test"],
}

# ---------------------------------------------------------------- 3. Model B: fine-tuned
training_log = []
finetuned = None
finetune_grid_rows = []
finetune_settings = settings["finetune"]
if not baseline_only:
    print("\nModel B: fine-tuned transformer")
    # GridSearchCV is too slow for a transformer (5 folds x every setting), so with
    # --grid-finetune we try each setting ONCE and score it on the VALIDATION set instead.
    if grid_finetune:
        candidates = list(ParameterGrid(settings["finetune_grid"]))
    else:
        candidates = [{}]                     # just the settings in config.yaml
    best_trial = None
    for params in candidates:
        trial = dict(settings["finetune"])
        trial.update(params)
        print(" ", "trying" if grid_finetune else "training", params if params else trial)
        model, tokenizer, log_rows = train.train_finetuned(
            texts["train"], labels["train"], texts["validation"], labels["validation"], trial, seed)
        val_probs = train.finetuned_probabilities(model, tokenizer, texts["validation"], trial["max_length"])
        val_pred = [1 if p >= threshold else 0 for p in val_probs]
        score = f1_score(labels["validation"], val_pred, average="macro", zero_division=0)
        finetune_grid_rows.append({"settings": str(params if params else "config.yaml"),
                                   "val_macro_f1": round(score, 4)})
        if best_trial is None or score > best_trial["score"]:
            best_trial = {"score": score, "model": model, "tokenizer": tokenizer,
                          "log": log_rows, "settings": trial}

    model, tokenizer = best_trial["model"], best_trial["tokenizer"]
    training_log = best_trial["log"]
    finetune_settings = best_trial["settings"]
    probs = {s: train.finetuned_probabilities(model, tokenizer, texts[s], finetune_settings["max_length"])
             for s in ["validation", "test"]}
    results["finetuned_minilm"] = {
        "validation": train.evaluate(labels["validation"], probs["validation"], threshold),
        "test": train.evaluate(labels["test"], probs["test"], threshold),
        "test_probs": probs["test"],
    }
    finetuned = (model, tokenizer)
    if grid_finetune:
        with open(os.path.join(reports, "classifier_finetune_grid.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["settings", "val_macro_f1"])
            writer.writeheader()
            writer.writerows(finetune_grid_rows)

# ---------------------------------------------------------------- 4. choose the model using VALIDATION only
chosen = "baseline_logreg"
for name in results:
    if results[name]["validation"]["macro_f1"] > results[chosen]["validation"]["macro_f1"]:
        chosen = name
print("\nChosen model (best validation macro-F1):", chosen)

# ---------------------------------------------------------------- 5. save the chosen model
meta = {"model_type": "logreg" if chosen == "baseline_logreg" else "finetuned",
        "threshold": threshold, "embedding_model": embedding_name,
        "max_length": finetune_settings["max_length"], "chosen_name": chosen}
if chosen == "baseline_logreg":
    import joblib
    joblib.dump(baseline, os.path.join(model_dir, "logreg.joblib"))
else:
    finetuned[0].save_pretrained(model_dir)
    finetuned[1].save_pretrained(model_dir)
with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------- 6. reports
table = []
for name in results:
    for split in ["validation", "test"]:
        r = results[name][split]
        for cls in ["in_scope", "out_of_scope"]:
            table.append({"model": name, "split": split, "class": cls,
                          "precision": round(r[cls]["precision"], 3), "recall": round(r[cls]["recall"], 3),
                          "f1": round(r[cls]["f1"], 3), "support": r[cls]["support"],
                          "accuracy": round(r["accuracy"], 3), "macro_f1": round(r["macro_f1"], 3)})
with open(os.path.join(reports, "classifier_results.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)

if training_log:
    with open(os.path.join(reports, "classifier_training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(training_log[0].keys()))
        writer.writeheader()
        writer.writerows(training_log)

# confusion matrix pictures (test set)
for name in results:
    matrix = results[name]["test"]["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(4, 3.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1]); axis.set_xticklabels(["predicted\nOUT", "predicted\nIN"])
    axis.set_yticks([0, 1]); axis.set_yticklabels(["real OUT", "real IN"])
    for i in range(2):
        for j in range(2):
            axis.text(j, i, matrix[i][j], ha="center", va="center", fontsize=16)
    axis.set_title(name + " (test)")
    figure.tight_layout()
    figure.savefig(os.path.join(reports, "confusion_matrix_" + name + ".png"), dpi=150)
    plt.close(figure)

# test mistakes of the chosen model (for error analysis)
errors = []
predictions = results[chosen]["test"]["predictions"]
for row, prediction, prob in zip(part["test"], predictions, results[chosen]["test_probs"]):
    if prediction != row["label"]:
        errors.append({"question": row["question"], "real_label": row["label"], "predicted": prediction,
                       "probability_in_scope": round(prob, 3), "category": row["category"]})
with open(os.path.join(reports, "classifier_errors.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "real_label", "predicted", "probability_in_scope", "category"])
    writer.writeheader()
    writer.writerows(errors)

# reproducible training configuration
try:
    import sklearn
    versions = {"scikit-learn": sklearn.__version__}
    import torch, transformers
    versions.update({"torch": torch.__version__, "transformers": transformers.__version__})
except ImportError:
    pass
with open(os.path.join(reports, "classifier_training_config.json"), "w", encoding="utf-8") as f:
    json.dump({"seed": seed, "threshold": threshold, "split_sizes": {s: len(texts[s]) for s in texts},
               "baseline": {"model": "LogisticRegression", "embedding": embedding_name,
                            "tuning": "GridSearchCV (StratifiedGroupKFold)", "grid": grid,
                            "cv_folds": settings["cv_folds"], "best_params": {k: v for k, v in best_params.items()},
                            "cv_best_macro_f1": round(best_cv_score, 4)},
               "finetune": None if baseline_only else finetune_settings,
               "finetune_grid": settings["finetune_grid"] if grid_finetune else None,
               "chosen": chosen, "library_versions": versions}, f, indent=2)

# ---------------------------------------------------------------- 7. print the summary
print("\n" + "=" * 78)
print("TEST SET RESULTS  (positive class = IN-SCOPE)")
print("=" * 78)
for name in results:
    r = results[name]["test"]
    print("\n" + name + ("   <-- chosen" if name == chosen else ""))
    print("  in-scope     precision", round(r["in_scope"]["precision"], 3), "| recall", round(r["in_scope"]["recall"], 3),
          "| F1", round(r["in_scope"]["f1"], 3), "| n =", r["in_scope"]["support"])
    print("  out-of-scope precision", round(r["out_of_scope"]["precision"], 3), "| recall", round(r["out_of_scope"]["recall"], 3),
          "| F1", round(r["out_of_scope"]["f1"], 3), "| n =", r["out_of_scope"]["support"])
    print("  accuracy", round(r["accuracy"], 3), "| macro-F1", round(r["macro_f1"], 3))
    m = r["confusion_matrix"]
    print("  confusion matrix:        predicted OUT   predicted IN")
    print("                real OUT   ", str(m[0][0]).ljust(13), m[0][1], "   <- these are wrongly ACCEPTED")
    print("                real IN    ", str(m[1][0]).ljust(13), m[1][1], "   <- the first number is wrongly REFUSED")
print("\nMistakes of the chosen model on the test set:", len(errors))
for e in errors:
    print("  -", e["question"], "| real", e["real_label"], "| p(in-scope)", e["probability_in_scope"], "|", e["category"])
print("\nSaved model in:", model_dir, "| reports in:", reports)
log.info("Classifier training finished. chosen=%s", chosen)
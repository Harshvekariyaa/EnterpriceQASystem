# build_classifier_data.py  (Step 9)
# Run:   python scripts/build_classifier_data.py
# Creates data/eval/classifier_dataset.csv with a train / validation / test split.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import data
from src.config import load_config, resolve_path
from src.evaluation.retrieval_eval import load_questions

config = load_config()
seed = config["project"]["random_seed"]
eval_folder = str(resolve_path(config["paths"]["eval_dir"]))

rows = data.assign_splits(data.build_rows(), seed)

# the 40 retrieval-test questions must NOT be in the classifier data
retrieval_questions = [q["question"] for q in load_questions(os.path.join(eval_folder, "retrieval_eval.csv"))]
data.check_dataset(rows, retrieval_questions)

data.save_csv(rows, os.path.join(eval_folder, "classifier_dataset.csv"))

print("Total questions:", len(rows), "| groups:", len({r["group_id"] for r in rows}))
print("\nsplit        in-scope  out-of-scope   (near-miss inside out-of-scope)")
for split in ["train", "validation", "test"]:
    part = [r for r in rows if r["split"] == split]
    inside = sum(1 for r in part if r["label"] == 1)
    near = sum(1 for r in part if r["category"] == "near_miss")
    print(split.ljust(12), str(inside).ljust(9), str(len(part) - inside).ljust(13), "(", near, ")")
print("\nSaved: data/eval/classifier_dataset.csv   (checks passed: no duplicates, no leakage)")
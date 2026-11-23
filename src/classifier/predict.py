# predict.py  (Step 9 - part 3)
# Job: load the saved classifier and use it inside the RAG pipeline.
# The object can be called like a function:   in_scope, probability = checker("question")

import json
import os

from src.classifier import train
from src.config import load_config, resolve_path


class ScopeClassifier:
    def __init__(self, model_dir=None):
        config = load_config()
        folder = model_dir or str(resolve_path(config["classifier"]["model_dir"]))
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("No trained classifier found in " + folder +
                                    ". Run first:  python scripts/train_classifier.py")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.threshold = self.meta["threshold"]
        self.model_type = self.meta["model_type"]

        if self.model_type == "logreg":
            import joblib
            self.model = joblib.load(os.path.join(folder, "logreg.joblib"))
            self.embedding_name = self.meta["embedding_model"]
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(folder)
            self.model = AutoModelForSequenceClassification.from_pretrained(folder)
            self.max_length = self.meta["max_length"]

    def probability_in_scope(self, question):
        """Probability (0 to 1) that the question can be answered from our documents."""
        if self.model_type == "logreg":
            vectors = train.embed_texts([question], self.embedding_name)
            return train.baseline_probabilities(self.model, vectors)[0]
        return train.finetuned_probabilities(self.model, self.tokenizer, [question], self.max_length)[0]

    def __call__(self, question):
        probability = self.probability_in_scope(question)
        return probability >= self.threshold, probability
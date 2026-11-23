# train.py  (Step 9 - part 2)
# Two models are trained and compared:
#   Model A (baseline)   : sentence embeddings  +  Logistic Regression
#   Model B (fine-tuned) : the small MiniLM transformer + a classification head, trained on our questions

import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

_embedding_model = None


def set_seed(seed):
    """Fix the random numbers so the results can be repeated."""
    random.seed(seed)
    np.random.seed(seed)


def embed_texts(texts, model_name):
    """Turn questions into embeddings (numbers). The model is loaded only once."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model.encode(list(texts), normalize_embeddings=True, batch_size=32)


# ---------------------------------------------------------------- Model A: baseline
def train_baseline(train_vectors, train_labels, seed):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(train_vectors, train_labels)
    return model


def baseline_probabilities(model, vectors):
    """Probability that each question is IN-SCOPE (label 1)."""
    return model.predict_proba(vectors)[:, 1].tolist()

def tune_baseline(train_vectors, train_labels, groups, param_grid, n_folds, seed):
    """GridSearchCV: try EVERY combination in param_grid and keep the best one.
    Each combination is scored with cross-validation on the TRAINING data only:
        the training data is cut into n_folds parts; train on n_folds-1 parts, score on the last part,
        repeat so every part is used for scoring once, then average the scores.
    IMPORTANT: StratifiedGroupKFold keeps the 3 paraphrases of a question in the SAME fold.
    (Normal folds would split them, and the score would be too good = data leakage.)
    Returns (best_model, best_params, best_cv_score, rows_for_a_report)."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=seed),
        param_grid=param_grid,
        scoring="f1_macro",            # the same score we use to choose the final model
        cv=cv,
        refit=True,                    # after the search, retrain the best setting on ALL training data
        return_train_score=True,       # also keep the training score, to spot over-fitting
    )
    search.fit(train_vectors, train_labels, groups=groups)

    results = search.cv_results_
    rows = []
    for i in range(len(results["params"])):
        rows.append({
            "rank": int(results["rank_test_score"][i]),
            "params": str(results["params"][i]),
            "cv_macro_f1_mean": round(float(results["mean_test_score"][i]), 4),
            "cv_macro_f1_std": round(float(results["std_test_score"][i]), 4),
            "train_macro_f1_mean": round(float(results["mean_train_score"][i]), 4),
        })
    rows.sort(key=lambda r: r["rank"])
    return search.best_estimator_, search.best_params_, float(search.best_score_), rows

# ---------------------------------------------------------------- Model B: fine-tuning
def finetuned_probabilities(model, tokenizer, texts, max_length, batch_size=32):
    import torch
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            logits = model(**batch).logits
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
    return probabilities


def train_finetuned(train_texts, train_labels, val_texts, val_labels, settings, seed):
    """Fine-tune the transformer. After every epoch we score the VALIDATION set,
    and we keep the weights of the best epoch (this prevents over-training).
    Returns (model, tokenizer, log)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(settings["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(settings["base_model"], num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])

    log = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = list(range(len(train_texts)))
        random.Random(seed + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            chosen = order[start:start + settings["batch_size"]]
            batch = tokenizer([train_texts[i] for i in chosen], padding=True, truncation=True,
                              max_length=settings["max_length"], return_tensors="pt")
            labels = torch.tensor([train_labels[i] for i in chosen])
            output = model(**batch, labels=labels)       # the model computes the loss for us
            output.loss.backward()                       # find how to improve
            optimizer.step()                             # improve the weights a little
            optimizer.zero_grad()
            total_loss = total_loss + output.loss.item() * len(chosen)

        val_probs = finetuned_probabilities(model, tokenizer, val_texts, settings["max_length"])
        val_pred = [1 if p >= 0.5 else 0 for p in val_probs]
        val_score = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        log.append({"epoch": epoch, "train_loss": round(total_loss / len(order), 4),
                    "val_macro_f1": round(val_score, 4)})
        print("  epoch", epoch, "| train loss", log[-1]["train_loss"], "| validation macro-F1", log[-1]["val_macro_f1"])
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, tokenizer, log


# ---------------------------------------------------------------- evaluation
def evaluate(labels, probabilities, threshold=0.5):
    """Precision, recall, F1 for BOTH classes + accuracy + confusion matrix.
    Confusion matrix layout:   rows = real class, columns = predicted class
                      predicted OUT   predicted IN
        real OUT         TN              FP        (FP = unanswerable question wrongly accepted)
        real IN          FN              TP        (FN = answerable question wrongly refused)
    """
    predictions = [1 if p >= threshold else 0 for p in probabilities]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    accuracy = sum(1 for a, b in zip(labels, predictions) if a == b) / len(labels)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "out_of_scope": {"precision": float(precision[0]), "recall": float(recall[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
        "in_scope": {"precision": float(precision[1]), "recall": float(recall[1]),
                     "f1": float(f1[1]), "support": int(support[1])},
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
    }
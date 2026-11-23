# NovaTech Enterprise Knowledge Q&A (RAG)

A question-answering system over internal company policy PDFs. An employee asks a question in plain English and gets an answer **with citations** (file, section, page). If the documents do not contain the answer, the system **refuses** instead of guessing.

The company (*NovaTech Solutions Pvt. Ltd.*) is fictional, so no LLM already knows these facts. Every correct answer must come from retrieval.

Everything runs **locally**: no API keys, no cloud services. The LLM is served by [Ollama](https://ollama.com).

---

## Contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Setup](#5-setup)
6. [Build the knowledge base (ingestion)](#6-build-the-knowledge-base-ingestion)
7. [Train the scope classifier](#7-train-the-scope-classifier)
8. [Ask questions](#8-ask-questions)
9. [Run the API and the UI](#9-run-the-api-and-the-ui)
10. [Experiments and evaluation](#10-experiments-and-evaluation)
11. [Run the tests](#11-run-the-tests)
12. [Configuration](#12-configuration)
13. [Results summary](#13-results-summary)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)

---

## 1. What it does

| Area | Feature |
|---|---|
| Ingestion | Reads 10 PDFs, extracts text per page with metadata, cleans, chunks, removes duplicate chunks, embeds, stores |
| Idempotent | Running ingestion twice adds nothing (content-hash chunk IDs + file hashes) |
| Retrieval | BM25, dense (embeddings) and hybrid (Reciprocal Rank Fusion) retrievers |
| RAG | Retrieve, build prompt, generate with Ollama, parse citations, refuse when evidence is weak |
| Trained model | In-scope / out-of-scope question classifier (Logistic Regression baseline vs fine-tuned MiniLM) |
| Caching | Exact-match cache and semantic cache with a threshold chosen by experiment, plus safety checks |
| Evaluation | Retrieval (Recall@k, MRR, nDCG@k), classifier (P/R/F1), answer quality (faithfulness, relevance, hallucination, citation correctness) with LLM judge vs human labels |
| Interface | FastAPI backend and Streamlit page |

---

## 2. Architecture

```
PDFs -> read text -> clean -> chunk -> remove duplicates -> embed -> store
                                                                  |
                                          SQLite (documents, chunks, cache)  +  ChromaDB (vectors)

Question -> scope classifier -> exact cache -> semantic cache -> retrieve chunks
         -> similarity check -> LLM (Ollama) -> parse citations
         -> answer + citations   (or "I could not find this in the company documents")
```

**Refusal has three layers:**
1. The trained classifier rejects clearly out-of-scope questions early.
2. A similarity threshold (`rag.min_similarity`) refuses when the best chunk is too weak. The LLM is not called.
3. The prompt tells the LLM to say it does not know if the context lacks the answer.

**Cache lookup order:** exact cache, then semantic cache, then the full RAG pipeline. Only answered, properly cited questions are cached. The `kb_version` is part of the cache key, so changing the PDFs invalidates old entries.

---

## 3. Project structure

```
enterprise-rag/
|-- config.yaml                 # all settings in one place
|-- requirements.txt
|-- pytest.ini
|-- data/
|   |-- pdfs/                   # the 10 source PDFs
|   |-- raw/                    # extracted text + metadata (JSON)
|   |-- processed/              # chunks_80.json, chunks_160.json
|   |-- eval/                   # test sets: retrieval, QA, classifier, cache pairs, human labels
|   `-- db/                     # rag.sqlite + chroma/  (created by ingestion, not in Git)
|-- models/                     # trained classifier (created by training, not in Git)
|-- reports/                    # all experiment outputs (CSV, PNG, JSONL)
|-- docs/                       # requirements and execution plan
|-- scripts/                    # command-line entry points (see sections 6 to 10)
|-- src/
|   |-- ingestion/              # pdf_reader, cleaner, chunker, dedup, pipeline
|   |-- storage/                # sqlite_store, vector_store
|   |-- retrieval/              # bm25, dense, hybrid, factory
|   |-- rag/                    # prompt, generator (Ollama), pipeline
|   |-- classifier/             # data, train, predict
|   |-- cache/                  # exact_cache, semantic_cache, cache_manager
|   |-- evaluation/             # metrics, retrieval_eval, llm_judge, answer_eval, error_analysis
|   |-- api/main.py             # FastAPI
|   `-- ui/app.py               # Streamlit
`-- tests/                      # pytest suite (LLM is mocked)
```

---

## 4. Prerequisites

- **Python 3.10 or newer** (developed on 3.12)
- **Git**
- **Ollama**, installed from <https://ollama.com/download>
- About **5 GB free disk space** (PyTorch, the embedding model and the LLM)
- 8 GB RAM is enough for the default `llama3.2:1b` model

The first run downloads the embedding model `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB), so an internet connection is needed once.

---

## 5. Setup

### 5.1 Clone and create a virtual environment

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd enterprise-rag
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5.2 Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch and is the slowest step.

### 5.3 Install the LLM with Ollama

Make sure the Ollama app is running, then download the model named in `config.yaml`:

```bash
ollama pull llama3.2:1b
```

Only if you want to run the **LLM-judge evaluation** (section 10.3), also pull the judge model:

```bash
ollama pull llama3.2:3b
```

### 5.4 Check that everything is ready

```bash
python scripts/check_setup.py
```

You want to see **ALL CHECKS PASSED**. It checks the libraries, `config.yaml`, the 10 PDFs and that Ollama is reachable with the model downloaded.

> Always run commands **from the project root folder** (the one containing `config.yaml`).

---

## 6. Build the knowledge base (ingestion)

The 10 PDFs are already in `data/pdfs/`. (To regenerate them, run `python scripts/generate_pdfs.py`.)

The databases in `data/db/` are **not stored in Git**, so build them once:

```bash
python scripts/run_extract.py      # PDFs -> data/raw/*.json  (text per page + metadata + file hash)
python scripts/run_chunking.py     # -> data/processed/chunks_80.json and chunks_160.json
python scripts/run_ingest.py       # -> SQLite + ChromaDB
```

**Prove it is idempotent:** run the last command a second time.

```bash
python scripts/run_ingest.py
```

The second run must report **0 new documents and 0 new chunks**.

Useful extras:

```bash
python scripts/run_ingest.py --reset   # delete both databases and rebuild from scratch
python scripts/check_db.py             # look inside the databases and try two searches
python scripts/check_chunk_labels.py   # check that chunk section and page labels are correct
```

> If you change a PDF, bump `project.kb_version` in `config.yaml` (for example `v2` to `v3`), then run the three commands above again. Changed files are re-ingested and old cache entries stop matching.

---

## 7. Train the scope classifier

The trained classifier is **not included in the repository** (the `models/` folder is git-ignored because of its size). Train it once before asking questions:

```bash
python scripts/train_classifier.py
```

Training uses the dataset in `data/eval/classifier_dataset.csv` (70/15/15 train, validation, test split, seed 42) and saves the chosen model in `models/scope_classifier/`. It also writes metrics and confusion matrices to `reports/`.

Options:

```bash
python scripts/train_classifier.py --baseline-only    # quick run, skips the transformer
python scripts/train_classifier.py --grid-finetune    # also searches the transformer settings (slower)
python scripts/build_classifier_data.py               # rebuild the dataset and its split
```

If you prefer to skip the classifier, set `rag.use_classifier: false` in `config.yaml`. The similarity check and the prompt still protect against out-of-scope questions.

---

## 8. Ask questions

### From the terminal

```bash
python scripts/ask.py "How many sick leave days do I get?"
```

Or start an interactive session (press Enter on an empty line to stop):

```bash
python scripts/ask.py
```

The output shows the answer, the sources (file, section, page), whether the question was refused and why, the top similarity score, the response time and the cache status (`exact`, `semantic` or `none`).

**Things to try:**

| Question | Expected behaviour |
|---|---|
| How many days of annual leave are credited each month? | Answer with a citation to the Leave Policy |
| What is the daily limit for meal claims on business duty? | Answer with a citation to the Expense or Travel policy |
| Who won the cricket world cup? | Refusal: not in the company documents |
| Ask the same answered question again | Served from the cache |

### Compare the three retrievers side by side

```bash
python scripts/compare_retrievers.py
python scripts/compare_retrievers.py "your own question here"
```

---

## 9. Run the API and the UI

Use **two terminals**, both with the virtual environment activated and both in the project root.

**Terminal 1: API**
```bash
uvicorn src.api.main:app
```
Interactive API docs: <http://localhost:8000/docs>

**Terminal 2: Streamlit UI**
```bash
streamlit run src/ui/app.py
```
The page opens at <http://localhost:8501>.

If the API runs on a different address, set it before starting the UI:

```bash
# Windows PowerShell
$env:API_URL = "http://localhost:8000"
# macOS / Linux
export API_URL="http://localhost:8000"
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status of the pipeline and of Ollama, plus the `kb_version` |
| POST | `/ask` | Ask a question |

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many sick leave days do I get?"}'
```

Response fields: `answer`, `refused`, `refusal_reason`, `citations` (number, file_name, section, page, snippet), `citation_warnings`, `cache` (`exact`, `semantic` or `none`), `cache_similarity`, `response_time_seconds`.

Error codes: `400` for an empty or too long question (maximum 500 characters), `503` when Ollama is not running or the system is still starting, `500` for anything unexpected.

---

## 10. Experiments and evaluation

All outputs go to `reports/`. Run the scripts in this order if you want to reproduce everything.

### 10.1 Retrieval experiments

```bash
python scripts/run_experiments.py
```

Runs 2 chunk sizes (80 and 160 words) x 3 retrievers (BM25, dense, hybrid) on the 40 questions in `data/eval/retrieval_eval.csv`. Writes `retrieval_results.csv`, `retrieval_by_type.csv` and `retrieval_per_question.csv`. A retrieved chunk counts as relevant if it contains the labelled gold phrase.

### 10.2 Refusal threshold

```bash
python scripts/tune_refusal_threshold.py
```

Compares best-chunk similarity for answerable and unanswerable questions and writes `refusal_threshold.csv`. Use it to choose `rag.min_similarity`.

### 10.3 Answer quality (needs Ollama)

Run these four commands in order:

```bash
python scripts/answer_eval.py generate   # ask the pipeline all 40 questions (slow, resumable)
python scripts/answer_eval.py sheet      # create the sheet for hand labelling
python scripts/answer_eval.py judge      # LLM judge grades the same answers (slow, resumable)
python scripts/answer_eval.py report     # compute metrics, compare judge with human labels
```

The repository already contains the saved answers and human labels (`reports/qa_runs.jsonl`, `data/eval/qa_human_labels.csv`), so you can run just the `report` step to recompute the metrics.

The four metrics and their denominators:

| Metric | Calculation |
|---|---|
| Faithfulness | supported answers / answered questions |
| Answer relevance | relevant answers / answered questions |
| Hallucination rate | answers with at least one unsupported claim / answered questions |
| Citation correctness | correct citations / total citations |

The judge is compared with the human labels using agreement percentage and Cohen's kappa.

### 10.4 Cache threshold

```bash
python scripts/tune_cache_threshold.py
```

Uses `data/eval/cache_pairs.csv` (pairs labelled same meaning or different meaning), tries thresholds from 0.70 to 0.99 and writes `cache_threshold_results.csv` and `cache_threshold_curve.png`. Set the chosen value in `cache.semantic_threshold`.

**Protection against wrong cache hits:** a high threshold chosen from data, a number and entity check (for example "Rs. 5,000" vs "Rs. 25,000" never match), only answered questions are cached, and `kb_version` is part of the key.

### 10.5 Error analysis and one-page report

```bash
python scripts/error_analysis.py   # 5 worst answers, with failure reason and proposed fix
python scripts/make_report.py      # builds reports/one_page_report.pdf
```

---

## 11. Run the tests

```bash
pytest -v
```

The suite has about 86 tests covering the metrics, ingestion (including "running twice adds nothing"), the retrievers, the classifier, the cache (exact hit, semantic hit, no hit when numbers differ, `kb_version` invalidation), the generator and the API end to end.

**The LLM is mocked, so the tests do not need Ollama running.**

To save the output to a file (UTF-8, also works on Windows PowerShell):

```bash
python scripts/run_tests.py
```

This writes `reports/test_results.txt`.

---

## 12. Configuration

Every setting lives in `config.yaml`. The ones you are most likely to change:

| Key | Default | Meaning |
|---|---|---|
| `project.kb_version` | `v2` | Bump when PDFs change; invalidates the cache |
| `ingestion.chunk_sizes_words` | `[80, 160]` | Chunk sizes built and compared |
| `ingestion.chunk_overlap_words` | `15` | Words shared between neighbouring chunks |
| `ingestion.default_chunk_size` | `80` | Chunk size used by the RAG pipeline |
| `embedding.model_name` | `all-MiniLM-L6-v2` | Embedding model (384 dimensions) |
| `retrieval.best_retriever` | `hybrid` | `bm25`, `dense` or `hybrid` |
| `retrieval.top_k` | `5` | Chunks retrieved per question |
| `llm.model` | `llama3.2:1b` | Ollama model that writes answers |
| `llm.temperature` | `0.0` | 0 gives the most stable answers |
| `rag.min_similarity` | `0.25` | Below this the system refuses without calling the LLM |
| `rag.use_classifier` | `true` | Turn the scope classifier on or off |
| `cache.semantic_threshold` | `0.90` | Minimum similarity for a semantic cache hit |
| `classifier.threshold` | `0.5` | A question is in scope if its probability is at least this |
| `judge.model` | `llama3.2:3b` | Model that grades answers in the evaluation |

Using a different LLM, for example a larger one for better answers:

```bash
ollama pull llama3.2:3b
```

Then set `llm.model: "llama3.2:3b"` in `config.yaml`.

---

## 13. Results summary

### Retrieval (40 questions, k = 5)

| Chunk size | Retriever | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| 80 | BM25 | 0.975 | 0.877 | 0.901 |
| 80 | Dense | 1.000 | 0.950 | 0.963 |
| **80** | **Hybrid** | **1.000** | **0.988** | **0.991** |
| 160 | BM25 | 1.000 | 0.885 | 0.915 |
| 160 | Dense | 0.975 | 0.906 | 0.924 |
| 160 | Hybrid | 1.000 | 0.942 | 0.957 |

The winner is **80-word chunks with hybrid retrieval**, which is the default in `config.yaml`.

### Scope classifier (test set, 39 questions)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline: MiniLM embeddings + Logistic Regression | 0.872 | 0.872 |
| **Fine-tuned MiniLM** (chosen) | **0.897** | **0.897** |

Confusion matrices are in `reports/confusion_matrix_*.png`. The settings for the baseline were chosen by GridSearchCV with group-aware folds, and the transformer settings were chosen on the validation set only.

Answer-quality numbers, the judge-versus-human agreement and the five worst cases are in `reports/` after running section 10.3 and 10.5, and are summarised in `reports/one_page_report.pdf`.

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama not reachable` | Start the Ollama app, then run `python scripts/check_setup.py` again |
| `model 'llama3.2:1b' not found` | Run `ollama pull llama3.2:1b` |
| `No trained classifier found in models/scope_classifier` | Run `python scripts/train_classifier.py`, or set `rag.use_classifier: false` |
| Empty answers or "no chunks" | Run the ingestion steps in section 6; the databases are not in Git |
| `ModuleNotFoundError: src` | Run commands from the project root, not from inside `scripts/` |
| First run is very slow | The embedding model is downloading; later runs use the local copy |
| UI shows "The API is not running" | Start `uvicorn src.api.main:app` in a separate terminal |
| Old answers after editing a PDF | Bump `project.kb_version` in `config.yaml` and re-run ingestion |
| Chunk labels look wrong after a code change | `python scripts/run_ingest.py --reset` |
| PowerShell will not activate the venv | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

Logs are written to `logs/app.log`.

---

## 15. Limitations

- **Small knowledge base.** Only 10 short PDFs, so retrieval scores are very high and differences between configurations are small. Hard questions with similar wording across documents were included to reduce this, but the test set is still small (40 questions).
- **Small local LLM.** `llama3.2:1b` can add extra details or cite imperfectly. A larger model would improve answer quality at the cost of speed.
- **Classifier data.** The classifier is trained on a small, self-written dataset (about 260 questions). Near-miss questions that look like HR questions but are not covered by the documents are the hardest cases.
- **Judge bias.** A small LLM judge may favour long or confident-sounding answers. This is why the judge is checked against human labels with Cohen's kappa.
- **Text PDFs only.** Scanned PDFs (images) would need OCR, which is not included.
- **Single language and single user.** English only, no authentication, no document-level access control.

**With one more week:** add a cross-encoder reranker, expand the test sets, add OCR for scanned files, add access control per department and containerise the API, the UI and Ollama with Docker Compose.

---

## Tech stack

Python, pypdf, LangChain (text splitting, Ollama and Chroma wrappers), sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB, SQLite, rank-bm25, scikit-learn, PyTorch and Transformers, Ollama (`llama3.2`), FastAPI, Streamlit, pytest, reportlab.

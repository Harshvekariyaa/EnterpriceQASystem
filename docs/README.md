# NovaTech Enterprise Knowledge Q&A (RAG)

A local question-answering system over 10 company policy PDFs. It retrieves the relevant passages, asks a local LLM to answer **only** from them, shows citations (file, section, page), and refuses when the documents do not contain the answer.

Everything runs locally and free: Python, SQLite, ChromaDB, Ollama.

## Status at a glance

| Area | Result |
|------|--------|
| Retrieval (hybrid, 80-word chunks) | Recall@5 1.000, MRR 0.988, nDCG@5 0.991 (best of 6 experiments) |
| Scope classifier (fine-tuned MiniLM, test set of 39) | macro F1 0.897 (baseline 0.872) |
| Answer quality | **Draft**: faithfulness 6.2%, hallucination 93.8% with the 1B model; saved answers are older than the code; human labels not yet filled in |
| Tests | 86 test functions in `tests/` |

Retrieval is strong; generation with a 1B model is the weak part. See `docs/05_HLD.md`, section 10, and `reports/one_page_report.pdf`.

## Documentation

| File | Content |
|------|---------|
| `docs/01_requirements_and_execution_plan.md` | original plan, risks, time budget |
| `docs/02_requirements_and_assumptions.md` | requirements, assumptions, target vs result |
| `docs/03_technology_selection.md` | why each tool, limits, alternatives |
| `docs/04_database_design.md` | tables, collections, idempotent ingestion, data flow |
| `docs/05_HLD.md` | big picture, request flow, decisions, limitations |
| `docs/06_LLD.md` | modules, classes, functions |
| `docs/07_architecture_diagram.png` (+ `.mmd`) | architecture diagram |
| `docs/08_ai_tools_and_licences.md` | models, licences, AI-tool disclosure |

## Project layout

```
config.yaml            all settings
src/                   ingestion, storage, retrieval, classifier, rag, cache, evaluation, api, ui
scripts/               runnable steps (ingest, experiments, training, evaluation, report, tests)
data/pdfs/             the 10 source PDFs
data/raw, processed/   extracted text and chunks (JSON)
data/eval/             evaluation and training CSVs
data/db/               SQLite + ChromaDB (created by ingestion)
models/                trained classifier (created by training; not in the zip)
reports/               result tables, plots, one_page_report.pdf
tests/                 pytest
```

## Setup

1. Python 3.10+ and a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```
2. Install [Ollama](https://ollama.com) and download the models:
   ```
   ollama pull llama3.2:1b
   ollama pull llama3.2:3b          # only needed for the answer-quality judge
   ```
3. Check the setup (should print ALL CHECKS PASSED):
   ```
   python scripts/check_setup.py
   ```
The first run downloads the embedding model (about 90 MB).

## Run order

All commands run from the project folder.

| Stage | Command |
|-------|---------|
| Build the PDFs (optional, they are included) | `python scripts/generate_pdfs.py` |
| Extract and chunk | `python scripts/run_extract.py` then `python scripts/run_chunking.py` |
| Ingest into SQLite + ChromaDB | `python scripts/run_ingest.py` |
| Show idempotency | run `run_ingest.py` again: 0 new documents, 0 new chunks |
| Rebuild both databases from scratch | `python scripts/run_ingest.py --reset` |
| Compare retrievers on one question | `python scripts/compare_retrievers.py` |
| Retrieval experiments (6 runs) | `python scripts/run_experiments.py` |
| Build classifier data | `python scripts/build_classifier_data.py` |
| **Train the classifier** (creates `models/scope_classifier/`) | `python scripts/train_classifier.py` |
| Tune the similarity gate | `python scripts/tune_refusal_threshold.py` |
| Tune the semantic-cache threshold | `python scripts/tune_cache_threshold.py` |
| Ask in the terminal | `python scripts/ask.py "How many sick leave days do I get?"` |
| Answer-quality evaluation | `python scripts/answer_eval.py generate`, then `sheet`, label by hand, `judge`, `report` |
| Error analysis | `python scripts/error_analysis.py` |
| One-page report | `python scripts/make_report.py` |
| Tests | `python scripts/run_tests.py` (or `pytest -v`) |

**Important:** `models/` is not in the submitted zip. `config.yaml` has `rag.use_classifier: true`, so the pipeline needs `models/scope_classifier/` and will stop with a clear message until you run `python scripts/train_classifier.py`. To run without the classifier, set `use_classifier: false`.

## Run the application

Make sure Ollama is running, then use two terminals:

```
uvicorn src.api.main:app            # API, docs at http://localhost:8000/docs
streamlit run src/ui/app.py         # UI, opens in the browser
```

Example request:

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave are credited each month?"}'
```

The response contains `answer`, `citations`, `refused`, `refusal_reason`, `cache` (`exact`, `semantic`, `none`) and `response_time_seconds`. `GET /health` shows pipeline, knowledge-base version and Ollama status.

## Updating the knowledge base

1. Replace or add PDFs in `data/pdfs/`.
2. Run extract, chunking and `run_ingest.py`. Changed files are detected by hash; stale chunks are removed.
3. Change `project.kb_version` in `config.yaml` (for example `v3`) so old cached answers stop matching.

## Key settings (`config.yaml`)

`ingestion.default_chunk_size` (80), `retrieval.best_retriever` (hybrid), `retrieval.top_k` (5), `rag.max_context_chunks` (4), `rag.min_similarity` (0.25), `rag.use_classifier`, `cache.semantic_threshold` (0.90), `llm.model`, `classifier.threshold` (0.5).

## Known issues and next steps

- `rag.min_similarity` is still the 0.25 placeholder; `reports/refusal_threshold.csv` shows the best accuracy (0.90) near 0.44. Update and re-check on the QA set.
- Re-run `answer_eval.py generate` with the current code (and preferably `llama3.2:3b`), fill `data/eval/qa_human_labels.csv`, then run `answer_eval.py report` for judge-vs-human agreement and Cohen's kappa.
- `reports/cache_threshold_results.csv` and `reports/test_results.txt` are not in the submitted folder; run the two scripts above to produce them.
- Evaluation sets are small and self-written; results apply to these 10 PDFs only.

## Licences

See `docs/08_ai_tools_and_licences.md`. Llama 3.2 models are used under the Llama 3.2 Community Licence.

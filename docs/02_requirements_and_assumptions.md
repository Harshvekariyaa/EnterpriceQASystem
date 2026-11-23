# 02. Requirements and Assumptions

**Project:** NovaTech Enterprise Knowledge Q&A (RAG), Problem Statement 1
**Knowledge base:** 10 self-written PDFs of a fictional company, *NovaTech Solutions Pvt. Ltd.* (`kb_version: v2`)
**Related:** the original plan is in `01_requirements_and_execution_plan.md`. This file records the final requirements, the assumptions that held, and how each requirement ended up.

---

## 1. Functional requirements and where they are met

| ID | Requirement | Where it lives | Status |
|----|-------------|----------------|--------|
| F1 | Ingest PDFs: extract, clean, chunk, de-duplicate, keep metadata, embed, store | `src/ingestion/`, `scripts/run_ingest.py` | Done |
| F2 | Re-running ingestion creates no duplicates | `src/ingestion/pipeline.py` (`sync_documents`, `sync_chunks`) | Done (content-hash IDs + file hash) |
| F3 | RAG: query cleaning, retrieval, context, LLM answer, citations | `src/rag/pipeline.py` | Done |
| F4 | Refuse when evidence is missing | classifier gate, similarity gate, model "unknown" | Done, with known gaps (section 4) |
| F5 | Retrieval experiments: 2 chunk sizes x 3 retrievers, Recall@k, MRR, nDCG | `src/evaluation/`, `scripts/run_experiments.py` | Done (6 experiments) |
| F6 | Trained model with train/val/test, P/R/F1, confusion matrix | `src/classifier/`, `scripts/train_classifier.py` | Done |
| F7 | Answer-quality evaluation with LLM judge vs human labels | `src/evaluation/answer_eval.py`, `llm_judge.py` | **Partly done**: human labels not yet filled in |
| F8 | Exact + semantic cache, threshold chosen by experiment, safe against wrong hits | `src/cache/`, `scripts/tune_cache_threshold.py` | Built and unit-tested; see open items |
| F9 | UI/API: ask, see answer, citations, "not found" | `src/api/main.py`, `src/ui/app.py` | Done |

## 2. Non-functional requirements

| Area | Requirement | How it is met |
|------|-------------|---------------|
| Modularity | Separate modules per job | `ingestion`, `storage`, `retrieval`, `rag`, `classifier`, `cache`, `evaluation`, `api`, `ui` |
| Reproducibility | Same input gives same output | one `config.yaml`, seed 42, content-hash IDs, temperature 0 |
| Reliability | Clear errors and logs | `src/logger.py` in every module; `GenerationError` when Ollama is down; API returns 400/503/500 |
| Testability | Unit + end-to-end tests | `tests/` (8 test files, 86 test functions); LLM is mocked |
| Cost | Free and local | open-source tools only, no API keys |
| Traceability | Every answer maps to file, page, section, chunk ID | `parse_citations()` returns these fields |
| Performance target | ~30 s per answer on laptop CPU, ~1 s cache hit | cache path is one SQLite read; LLM time depends on the machine and was not benchmarked in these docs |

## 3. Assumptions (and whether they held)

1. **Own-created PDFs, no scraping.** Held. 10 PDFs, 13 pages in total, 4 departments (HR, IT, Finance, Legal and Compliance).
2. **Fictional company, so the LLM cannot know the facts.** Held. Correct answers must come from retrieval.
3. **Small corpus (about 3,300 words).** Held. After de-duplication there are **50 chunks at 80 words** and **22 chunks at 160 words**. Small corpus means retrieval scores sit near the ceiling.
4. **Laptop, 8 GB RAM, no GPU.** The configured LLM is `llama3.2:1b` (the plan said 3B; the config was lowered for speed). The judge is `llama3.2:3b`.
5. **English, single-turn questions.** Held. No chat memory.
6. **Rare updates.** `kb_version` in `config.yaml` is bumped when PDFs change; it is part of every cache key.
7. **Labels are written by the candidate.** Held for retrieval (40), classifier (264), QA (40) and cache pairs (40). Human answer labels (32 rows) are still empty.

## 4. Success criteria: target vs result

Targets come from `01_requirements_and_execution_plan.md`, section A5. Results come from the files in `reports/`.

| Area | Target | Result | Met? |
|------|--------|--------|------|
| Ingestion | 0 new rows on second run | Designed and unit-tested (`test_ingestion.py`). Re-run `run_ingest.py` twice to show it live | Yes (by design) |
| Retrieval | Recall@5 >= 0.90, MRR >= 0.75 | Hybrid, 80 words: Recall@5 **1.000**, MRR **0.988**, nDCG@5 **0.991** | Yes |
| Classifier | Test F1 >= 0.90 | Fine-tuned MiniLM: macro F1 **0.897** (in-scope 0.895, out-of-scope 0.900); baseline 0.872 | Just missed |
| Refusal | >= 90% of unanswerable refused | 7 of 8 = **87.5%** (draft run) | Missed |
| Faithfulness | >= 0.85 | **6.2%** (2/32) in the draft run | Missed |
| Hallucination | <= 10% | **93.8%** (30/32) in the draft run | Missed |
| Citation correctness | >= 0.85 | **18.8%** (6/32) by the judge; 68.3% cite the right file, 51.7% cite a chunk with the gold passage (automatic) | Missed |
| Judge vs human | kappa >= 0.6 | **Not computed**: `qa_human_labels.csv` has 0 of 32 rows labelled | Pending |
| Cache | 0 wrong semantic hits on tricky pairs | Guard logic is unit-tested. `reports/cache_threshold_results.csv` is not in the submitted folder | Not shown |
| Tests | All pass | 86 test functions exist; no `reports/test_results.txt` in the folder | Run `python scripts/run_tests.py` |

**How to read the answer-quality row.** The one-page report itself is marked DRAFT: the saved answers in `reports/qa_runs.jsonl` are older than the current code, and 21 of the 32 answers contain only `[n]` markers with no sentence (the 1B model failed to write text). That is a generation failure, not a retrieval failure: retrieval is at the ceiling. The honest statement for the report is "retrieval works; generation with a 1B model does not yet meet the targets." Re-run `answer_eval.py generate` with the current code (ideally with `llama3.2:3b`), then label and judge again.

## 5. Dependencies and constraints

- Ollama must be installed and the model pulled once (`ollama pull llama3.2:1b`, and `llama3.2:3b` for the judge).
- `sentence-transformers/all-MiniLM-L6-v2` downloads on first use (about 90 MB).
- The trained classifier lives in `models/scope_classifier/`. That folder is **not** in the submitted zip (too large). Recreate it with `python scripts/train_classifier.py`.
- 20-hour time limit; P2 items (re-ranker, polished UI) were not built.

## 6. Out of scope (deferred on purpose)

OCR for scanned PDFs, multi-turn memory, user login, multi-language support, GPU tuning, cross-encoder re-ranking.

# 06. Low-Level Design (LLD)

Module-by-module description. Names match the code in `src/`.

## 1. Configuration and logging

| File | Item | Behaviour |
|------|------|-----------|
| `src/config.py` | `load_config()` | reads `config.yaml` once (`lru_cache`); checks required sections `project, paths, ingestion, embedding, retrieval, llm, rag, cache, logging` |
| | `resolve_path(relative)` | turns a path from the config into an absolute path under the project root |
| `src/logger.py` | `get_logger(name)` | shared log format, console + `logs/app.log` |

Key settings: `kb_version: v2`, seed 42, `chunk_sizes_words: [80, 160]`, `chunk_overlap_words: 15`, `default_chunk_size: 80`, `top_k: 5`, `rrf_k: 60`, `best_retriever: hybrid`, LLM `llama3.2:1b` (temperature 0, 300 tokens, 120 s timeout), `max_context_chunks: 4`, `min_similarity: 0.25`, `use_classifier: true`, `semantic_threshold: 0.90`, classifier threshold 0.5, judge `llama3.2:3b`.

## 2. Ingestion (`src/ingestion`)

| Function / class | Input -> output |
|------------------|-----------------|
| `pdf_reader.load_pages(folder)` | folder -> list of page Documents (LangChain `DirectoryLoader` + `PyPDFLoader`) |
| `pdf_reader.group_pages_by_file`, `build_document` | pages -> one document dict: `file_name, doc_id, title, department, version, effective_date, num_pages, pages` |
| `pdf_reader.get_value_after(lines, label)` | reads header-table fields such as `Version` |
| `pdf_reader.read_all_pdfs`, `save_as_json` | writes one JSON per PDF to `data/raw/` |
| `cleaner.py` | fixes whitespace, removes the header table text, joins split words |
| `chunker.make_chunks(document, blocks, chunk_size, overlap)` | `RecursiveCharacterTextSplitter` (separators newline, ". ", space), word-counted, with `add_start_index`; `find_section_and_page` attaches section and page |
| `dedup.py` | normalise text (lowercase, collapse spaces), `chunk_id = sha256`, keep the first occurrence |
| `pipeline.file_hash(path)` | SHA-256 of the PDF bytes |
| `pipeline.sync_documents(conn, documents, pdf_folder, kb_version)` | returns `{new, updated, unchanged}` |
| `pipeline.sync_chunks(conn, size, chunks, store)` | adds new, removes stale, in SQLite and Chroma; returns counts |

Chunk dictionary: `chunk_id, chunk_size, doc_id, file_name, title, section, page, chunk_index, word_count, text`.

## 3. Storage (`src/storage`)

`sqlite_store.py`: `connect` (foreign keys on, `Row` factory), `create_tables`, `get_saved_hash`, `save_document` (upsert), `get_chunk_ids`, `insert_chunks` (`INSERT OR IGNORE`), `delete_chunks`, `count_rows`, `get_chunks(size)`, `save_eval_run`.

`vector_store.py`: `get_embeddings()` (HuggingFaceEmbeddings, normalised), `get_vector_store(size, embeddings)` (collection `chunks_<size>`, cosine), `get_all_ids`, `add_chunks` (batches of 64, Chroma ID = `chunk_id`), `remove_ids`.

Table layouts are in `04_database_design.md`.

## 4. Retrieval (`src/retrieval`)

Common interface: `retrieve(question, k) -> list of result dicts`. A result holds `rank, chunk_id, text, score, file_name, section, page, doc_id, chunk_size, retriever` (`base.make_result`).

| Class | Notes |
|-------|-------|
| `BM25Search(chunks)` | LangChain `BM25Retriever`, own `tokenize` (lowercase letters and digits); score not exposed (`None`) |
| `DenseSearch(vector_store)` | `similarity_search_with_score`; `score = 1 - cosine distance` |
| `HybridSearch(bm25, dense, rrf_k=60, candidates=20)` | RRF: `1/(rrf_k + rank)` summed over both lists; returns top k |
| `factory.build_retriever(name, size, conn, embeddings)` | `"bm25"`, `"dense"`, `"hybrid"`; raises `ValueError` otherwise |

## 5. RAG (`src/rag`)

`prompt.py`: `NOT_FOUND_MESSAGE = "I could not find this in the company documents."` (also used to detect refusals), `SYSTEM_PROMPT` (four rules: only the context, cite after every fact, exact refusal sentence, one to three sentences), `build_context(chunks)` (numbered passages with Source / Section / Page), `build_messages`.

`generator.py`: `OllamaGenerator(model, max_tokens)` wraps `ChatOllama`; `generate(messages)` returns text or raises `GenerationError`.

`pipeline.py`:

| Item | Behaviour |
|------|-----------|
| `clean_question` | collapse spaces; `ValueError` if empty or over 500 characters |
| `clean_answer` | collapses repeated identical markers (`[1] [1]` -> `[1]`) |
| `parse_citations(answer, chunks)` | returns `(citations, invalid_numbers)`; citation = `number, file_name, section, page, doc_id, chunk_id, snippet (160 chars)` |
| `RAGPipeline.ask(question)` | classifier gate -> similarity gate (dense, k=1) -> retrieve -> prompt -> generate -> model-refusal check -> citations and warnings |
| `build_pipeline(scope_checker=None)` | builds retrievers from the config; loads `ScopeClassifier` when `rag.use_classifier` is true |

Result dictionary: `question, answer, refused, refusal_reason (None | out_of_scope_classifier | low_similarity | model_said_unknown), top_similarity, citations, citation_warnings, retrieved, latency_seconds`.

## 6. Classifier (`src/classifier`)

- `data.py`: `build_rows` (264 labelled questions: 132 policy, 90 general, 42 near-miss), `assign_splits` (70/15/15 by `group_id`, seed 42, so paraphrases stay together), `check_dataset`, `save_csv`, `load_csv`.
- `train.py`: `embed_texts`, `train_baseline`, `tune_baseline` (`GridSearchCV` with `StratifiedGroupKFold`, 5 folds, grid `C` x `class_weight`), fine-tuning of MiniLM with a classification head (5 epochs, batch 16, learning rate 5e-5, max length 64), `baseline_probabilities`, `finetuned_probabilities`.
- `predict.ScopeClassifier`: loads `models/scope_classifier/meta.json` and the model files; `__call__(question) -> (in_scope: bool, probability)`; raises `FileNotFoundError` with instructions if the folder is missing.

Split sizes: train 186, validation 39, test 39.

## 7. Cache (`src/cache`)

| Item | Behaviour |
|------|-----------|
| `exact_cache.normalize_question` | lowercase, `5,000 -> 5000`, punctuation to spaces, collapse spaces |
| `make_key(question, kb_version)` | `sha256(kb_version + "|" + normalised)` |
| `ExactCache.get/put/clear_old_versions` | SQLite `cache` table; `put` stores the embedding too |
| `semantic_cache.extract_numbers`, `extract_terms`, `facts_match` | numbers must be identical; in each `TERM_GROUPS` group (leave type, stage, direction, scope, role, family, work mode, expense, period) both questions may not name different words |
| `SemanticCache.get(question, embedding)` | best cosine >= threshold among current-`kb_version` rows that also pass `facts_match` |
| `cache_manager.is_cacheable(result)` | not refused, at least one citation, no warnings |
| `CachedRAG.ask` | exact -> semantic -> pipeline; adds `cache` and `cache_similarity` to the result |

## 8. API and UI

`src/api/main.py`: pydantic models `AskRequest` (1 to 500 characters), `Citation`, `AskResponse`; `lifespan` loads `CachedRAG(build_pipeline())` once; `GET /health` (status ok or degraded, `pipeline_loaded`, `kb_version`, Ollama reachability and model availability); `POST /ask` (400 bad input, 503 Ollama down or still starting, 500 unexpected).

`src/ui/app.py`: Streamlit; sidebar status from `/health`; three example questions; shows answer, a cache badge, citations (file, section, page, snippet), and a clear message for refusals. `API_URL` environment variable overrides `http://localhost:8000`.

## 9. Evaluation (`src/evaluation`)

- `metrics.py`: `recall_at_k`, `reciprocal_rank`, `dcg_at_k`, `ndcg_at_k`, `mean`, `percent_agreement`, `cohen_kappa` (hand-written, unit-tested).
- `retrieval_eval.py`: `evaluate_retriever`; a chunk is relevant if it contains the gold phrase (`count_relevant_chunks`), so the same test set works for both chunk sizes.
- `llm_judge.py`: `LLMJudge` asks four one-word questions (faithfulness YES/PARTIAL/NO, relevance, correctness, citation).
- `answer_eval.py`: `generate_answers` (resumable JSONL), `make_label_sheet`, interactive labelling, metric computation, judge-vs-human agreement.
- `error_analysis.py`: `diagnose` assigns a failure category (`empty_answer, false_refusal, answered_unanswerable, bad_citations, invalid_citation_number, ...`); `pick_worst` selects the 5 worst; `stale_files` warns when saved answers are older than the code.

## 10. Scripts

| Script | Purpose |
|--------|---------|
| `check_setup.py` | environment check |
| `generate_pdfs.py` | builds the 10 PDFs |
| `run_extract.py`, `run_chunking.py`, `run_ingest.py` | ingestion steps (run ingest twice to show idempotency) |
| `check_db.py`, `check_chunk_labels.py` | inspect the database and gold phrases |
| `compare_retrievers.py` | one question, three retrievers side by side |
| `run_experiments.py` | the 6 retrieval experiments |
| `build_classifier_data.py`, `train_classifier.py` | dataset and training |
| `tune_refusal_threshold.py`, `tune_cache_threshold.py` | threshold sweeps |
| `ask.py` | terminal Q&A |
| `answer_eval.py` | `generate`, `sheet`, `judge`, `report` |
| `error_analysis.py`, `make_report.py` | error cases and the one-page PDF |
| `run_tests.py` | runs pytest and saves `reports/test_results.txt` |

## 11. Tests (`tests/`)

`test_metrics`, `test_ingestion` (chunk sizes, overlap, duplicates, re-run adds nothing), `test_retrieval` (tiny corpus), `test_classifier` (mocked model), `test_cache` (exact hit, semantic hit, different numbers, `kb_version` invalidation), `test_generator` (Ollama mocked), `test_error_analysis`, `test_e2e` (answer with citations; out-of-scope refusal). Shared fixtures in `helpers.py`. 86 test functions in total.

## 12. Error handling summary

| Situation | Handling |
|-----------|----------|
| Empty or too long question | `ValueError` -> API 400 |
| Ollama not running or model missing | `GenerationError` -> API 503, `/health` shows `degraded` |
| Classifier folder missing | `FileNotFoundError` with the training command |
| Citation number not in the context | removed from the citation list, warning added, answer not cached |
| Unknown retriever name | `ValueError` |

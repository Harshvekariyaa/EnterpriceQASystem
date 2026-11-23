# 05. High-Level Design (HLD)

## 1. Purpose

Employees of the fictional company NovaTech ask questions in plain English. The system answers **only** from 10 company policy PDFs, shows which file, section and page the answer came from, and says "I could not find this in the company documents." when the documents do not contain the answer.

## 2. System context

```
Employee --> Streamlit UI --> FastAPI --> CachedRAG --> RAGPipeline --> Ollama (local LLM)
                                              |               |
                                              v               v
                                         SQLite (cache)  SQLite (chunks) + ChromaDB (vectors)
Admin --> scripts/run_ingest.py --> PDFs -> SQLite + ChromaDB
```

External dependencies: Ollama server on `localhost:11434` and the Hugging Face model files (downloaded once). Nothing is sent to a cloud service.

## 3. Main components

| # | Component | Package | Responsibility |
|---|-----------|---------|----------------|
| 1 | Ingestion | `src/ingestion` | read PDFs, clean, chunk, hash, de-duplicate |
| 2 | Storage | `src/storage` | SQLite tables and ChromaDB collections; idempotent sync |
| 3 | Retrieval | `src/retrieval` | BM25, dense and hybrid (RRF) behind one `retrieve(question, k)` interface |
| 4 | Scope classifier | `src/classifier` | trained model: is the question about our documents? |
| 5 | RAG pipeline | `src/rag` | gates, retrieval, prompt, LLM call, citation parsing, refusal |
| 6 | Cache | `src/cache` | exact-match and semantic cache with safety checks |
| 7 | Evaluation | `src/evaluation`, `scripts/` | retrieval metrics, answer metrics, LLM judge, error analysis, report |
| 8 | API | `src/api` | `GET /health`, `POST /ask` |
| 9 | UI | `src/ui` | Streamlit page that calls the API |
| 10 | Config + logging | `config.yaml`, `src/config.py`, `src/logger.py` | one settings file, one log format |

## 4. Request flow (`POST /ask`)

1. **Clean** the question (trim; reject empty or longer than 500 characters).
2. **Exact cache** (normalised text + `kb_version`, hashed). Hit: return.
3. **Semantic cache** (embedding cosine >= `cache.semantic_threshold`, plus number and key-term check). Hit: return.
4. **Refusal layer 1, classifier:** probability in-scope below 0.5 gives an early refusal. The LLM is not called.
5. **Refusal layer 2, similarity gate:** best dense similarity below `rag.min_similarity` gives a refusal.
6. **Retrieval:** hybrid (BM25 + dense, RRF), top 5, the first 4 go into the prompt.
7. **Generation:** strict prompt ("use only the numbered passages, cite like [1], otherwise reply with the not-found sentence"); temperature 0.
8. **Refusal layer 3:** if the model replies with the not-found sentence, the result is a refusal.
9. **Citation parsing:** `[n]` markers are mapped to file, section, page, chunk ID, snippet. Invalid numbers and missing citations become warnings.
10. **Cache write** only if the answer is not a refusal, has at least one citation and no warnings.
11. **Response:** answer, citations, `refused`, `refusal_reason`, `cache` (`exact` / `semantic` / `none`), response time.

## 5. Offline flow (ingestion)

PDF -> text per page + metadata (title, doc ID, version, effective date, file hash) -> clean -> chunk at 80 and 160 words (overlap 15) -> `chunk_id = sha256(normalised text)` -> skip duplicates -> embed only new chunks -> SQLite + ChromaDB. Re-running changes nothing.

## 6. Key design decisions

| Decision | Reason |
|----------|--------|
| Content-hash chunk IDs and file hashes | idempotent re-runs without clearing the database |
| Hybrid retrieval as default | best MRR and nDCG in the experiments; handles both exact numbers and paraphrases |
| Three refusal layers (classifier, similarity, model) | each catches a different failure; the LLM is skipped when refusal is obvious |
| Common retriever interface | swap or mock retrievers in tests and experiments |
| Cache key includes `kb_version` | updating the PDFs cannot return stale answers |
| Only cited, warning-free answers are cached | a wrong answer is never repeated |
| UI talks only to the API | no RAG logic in the UI |
| One `config.yaml` | all knobs (chunk size, retriever, thresholds, models) in one place |

## 7. Results that shaped the design

| Finding | Source | Effect |
|---------|--------|--------|
| Hybrid, 80 words: Recall@5 1.000, MRR 0.988, nDCG@5 0.991 (best of 6) | `reports/retrieval_results.csv` | `best_retriever: hybrid`, `default_chunk_size: 80` |
| Fine-tuned MiniLM test macro F1 0.897 vs baseline 0.872 | `reports/classifier_results.csv` | classifier gate uses the fine-tuned model |
| Similarity sweep: accuracy peaks at 0.44 (0.90); config still uses 0.25 | `reports/refusal_threshold.csv` | open item: update `rag.min_similarity` |

## 8. Non-functional design

- **Reliability:** every module logs through `get_logger`; Ollama failures raise `GenerationError` and the API answers 503; bad input answers 400.
- **Security and privacy:** local models, no external calls, no keys in the repo (`.env`, `models/`, `data/db/` are git-ignored). The API has no authentication (see limitations).
- **Scalability:** designed for one machine and a small corpus. BM25 is rebuilt in memory; SQLite allows one writer.
- **Observability:** `/health` reports pipeline state, `kb_version`, Ollama reachability and model availability; logs go to `logs/app.log`.

## 9. Limitations

- Only 10 short PDFs; retrieval scores are near the ceiling, so MRR is the more informative number.
- English, single-turn, text-based PDFs only.
- Small generator (1B parameters): the draft evaluation shows many answers with citation markers but no sentence.
- Small self-written evaluation sets; one person labels; the judge is a small local model.
- No authentication, no rate limiting, no Docker deployment.

## 10. If there were one more week

1. Larger generator (`llama3.2:3b` or an 8B model) and an empty-answer retry guard.
2. Grounding check: every number in the answer must appear in the cited chunk.
3. Classifier gate: refuse early only when confident (for example probability below 0.3); let retrieval decide in the 0.3 to 0.5 grey zone.
4. Update `rag.min_similarity` from the sweep and confirm on the QA set.
5. Cross-encoder re-ranker; bigger, more varied corpus.
6. Finish human labels and report judge-vs-human kappa.

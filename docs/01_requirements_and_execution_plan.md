# Requirements Analysis and Execution Plan

**Project:** Enterprise Knowledge Intelligence - RAG-Based Knowledge Q&A System
**Domain:** Internal HR / IT / Finance / Legal policies of a fictional company, *NovaTech Solutions Pvt. Ltd.*
**Problem Statement chosen:** Problem Statement 1
**Time budget:** 20 working hours (2 days)

---

## A. Requirement Analysis

### A1. Core functional requirements (from the assessment PDF)

| ID | Requirement | PDF section |
|----|-------------|-------------|
| F1 | Ingestion pipeline: collect PDFs, extract text, clean, chunk, de-duplicate, keep metadata, embed, store in vector DB | 2A |
| F2 | Re-running ingestion must NOT create duplicate documents or chunks (idempotent) | 2A |
| F3 | RAG: query processing, retrieval, context building, LLM answer, source citations | 2B |
| F4 | Refuse to answer when evidence is not in the knowledge base | 2B |
| F5 | Retrieval experiments: >= 2 chunk sizes x >= 2 retrievers; metrics Recall@k, MRR, nDCG | 2C |
| F6 | Train one model (we choose: in-scope / out-of-scope classifier) with train/val/test split, Precision, Recall, F1, confusion matrix | 2D |
| F7 | Answer-quality evaluation on 30-50 labelled triples: faithfulness, relevance, hallucination rate, citation correctness; LLM judge vs human labels | 2E |
| F8 | Exact-match cache and semantic cache with a threshold chosen from experiments; protection against wrong cache hits | 2F |
| F9 | Simple UI / API: ask, see answer, see citations, see "not found" message | 2G |

### A2. Non-functional requirements

- **Modularity:** separate Python modules for ingestion, storage, retrieval, RAG, cache, evaluation.
- **Reproducibility:** fixed random seeds, one `config.yaml`, pinned `requirements.txt`.
- **Reliability:** exception handling and logging in every module.
- **Testability:** pytest unit tests + one end-to-end test.
- **Local and free:** only open-source tools; no paid API keys.
- **Traceability:** every answer can be traced back to file name, page number and chunk ID.
- **Performance (target):** answer within ~30 s on a laptop CPU; cache hit within ~1 s.

### A3. Assumptions

1. The knowledge base is a set of **10 own-created PDF files** (text-based, English). No scraping is done, as agreed.
2. The company is **fictional**, so no LLM knows these facts. Correct answers must come from retrieval.
3. The corpus is small (about 3,300 words), so chunk sizes are measured in **words** and are smaller than in large projects (see Section C).
4. A laptop with 8 GB+ RAM and no GPU. LLM = a 3B-parameter model through Ollama.
5. Questions are in English and are single-turn (no chat memory).
6. The knowledge base is updated rarely, so a **knowledge-base version** tag is enough to invalidate the cache.
7. Evaluation questions and labels are written by me (the candidate). Human labels are treated as ground truth.

### A4. Dependencies and constraints

- Ollama must be installed and the model downloaded (needs internet once).
- `sentence-transformers` model download (needs internet once, ~90 MB).
- 20-hour limit forces a small scope; limited LLM speed on CPU.
- Small corpus means retrieval scores may look "too good". This will be stated honestly in the report.

### A5. Measurable success criteria

| Area | Target |
|------|--------|
| Ingestion | Running twice gives **0 new** documents/chunks the second time |
| Retrieval | Best config: Recall@5 >= 0.90, MRR >= 0.75 |
| Classifier | Test F1 >= 0.90 on in/out-of-scope |
| Refusal | >= 90% of out-of-scope test questions refused |
| Answer quality | Faithfulness >= 0.85, hallucination rate <= 10%, citation correctness >= 0.85 |
| Judge vs human | Cohen's kappa reported (target >= 0.6) |
| Cache | 0 wrong semantic-cache hits on the "tricky pairs" test set |
| Tests | All unit tests pass; one end-to-end test passes |

If a target is missed we report it honestly and do error analysis. That is also part of the marks.

---

## B. Scope and Prioritisation

| Priority | Feature | Deal-breaker? |
|----------|---------|---------------|
| **P0** | PDF ingestion (pypdf) + cleaning + chunking + metadata | Yes |
| **P0** | Duplicate removal and idempotent re-run | Yes |
| **P0** | Embeddings + ChromaDB + SQLite metadata | Yes |
| **P0** | RAG answer with citations + refusal when no evidence | Yes |
| **P0** | Retrieval experiments (2 chunk sizes x 2+ retrievers, 3 metrics) | Yes |
| **P0** | One trained model with P/R/F1 + confusion matrix | Yes |
| **P0** | 30-50 labelled evaluation set + 4 answer metrics | Yes |
| **P0** | Working API or UI | Yes |
| **P1** | Hybrid retrieval (BM25 + dense, RRF) | No |
| **P1** | LLM-judge vs human agreement | No |
| **P1** | Exact + semantic cache | No (but asked in PDF, so we do it) |
| **P1** | Unit tests + E2E test | No (asked in PDF, so we do it) |
| **P1** | Error analysis (5 worst answers) | No |
| **P2** | Dockerfile / deployment | No |
| **P2** | Cross-encoder re-ranker | No |
| **P2** | Nicer UI, query dashboard | No |

**Deferred on purpose:** OCR for scanned PDFs, multi-turn chat memory, user login, multi-language support, GPU optimisation.

---

## C. Technical Planning

### C1. Architecture (summary)

```
PDFs -> pypdf extract -> clean -> chunk -> dedup (hash) -> metadata
     -> embeddings -> ChromaDB (vectors)   +   SQLite (documents, chunks, cache, eval)
User question -> classifier gate -> exact cache -> semantic cache
     -> retrieval (BM25 / dense / hybrid) -> prompt -> Ollama LLM
     -> answer + citations  (or refusal if evidence is weak)
FastAPI  <->  Streamlit UI
```

### C2. Technology choices

| Component | Choice | Alternative considered | Why this choice |
|-----------|--------|------------------------|-----------------|
| PDF text extraction | **pypdf** | pdfplumber, PyMuPDF | Required by the project owner; pure Python, easy |
| Metadata / cache DB | **SQLite** | PostgreSQL | Zero setup, single file, enough for MVP |
| Vector DB | **ChromaDB** | FAISS | Stores text + metadata with vectors, persistent |
| Embeddings | **all-MiniLM-L6-v2** | bge-small-en | Small, fast on CPU, good quality |
| Keyword search | **rank_bm25** | Elasticsearch | No server needed |
| LLM | **Ollama + llama3.2:3b** | Hugging Face transformers | Easy local run |
| Classifier | **Logistic Regression baseline + fine-tuned MiniLM** | Reranker | Also gives the refusal gate |
| API / UI | **FastAPI / Streamlit** | Flask / Gradio | Auto docs, quick chat UI |
| Tests | **pytest** | unittest | Simple, standard |

### C3. Data pipeline
PDF -> text per page -> clean (remove extra spaces, headers) -> chunk with overlap -> SHA-256 hash of normalised text -> skip if hash exists -> embed -> save to SQLite + ChromaDB (hash used as ID, so re-run = no duplicates).

### C4. Chunking plan (important change for a small corpus)
The corpus has only ~3,300 words. Chunks of 256/512 *tokens* would give only ~10 big chunks, which makes experiments meaningless. So we test **two word-based sizes: 80 words and 160 words** (overlap 15 words). Both are recorded in `config.yaml`.

### C5. Model training and evaluation approach
Classifier on ~300 questions (in-scope about NovaTech policies, out-of-scope about everything else, including *tricky* ones like "What is the leave policy of Google?"). Split 70/15/15, fixed seed. Report P/R/F1 and confusion matrix on the **test** set only.

### C6. Testing strategy
Unit tests for: metrics (known small examples), ingestion (re-run = no duplicates), retrieval, classifier wrapper, cache. One E2E test: question -> answer with citations; out-of-scope question -> refusal.

### C7. Integration approach
Every component is a class with a small interface (`retrieve(query, k)`, `generate(prompt)`, `get/set` for cache), so parts can be swapped and mocked in tests.

---

## D. Time Management (20 hours)

| Activity | Hours | Steps |
|----------|-------|-------|
| Requirements, planning, project setup | 2 | 1-2 |
| Knowledge base (PDFs) and preprocessing | 2 | 3-4 |
| Architecture and database design | 2 | 5 |
| Core implementation (retrieval, RAG, cache, API/UI) | 5 | 6, 8, 11, 12 |
| Model training and experiments | 4 | 7, 9, 10 |
| Testing and debugging | 2 | 13 |
| Documentation, error analysis, demo | 3 | 14-16 |
| **Total** | **20** | |

---

## E. Risk Identification

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Slow LLM on laptop CPU | Long waits in demo | Use 3B (or 1B) model, short context (top 3-4 chunks), cache answers |
| Small corpus gives near-perfect retrieval | Experiments look unconvincing | Use different chunk sizes, add tricky questions, report honestly |
| Retrieval returns wrong chunk | Wrong answers | Hybrid retrieval, similarity threshold, refusal rule |
| LLM hallucination | Unsupported claims | Strict prompt ("answer only from context"), citation check, refusal |
| LLM judge is biased or inconsistent | Misleading scores | Compare with human labels, report kappa and disagreements |
| Small classifier dataset | Overfitting | Baseline model, validation set, report limitations |
| Semantic cache returns wrong answer | Wrong response | High threshold, entity/number check, KB-version key, cache only answered questions |
| Model download / install problems | Lost time | Install Ollama and models early (Step 2) |
| Time overrun | Missing P0 items | Follow P0 list first, drop P2 |

---

## F. Demo Strategy (about 20-25 minutes)

1. **Architecture** (2 min): show diagram.
2. **Ingestion** (3 min): run pipeline, then run again to show *0 duplicates*.
3. **Retrieval workflow** (3 min): BM25 vs dense vs hybrid on one question.
4. **Successful case** (3 min): "How many sick leave days do I get?" -> answer + citation.
5. **Edge / failure case** (3 min): out-of-scope question -> refusal; a tricky near-miss question.
6. **Evaluation results** (4 min): experiment table, classifier confusion matrix, answer-quality metrics, judge vs human.
7. **Limitations and improvements** (3 min): small corpus, English only, one more week = re-ranker, bigger data, better judge.

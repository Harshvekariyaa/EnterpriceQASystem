# 03. Technology Selection

For every tool: what it does, why it was chosen, advantages, limitations, and alternatives considered. Versions are the minimums in `requirements.txt`.

## Summary

| Layer | Choice | Main alternative |
|-------|--------|------------------|
| PDF reading | pypdf via LangChain `PyPDFLoader` | pdfplumber, PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (word-counted) | fixed-size sliding window |
| Embeddings | `all-MiniLM-L6-v2` | bge-small-en, OpenAI embeddings |
| Vector DB | ChromaDB | FAISS, pgvector |
| Relational DB | SQLite | PostgreSQL |
| Keyword search | `rank_bm25` | Elasticsearch |
| LLM runtime | Ollama + `llama3.2:1b` (judge `llama3.2:3b`) | Hugging Face transformers, hosted APIs |
| Classifier | Logistic Regression baseline + fine-tuned MiniLM | zero-shot LLM, cross-encoder |
| API | FastAPI + Uvicorn | Flask |
| UI | Streamlit | Gradio |
| Tests | pytest | unittest |
| Report | reportlab | LaTeX |

---

## Details

### pypdf (through LangChain `DirectoryLoader` + `PyPDFLoader`)
- **What:** reads each PDF page into text.
- **Why:** pure Python, no system install, the PDFs are text-based.
- **Advantages:** simple, gives page numbers (needed for citations).
- **Limitations:** a PDF stores layout, not text, so spacing can break; no OCR for scanned files; tables become plain lines. `cleaner.py` repairs the common problems.
- **Alternatives:** pdfplumber (better tables), PyMuPDF (faster, AGPL licence).

### LangChain text splitter
- **What:** splits text at newline, then sentence end, then space, so chunks end at natural places.
- **Why:** better chunk boundaries than a blind word window. `length_function` counts words, not characters.
- **Advantages:** small, well tested. Overlap (15 words) keeps facts that sit on a boundary.
- **Limitations:** a chunk size of 80 words suits this tiny corpus; a large corpus would need re-tuning.
- **Alternatives:** semantic chunking, markdown/heading-aware splitting.

### all-MiniLM-L6-v2 (sentence-transformers)
- **What:** turns text into a 384-number vector.
- **Why:** small (about 90 MB), fast on CPU, good general quality. Also the base of the fine-tuned classifier.
- **Advantages:** runs offline after one download; one model serves retrieval, cache and classifier.
- **Limitations:** English only; 256-token input limit; weaker than larger models on rare terms.
- **Alternatives:** bge-small-en (slightly better), larger e5/bge models (slower).

### ChromaDB
- **What:** stores vectors with metadata and finds nearest neighbours (cosine).
- **Why:** persistent on disk, no server, works with LangChain.
- **Advantages:** `upsert`-style IDs make ingestion idempotent; two collections (`chunks_80`, `chunks_160`) support the experiments.
- **Limitations:** single machine; no built-in hybrid search; API has changed across versions.
- **Alternatives:** FAISS (faster, no metadata store), pgvector (needs PostgreSQL).

### SQLite
- **What:** one-file relational database (`data/db/rag.sqlite`).
- **Why:** zero setup; holds documents, chunks, cache and evaluation runs.
- **Advantages:** ACID, foreign keys, easy to inspect with `scripts/check_db.py`.
- **Limitations:** one writer at a time; not meant for many concurrent users.
- **Alternatives:** PostgreSQL for a multi-user deployment.

### rank_bm25
- **What:** BM25 keyword scoring in Python.
- **Why:** strong on exact terms and numbers ("MFA", "Rs. 5,000"); no server.
- **Advantages:** index is rebuilt in memory from SQLite in milliseconds for 72 chunks.
- **Limitations:** misses paraphrases; the whole index sits in RAM.
- **Alternatives:** Elasticsearch/OpenSearch for large data.

### Hybrid retrieval (Reciprocal Rank Fusion)
- **What:** `score = 1/(60 + rank_bm25) + 1/(60 + rank_dense)`.
- **Why:** combines keyword and meaning search without matching two score scales.
- **Result:** best configuration in the experiments (see `reports/retrieval_results.csv`).
- **Limitation:** adds a second search per question.

### Ollama with llama3.2
- **What:** runs open LLMs locally behind an HTTP API (`localhost:11434`), called through `langchain-ollama` `ChatOllama`.
- **Why:** free, private, runs on CPU. Temperature 0 and `max_tokens` 300 keep answers stable and short.
- **Advantages:** no API key; model swap is one line in `config.yaml`.
- **Limitations:** the **1B model is weak**: in the draft evaluation 21 of 32 answers were only citation markers. The 3B judge is also small and noisy.
- **Alternatives:** larger local models (Llama 3.1 8B, Qwen), hosted APIs (not used: cost, privacy).

### scikit-learn, PyTorch, transformers (classifier)
- **What:** Logistic Regression on MiniLM embeddings (baseline, tuned with `GridSearchCV` and `StratifiedGroupKFold`), and a fine-tuned MiniLM with a classification head.
- **Why:** gives the required trained model and a smarter refusal layer than a similarity cut-off.
- **Advantages:** group-aware folds stop paraphrases leaking across splits.
- **Limitations:** only 264 self-written questions; near-miss questions are the hardest; threshold 0.5 produces borderline errors.
- **Alternatives:** zero-shot LLM classification, cross-encoder re-ranker scores.

### FastAPI + Uvicorn
- **What:** the web API (`GET /health`, `POST /ask`) with automatic docs at `/docs`.
- **Why:** typed request/response models (pydantic), quick to test.
- **Limitations:** pipeline is loaded once at start-up, so the first start is slow; no authentication.
- **Alternatives:** Flask.

### Streamlit
- **What:** the employee-facing page; talks only to the API.
- **Why:** a working UI in under 100 lines.
- **Limitations:** basic styling; reruns the script on each interaction.
- **Alternatives:** Gradio, a React front end.

### pytest, reportlab, matplotlib, pandas, numpy
- pytest runs 86 tests with the LLM mocked. reportlab builds the one-page PDF report. matplotlib draws the confusion matrices and threshold curves. pandas/numpy handle tables and vectors.

### Not used (and why)
- **Docker:** P2 item, not built. A local run is enough for the assessment.
- **Cross-encoder re-ranker:** P2; proposed as the "one more week" improvement.

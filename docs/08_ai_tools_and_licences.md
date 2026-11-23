# 08. AI Tools, Models and Licences

## 1. Models used by the system

| Model | Version / id | Used for | Licence | Source |
|-------|--------------|----------|---------|--------|
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | embeddings (retrieval, cache, classifier base) | Apache 2.0 | Hugging Face |
| Fine-tuned MiniLM classifier | trained here from the model above (5 epochs, lr 5e-5, seed 42) | in-scope / out-of-scope gate | Apache 2.0 (derived work, own training data) | saved to `models/scope_classifier/` (not in the zip) |
| Logistic Regression baseline | scikit-learn, `C=0.01`, `class_weight=balanced` | classifier baseline | BSD-3-Clause (library) | trained here |
| Llama 3.2 1B | Ollama tag `llama3.2:1b` | answer generation | Llama 3.2 Community Licence | Meta, via Ollama |
| Llama 3.2 3B | Ollama tag `llama3.2:3b` | LLM judge | Llama 3.2 Community Licence | Meta, via Ollama |

Training data for the classifier is self-written (264 questions, `data/eval/classifier_dataset.csv`). The PDFs are self-written for a fictional company. No third-party or personal data is used.

Llama 3.2 licence note: it allows use and modification, requires the licence and the "Built with Llama" attribution when redistributing, and carries an acceptable-use policy. Please read the current licence text on the model page before any commercial use.

## 2. Software libraries (from `requirements.txt`)

| Library | Purpose | Licence |
|---------|---------|---------|
| pypdf | PDF text | BSD-3-Clause |
| reportlab | PDF report | BSD |
| langchain-community, langchain-text-splitters, langchain-huggingface, langchain-chroma, langchain-ollama | loaders, splitter, integrations | MIT |
| numpy | arrays | BSD-3-Clause |
| pandas | tables | BSD-3-Clause |
| scikit-learn | baseline model, metrics, GridSearchCV | BSD-3-Clause |
| matplotlib | charts | Matplotlib licence (PSF-style) |
| torch | fine-tuning | BSD-style |
| transformers | fine-tuning | Apache 2.0 |
| sentence-transformers | embeddings | Apache 2.0 |
| chromadb | vector database | Apache 2.0 |
| rank_bm25 | BM25 | Apache 2.0 |
| requests, httpx | HTTP | Apache 2.0, BSD-3-Clause |
| fastapi | API | MIT |
| uvicorn | server | BSD-3-Clause |
| streamlit | UI | Apache 2.0 |
| pyyaml | config | MIT |
| pytest | tests | MIT |

Runtime tools: **Ollama** (MIT), **SQLite** (public domain). Licences were listed from general knowledge of these projects; check each project's repository for the exact current text before publishing.

## 3. AI tools used to build the project (disclosure)

> **Please check and edit this section so it is true for you.** The assessment asks for an honest disclosure.

- **Claude (Anthropic)** was used as a coding and learning assistant during the project: it proposed the step-by-step plan, generated and explained code for the ingestion, retrieval, RAG, classifier, evaluation, cache, API/UI and test steps, and drafted these documents from the project's real files and results.
- All code was run, tested and reviewed by the candidate. Evaluation questions, classifier questions and human labels are the candidate's own work. *(Edit if this is not accurate; the human labels are not yet filled in.)*
- Claude was **not** used at run time. The running system uses only the local models listed in section 1.
- The LLM judge (`llama3.2:3b`) grades answers automatically; its scores are checked against human labels, not trusted on their own.

## 4. Reproducibility

Seed 42; temperature 0; library versions of the classifier run are saved in `reports/classifier_training_config.json` (scikit-learn 1.9.1, torch 2.15.0+cpu, transformers 5.24.0).

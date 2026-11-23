# pipeline.py  (Step 8 - part 3)
# Job: the full RAG flow: question -> checks -> retrieval -> LLM -> answer + citations.

import re
import time

from src.config import load_config, resolve_path
from src.logger import get_logger
from src.rag.prompt import NOT_FOUND_MESSAGE, build_messages

log = get_logger("rag_pipeline")


def clean_question(question):
    """Remove extra spaces. Reject empty or very long questions."""
    question = " ".join(str(question).split())
    if question == "":
        raise ValueError("The question is empty.")
    if len(question) > 500:
        raise ValueError("The question is too long (maximum 500 characters).")
    return question

def clean_answer(answer):
    """Small models sometimes repeat a citation marker: '[1] [1] You get 12 days.'
    This turns repeated identical markers into one: '[1] You get 12 days.'"""
    return re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", answer)

def parse_citations(answer, chunks):
    """Find [1], [2] ... in the answer and connect them to the real chunks.
    Returns (citations, invalid_numbers).
    invalid_numbers = numbers the LLM used that do not exist (like [7] when we gave 4 passages)."""
    citations = []
    invalid = []
    used_numbers = []
    for text in re.findall(r"\[(\d+)\]", answer):
        number = int(text)
        if number < 1 or number > len(chunks):
            if number not in invalid:
                invalid.append(number)
        elif number not in used_numbers:
            used_numbers.append(number)

    for number in used_numbers:
        chunk = chunks[number - 1]
        citations.append({
            "number": number,
            "file_name": chunk["file_name"],
            "section": chunk["section"],
            "page": chunk["page"],
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "snippet": chunk["text"].replace("\n", " ")[:160],
        })
    return citations, invalid


class RAGPipeline:
    def __init__(self, retriever, gate_retriever, generator, top_k, max_context_chunks,
                 min_similarity, scope_checker=None):
        self.retriever = retriever                # finds the chunks for the prompt
        self.gate_retriever = gate_retriever      # dense search, used ONLY for the similarity gate
        self.generator = generator                # talks to the LLM
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.min_similarity = min_similarity
        self.scope_checker = scope_checker        # optional classifier (added in Step 9)

    def _result(self, question, answer, refused, reason, similarity, chunks, citations,
                warnings, start_time):
        return {
            "question": question,
            "answer": answer,
            "refused": refused,
            "refusal_reason": reason,             # None, "out_of_scope_classifier",
                                                  # "low_similarity" or "model_said_unknown"
            "top_similarity": similarity,
            "citations": citations,
            "citation_warnings": warnings,
            "retrieved": chunks,
            "latency_seconds": round(time.perf_counter() - start_time, 2),
        }

    def ask(self, question):
        start_time = time.perf_counter()
        question = clean_question(question)

        # Layer 1 (optional): trained classifier says "not about our documents"
        if self.scope_checker is not None:
            in_scope, probability = self.scope_checker(question)
            if not in_scope:
                log.info("Refused (classifier): %s", question)
                return self._result(question, NOT_FOUND_MESSAGE, True, "out_of_scope_classifier",
                                    None, [], [], [], start_time)

        # Layer 2: similarity gate. Is the BEST chunk close enough to the question?
        gate = self.gate_retriever.retrieve(question, k=1)
        top_similarity = gate[0]["score"] if len(gate) > 0 else 0.0
        if top_similarity < self.min_similarity:
            log.info("Refused (low similarity %.3f): %s", top_similarity, question)
            return self._result(question, NOT_FOUND_MESSAGE, True, "low_similarity",
                                top_similarity, [], [], [], start_time)

        # Retrieve chunks and ask the LLM
        chunks = self.retriever.retrieve(question, k=self.top_k)[:self.max_context_chunks]
        messages = build_messages(question, chunks)
        answer = clean_answer(self.generator.generate(messages))

        # Layer 3: the LLM itself said the context has no answer
        if NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower():
            log.info("Refused (model said unknown): %s", question)
            return self._result(question, NOT_FOUND_MESSAGE, True, "model_said_unknown",
                                top_similarity, chunks, [], [], start_time)

        # Turn [1] [2] into real citations and check them
        citations, invalid = parse_citations(answer, chunks)
        warnings = []
        if len(citations) == 0:
            warnings.append("The answer has no citation.")
        if len(invalid) > 0:
            warnings.append("The answer cites passages that do not exist: " + str(invalid))

        return self._result(question, answer, False, None, top_similarity, chunks,
                            citations, warnings, start_time)


def build_pipeline(scope_checker=None):
    """Create a ready-to-use RAGPipeline from config.yaml."""
    # imported here so this file can be tested without LangChain installed
    from src.rag.generator import OllamaGenerator
    from src.retrieval.factory import build_retriever
    from src.storage import sqlite_store, vector_store

    config = load_config()
    chunk_size = config["ingestion"]["default_chunk_size"]
    retriever_name = config["retrieval"].get("best_retriever", "hybrid")

    conn = sqlite_store.connect(str(resolve_path(config["paths"]["sqlite_path"])))
    embeddings = vector_store.get_embeddings()
    retriever = build_retriever(retriever_name, chunk_size, conn, embeddings)
    gate_retriever = build_retriever("dense", chunk_size, conn, embeddings)
    conn.close()      # the retrievers keep their own copies of the data

    # Optional: the trained in-scope / out-of-scope classifier (Step 9)
    if scope_checker is None and config["rag"].get("use_classifier", False):
        from src.classifier.predict import ScopeClassifier
        scope_checker = ScopeClassifier()

    log.info("Pipeline ready: retriever=%s chunk_size=%s classifier=%s",
             retriever_name, chunk_size, scope_checker is not None)
    return RAGPipeline(
        retriever=retriever,
        gate_retriever=gate_retriever,
        generator=OllamaGenerator(),
        top_k=config["retrieval"]["top_k"],
        max_context_chunks=config["rag"]["max_context_chunks"],
        min_similarity=config["rag"]["min_similarity"],
        scope_checker=scope_checker,
    )
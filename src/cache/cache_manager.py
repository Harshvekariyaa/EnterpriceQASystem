# cache_manager.py  (Step 11 - part 3)
# Job: put the caches in front of the RAG pipeline.
# Order:  exact cache  ->  semantic cache  ->  full RAG pipeline.

import time

from src.cache.exact_cache import ExactCache
from src.cache.semantic_cache import SemanticCache
from src.config import load_config, resolve_path
from src.logger import get_logger
from src.rag.pipeline import clean_question

log = get_logger("cache_manager")


def is_cacheable(result):
    """Safety 3: only good, answered, properly cited answers are stored.
    Refusals are NOT cached (the knowledge base may be updated later),
    and neither are answers with citation warnings (they may be wrong)."""
    return (not result["refused"]
            and len(result["citations"]) > 0
            and len(result["citation_warnings"]) == 0)


class CachedRAG:
    def __init__(self, pipeline, embed_fn=None, db_path=None, kb_version=None, threshold=None):
        config = load_config()
        if db_path is None:
            db_path = str(resolve_path(config["paths"]["sqlite_path"]))
        if kb_version is None:
            kb_version = config["project"]["kb_version"]
        if threshold is None:
            threshold = config["cache"]["semantic_threshold"]
        if embed_fn is None:
            from src.storage import vector_store                   # loads the embedding model once
            embed_fn = vector_store.get_embeddings().embed_query

        self.pipeline = pipeline
        self.embed_fn = embed_fn
        self.exact = ExactCache(db_path, kb_version)
        self.semantic = SemanticCache(db_path, kb_version, embed_fn, threshold)

    def _hit_result(self, question, cached, source, similarity, start_time):
        return {
            "question": question, "answer": cached["answer"], "refused": False,
            "refusal_reason": None, "top_similarity": None,
            "citations": cached["citations"], "citation_warnings": [], "retrieved": [],
            "cache": source, "cache_similarity": similarity,
            "latency_seconds": round(time.perf_counter() - start_time, 3),
        }

    def ask(self, question):
        start_time = time.perf_counter()
        question = clean_question(question)            # same checks as the pipeline

        # 1) exact cache: no embedding needed, fastest
        cached = self.exact.get(question)
        if cached is not None:
            log.info("Cache HIT (exact): %s", question)
            return self._hit_result(question, cached, "exact", 1.0, start_time)

        # 2) semantic cache. We embed once and reuse that embedding if we must store the answer.
        embedding = self.embed_fn(question)
        cached = self.semantic.get(question, embedding)
        if cached is not None:
            log.info("Cache HIT (semantic %.3f): %s  ~  %s",
                     cached["similarity"], question, cached["matched_question"])
            return self._hit_result(question, cached, "semantic", cached["similarity"], start_time)

        # 3) full pipeline
        result = self.pipeline.ask(question)
        result["cache"] = "none"
        result["cache_similarity"] = None
        if is_cacheable(result):
            self.exact.put(question, result["answer"], result["citations"], embedding)
            log.info("Cached answer for: %s", question)
        return result
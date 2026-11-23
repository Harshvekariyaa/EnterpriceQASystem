# helpers.py  (Step 13)
# Small FAKE parts used by several test files. A fake is a simple stand-in for a slow or
# external part (the LLM, the vector database, the classifier), so tests run in seconds
# and give the same result every time.

from src.rag.pipeline import RAGPipeline


def make_chunk(number=1, text="Employees are entitled to 12 days of paid sick leave per year.",
               file_name="01_Leave_Policy.pdf", section="3. Sick Leave", page=1):
    return {
        "rank": number, "chunk_id": "chunk" + str(number), "text": text, "score": 0.8,
        "file_name": file_name, "section": section, "page": page,
        "doc_id": "HR-LEAVE-01", "chunk_size": 80, "retriever": "fake",
    }


class FakeRetriever:
    """Returns a fixed list of results and counts how often it was called."""
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def retrieve(self, question, k=5):
        self.calls += 1
        return self.results[:k]


class FakeGenerator:
    """Pretends to be the LLM: always returns the same reply, counts the calls."""
    def __init__(self, reply="Employees get 12 days of paid sick leave [1]."):
        self.reply = reply
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return self.reply


def make_pipeline(reply="Employees get 12 days of paid sick leave [1].", gate_score=0.8,
                  scope_checker=None, chunks=None):
    """A real RAGPipeline built from fake parts."""
    if chunks is None:
        chunks = [make_chunk(1), make_chunk(2, text="Sick leave needs a medical certificate.", page=2)]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(chunks),
        gate_retriever=FakeRetriever([{"score": gate_score}]),
        generator=FakeGenerator(reply),
        top_k=5, max_context_chunks=4, min_similarity=0.25,
        scope_checker=scope_checker,
    )
    return pipeline


class FakePipeline:
    """A very small stand-in for the whole RAG pipeline (used by the cache tests)."""
    def __init__(self, refused=False, citations=None, warnings=None):
        self.calls = 0
        self.refused = refused
        self.citations = [{"number": 1, "file_name": "01_Leave_Policy.pdf"}] if citations is None else citations
        self.warnings = [] if warnings is None else warnings

    def ask(self, question):
        self.calls += 1
        return {
            "question": question, "answer": "12 days [1]", "refused": self.refused,
            "refusal_reason": "low_similarity" if self.refused else None,
            "top_similarity": 0.8, "citations": self.citations,
            "citation_warnings": self.warnings, "retrieved": [], "latency_seconds": 0.1,
        }
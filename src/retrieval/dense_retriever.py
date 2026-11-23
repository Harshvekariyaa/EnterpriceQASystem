# dense_retriever.py  (Step 6 - meaning search)
# The question is turned into numbers (embedding). ChromaDB finds the chunks
# whose numbers are closest. Score = cosine similarity (1.0 = same meaning).

from src.retrieval.base import make_result


class DenseSearch:
    def __init__(self, vector_store):
        self.store = vector_store
        self.name = "dense"

    def retrieve(self, question, k=5):
        pairs = self.store.similarity_search_with_score(question, k=k)
        results = []
        for position, (doc, distance) in enumerate(pairs, start=1):
            similarity = 1 - distance            # Chroma gives a distance; small = close
            results.append(make_result(position, doc.metadata["chunk_id"], doc.page_content,
                                       doc.metadata, round(similarity, 4), self.name))
        return results
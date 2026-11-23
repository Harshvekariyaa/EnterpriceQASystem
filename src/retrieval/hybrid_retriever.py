# hybrid_retriever.py  (Step 6 - BM25 + Dense together)
# Reciprocal Rank Fusion (RRF):
#     score(chunk) = 1/(rrf_k + rank_in_bm25) + 1/(rrf_k + rank_in_dense)
# A chunk ranked high by BOTH searches gets the highest score.

from src.retrieval.base import make_result


class HybridSearch:
    def __init__(self, bm25_search, dense_search, rrf_k=60, candidates=20):
        self.bm25 = bm25_search
        self.dense = dense_search
        self.rrf_k = rrf_k                # a constant that softens the effect of rank 1 vs 2
        self.candidates = candidates      # how many results to take from each search
        self.name = "hybrid"

    def retrieve(self, question, k=5):
        bm25_results = self.bm25.retrieve(question, self.candidates)
        dense_results = self.dense.retrieve(question, self.candidates)

        scores = {}      # chunk_id -> RRF score
        details = {}     # chunk_id -> one result dictionary (to keep text and metadata)
        for results in [bm25_results, dense_results]:
            for r in results:
                chunk_id = r["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + r["rank"])
                details[chunk_id] = r

        best_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]

        final = []
        for position, chunk_id in enumerate(best_ids, start=1):
            r = details[chunk_id]
            final.append(make_result(position, chunk_id, r["text"], r,
                                     round(scores[chunk_id], 5), self.name))
        return final
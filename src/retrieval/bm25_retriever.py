# bm25_retriever.py  (Step 6 - keyword search)
# BM25 gives a higher score to chunks that contain the question's words,
# especially RARE words ("maternity" counts more than "the").

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.retrieval.base import make_result, tokenize


class BM25Search:
    def __init__(self, chunks):
        """chunks = list of chunk dictionaries (from SQLite)."""
        documents = []
        for c in chunks:
            documents.append(Document(
                page_content=c["text"],
                metadata={"chunk_id": c["chunk_id"], "file_name": c["file_name"],
                          "section": c["section"], "page": c["page"],
                          "doc_id": c["doc_id"], "chunk_size": c["chunk_size"]},
            ))
        # preprocess_func = how text is cut into words (same function for chunks and questions)
        self.retriever = BM25Retriever.from_documents(documents, preprocess_func=tokenize)
        self.name = "bm25"

    def retrieve(self, question, k=5):
        self.retriever.k = k
        documents = self.retriever.invoke(question)
        results = []
        for position, doc in enumerate(documents, start=1):
            results.append(make_result(position, doc.metadata["chunk_id"], doc.page_content,
                                       doc.metadata, None, self.name))   # BM25 score is not exposed
        return results
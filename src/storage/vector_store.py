# vector_store.py  (Step 5 - part 2)
# Job: embeddings + ChromaDB, using LangChain.
#
# Embedding   = a list of 384 numbers that describes the MEANING of a text.
# ChromaDB    = a database that finds the texts whose numbers are closest to the question's numbers.
# We use TWO collections (like two folders): chunks_80 and chunks_160.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import load_config, resolve_path


def get_embeddings():
    """Load the embedding model (first time it downloads about 90 MB)."""
    settings = load_config()["embedding"]
    return HuggingFaceEmbeddings(
        model_name=settings["model_name"],
        encode_kwargs={"normalize_embeddings": True},   # makes cosine similarity work well
    )


def get_vector_store(chunk_size, embeddings):
    """Open (or create) the Chroma collection for one chunk size."""
    chroma_folder = str(resolve_path(load_config()["paths"]["chroma_dir"]))
    return Chroma(
        collection_name="chunks_" + str(chunk_size),
        embedding_function=embeddings,
        persist_directory=chroma_folder,                 # saved on disk
        collection_metadata={"hnsw:space": "cosine"},    # compare by cosine distance
    )


def get_all_ids(store):
    """All chunk ids currently inside the collection (as a set)."""
    return set(store.get()["ids"])


def add_chunks(store, chunks, batch_size=64):
    """Embed and store chunks. The chunk_id is used as the Chroma id."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = []
        metadatas = []
        ids = []
        for c in batch:
            texts.append(c["text"])
            metadatas.append({                     # Chroma only accepts simple values
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "file_name": c["file_name"],
                "title": c["title"],
                "section": c["section"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "chunk_size": c["chunk_size"],
                "word_count": c["word_count"],
            })
            ids.append(c["chunk_id"])
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def remove_ids(store, ids):
    """Delete chunks from the collection."""
    if len(ids) > 0:
        store.delete(ids=list(ids))
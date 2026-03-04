import os
import chromadb
import voyageai

VOYAGE_MODEL = "voyage-3"

_client = None
_collection = None
_vo = None

def _init():
    global _client, _collection, _vo

    chroma_dir = os.getenv("CHROMA_DIR", "data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "gotonanotech")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not voyage_key:
        raise ValueError("Missing VOYAGE_API_KEY")

    if _vo is None:
        _vo = voyageai.Client(api_key=voyage_key)

    if _client is None:
        _client = chromadb.PersistentClient(path=chroma_dir)

    if _collection is None:
        _collection = _client.get_collection(name=collection_name)

def retrieve(question: str, k: int = 5):
    _init()

    q_emb = _vo.embed(texts=[question], model=VOYAGE_MODEL).embeddings[0]

    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    sources = []

    for doc, meta in zip(docs, metas):
        src = meta.get("source", "")
        sources.append(src)
        context_blocks.append(f"Source: {src}\n{doc}")

    unique_sources = sorted(set([s for s in sources if s]))
    return "\n\n".join(context_blocks), unique_sources
import os
from dotenv import load_dotenv
import chromadb
import voyageai

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "gotonanotech"
VOYAGE_MODEL = "voyage-3"

def main():
    load_dotenv()
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        raise ValueError("Missing VOYAGE_API_KEY in .env")

    vo = voyageai.Client(api_key=voyage_key)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    while True:
        q = input("\nAsk a question (or type quit): ").strip()
        if not q or q.lower() in ["quit", "exit"]:
            break

        # Embed the query with the same model used during ingestion (1024 dims)
        q_emb = vo.embed(texts=[q], model=VOYAGE_MODEL).embeddings[0]

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        print("\nTop matches:\n")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", "na")
            print(f"{i}) source: {source} | chunk: {chunk_index} | distance: {dist:.4f}")
            print(doc[:500].replace("\n", " "))
            print()

if __name__ == "__main__":
    main()

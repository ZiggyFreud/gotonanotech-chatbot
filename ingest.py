import os
import re
import hashlib
from dotenv import load_dotenv

import chromadb
import voyageai

INPUT_FILE = "data/website_text.txt"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "gotonanotech"

CHUNK_SIZE = 1200        # characters per chunk
CHUNK_OVERLAP = 200      # overlap to keep context
MIN_CHUNK_CHARS = 300    # skip tiny chunks

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks

def parse_sections(big_text: str) -> list[tuple[str, str]]:
    """
    Splits the combined file into (url, page_text) pairs based on:
    ===== URL =====
    """
    pattern = r"===== (https?://[^ ]+) ====="
    parts = re.split(pattern, big_text)

    sections = []
    if len(parts) < 3:
        return sections

    for i in range(1, len(parts), 2):
        url = parts[i].strip()
        page_text = parts[i + 1].strip()
        if page_text:
            sections.append((url, page_text))

    return sections

def main():
    load_dotenv()

    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        raise ValueError("Missing VOYAGE_API_KEY in .env")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        big_text = f.read()

    pages = parse_sections(big_text)
    if not pages:
        raise ValueError("Could not parse any pages. Check the format in website_text.txt")

    print(f"Parsed {len(pages)} pages from {INPUT_FILE}")

    # Chroma setup (persistent)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    vo = voyageai.Client(api_key=voyage_key)

    total_added = 0

    for (url, page_text) in pages:
        chunks = split_into_chunks(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            continue

        # Create embeddings in batches
        batch_size = 64
        for b in range(0, len(chunks), batch_size):
            batch = chunks[b:b + batch_size]

            # IDs must be unique
            ids = [stable_id(url + "||" + str(b + j) + "||" + batch[j]) for j in range(len(batch))]

            # Metadata for retrieval and debugging
            metadatas = [{"source": url, "chunk_index": b + j} for j in range(len(batch))]

            emb = vo.embed(texts=batch, model="voyage-3").embeddings

            collection.add(
                ids=ids,
                documents=batch,
                metadatas=metadatas,
                embeddings=emb
            )

            total_added += len(batch)

        print(f"Stored {len(chunks)} chunks from {url}")

    print(f"Done. Total chunks stored: {total_added}")
    print(f"Chroma DB folder: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_documents"

CHUNK_SIZE = 160
CHUNK_OVERLAP = 30
TOP_K = 5


# --------------------------------------------------
# 2. LOAD THE EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# 3. CONNECT TO CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------


def chunk_text(text: str) -> list[dict]:
    """
    Split documents by their headings and paragraphs.
    Keep related information together.
    Split oversized paragraphs when necessary.
    """

    sections = []

    current_heading = "Document overview"
    current_paragraphs = []
    current_paragraph = []

    def save_paragraph():
        if current_paragraph:
            current_paragraphs.append(
                " ".join(current_paragraph)
            )
            current_paragraph.clear()

    def save_section():
        save_paragraph()

        if current_paragraphs:
            sections.append({
                "section": current_heading,
                "paragraphs": current_paragraphs.copy()
            })
            current_paragraphs.clear()

    # Identify headings and paragraphs
    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            save_paragraph()
            continue

        # Recognise headings such as:
        # Returns and Replacements:
        is_heading = (
            line.endswith(":")
            and len(line.split()) <= 8
        )

        if is_heading:
            save_section()
            current_heading = line[:-1].strip()

        else:
            current_paragraph.append(line)

    save_section()

    # Convert sections into searchable chunks
    chunks = []

    for section in sections:

        heading = section["section"]
        paragraphs = section["paragraphs"]

        current_words = []

        def save_chunk():
            if current_words:
                chunks.append({
                    "section": heading,
                    "text": (
                        f"SECTION: {heading}\n\n"
                        + " ".join(current_words)
                    )
                })
                current_words.clear()

        for paragraph in paragraphs:

            words = paragraph.split()

            # Split exceptionally long paragraphs
            if len(words) > CHUNK_SIZE:

                save_chunk()

                start = 0

                while start < len(words):

                    end = start + CHUNK_SIZE

                    chunk_words = words[start:end]

                    chunks.append({
                        "section": heading,
                        "text": (
                            f"SECTION: {heading}\n\n"
                            + " ".join(chunk_words)
                        )
                    })

                    if end >= len(words):
                        break

                    start = end - CHUNK_OVERLAP

                continue

            # Keep paragraphs together when possible
            if (
                current_words
                and len(current_words) + len(words) > CHUNK_SIZE
            ):
                save_chunk()

            current_words.extend(words)

        save_chunk()

    return chunks


# --------------------------------------------------
# 5. GENERATE STABLE DOCUMENT IDs
# --------------------------------------------------

def generate_chunk_id(source: str, index: int, text: str) -> str:
    """
    Generate a stable ID for each document chunk.
    """

    unique_text = f"{source}:{index}:{text}"

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 6. INGEST DOCUMENTS INTO CHROMADB
# --------------------------------------------------



def ingest_documents():
    files = list(DATA_DIR.glob("*.txt"))
    current_sources = {file.name for file in files}

    print("\nSynchronising document knowledge base...")

    # Find documents already stored in ChromaDB.
    existing_records = collection.get(include=["metadatas"])

    existing_sources = {
        metadata["source"]
        for metadata in existing_records["metadatas"]
    }

    # Remove records belonging to deleted files.
    for source in existing_sources - current_sources:
        collection.delete(where={"source": source})
        print(f"Removed deleted document: {source}")

    total_chunks = 0
    processed_files = 0
    skipped_files = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        # Generate a fingerprint of the document content.
        content_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        existing = collection.get(
            where={"source": file_path.name},
            include=["metadatas"]
        )

        existing_hashes = {
            item.get("content_hash")
            for item in existing["metadatas"]
        }

        # Skip documents that have not changed.
        if existing["ids"] and existing_hashes == {content_hash}:
            print(f"Unchanged, skipped: {file_path.name}")
            skipped_files += 1
            continue

        # Remove old chunks before indexing updated content.
        if existing["ids"]:
            collection.delete(where={"source": file_path.name})

        chunks = chunk_text(text)

        if not chunks:
            print(f"No content found: {file_path.name}")
            continue

        documents = []
        ids = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            chunk_text_value = chunk["text"]
            section = chunk["section"]

            documents.append(chunk_text_value)

            ids.append(
                generate_chunk_id(
                    file_path.name,
                    index,
                    f"{section}:{chunk_text_value}"
                )
            )

            metadatas.append({
                "source": file_path.name,
                "section": section,
                "chunk_index": index,
                "content_hash": content_hash
            })

        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(documents)
        processed_files += 1

        print(
            f"Indexed {file_path.name}: "
            f"{len(documents)} chunks"
        )

    print("\nKnowledge base synchronisation complete.")
    print(f"Updated documents: {processed_files}")
    print(f"Unchanged documents skipped: {skipped_files}")
    print(f"Newly indexed chunks: {total_chunks}")
    print(f"Total records in collection: {collection.count()}")


# --------------------------------------------------
# 7. SEARCH THE DOCUMENTS
# --------------------------------------------------

def search_documents(query: str, top_k: int = TOP_K):
    """
    Convert the user's question into an embedding
    and retrieve the most similar document chunks.
    """

    if not query.strip():
        print("Please enter a search query.")
        return

    if collection.count() == 0:
        print("The collection is empty. Ingest documents first.")
        return

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\n" + "=" * 60)
    print(f"SEARCH QUERY: {query}")
    print("=" * 60)

    for index, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        print(f"\nRESULT {index + 1}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"\n{document}")
        print("-" * 60)


# --------------------------------------------------
# 8. INTERACTIVE SEARCH
# --------------------------------------------------

def interactive_search():

    print("\nSEMANTIC DOCUMENT SEARCH ENGINE")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        search_documents(query)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    ingest_documents()

    interactive_search()

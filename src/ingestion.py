import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunker import recursive_chunk
from src.loader import load_document


# ==================================================
# CONFIGURATION
# ==================================================

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "./chroma_db"
)


# ==================================================
# EMBEDDING MODEL
# ==================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CHROMADB
# ==================================================

Path(CHROMA_DB_PATH).mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)


collection = client.get_or_create_collection(
    name="mini_rag"
)


# ==================================================
# DOCUMENT INGESTION
# ==================================================

def ingest_document(
    file_path: str,
    session_id: str
) -> dict[str, Any]:
    """
    Load, chunk, embed and store a document.
    """

    # --------------------------------------------------
    # 1. Load document
    # --------------------------------------------------

    document = load_document(
        file_path
    )


    # --------------------------------------------------
    # 2. Split document into chunks
    # --------------------------------------------------

    chunks = recursive_chunk(
        document,
        max_words=40
    )


    if not chunks:

        raise ValueError(
            "No readable text was found in the document."
        )


    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    embeddings = model.encode(
        chunks
    )


    # --------------------------------------------------
    # 4. Create unique document ID
    # --------------------------------------------------

    document_id = str(
        uuid.uuid4()
    )


    # --------------------------------------------------
    # 5. Create unique chunk IDs
    # --------------------------------------------------

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]


    # --------------------------------------------------
    # 6. Create metadata
    # --------------------------------------------------

    metadatas: list[dict[str, str]] = [

        {
            "source": file_path,
            "document_id": document_id,
            "session_id": session_id,
            "chunk_id": str(i)
        }

        for i in range(len(chunks))
    ]


    # --------------------------------------------------
    # 7. Store chunks
    # --------------------------------------------------

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


    # --------------------------------------------------
    # 8. Return document information
    # --------------------------------------------------

    return {

        "file": file_path,

        "document_id": document_id,

        "session_id": session_id,

        "chunks": len(chunks)

    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_session_id = str(
        uuid.uuid4()
    )


    result = ingest_document(
        "data/test_document.docx",
        test_session_id
    )


    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)

    print(
        "ChromaDB path:",
        CHROMA_DB_PATH
    )

    print(
        "File:",
        result["file"]
    )

    print(
        "Session ID:",
        result["session_id"]
    )

    print(
        "Document ID:",
        result["document_id"]
    )

    print(
        "Chunks stored:",
        result["chunks"]
    )
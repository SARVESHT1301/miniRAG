import os
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
# BUILD VECTOR STORE
# ==================================================

def build_vector_store() -> None:

    # --------------------------------------------------
    # 1. Load document
    # --------------------------------------------------

    document = load_document(
        "data/cloud_computing.txt"
    )


    # --------------------------------------------------
    # 2. Chunk document
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
    # 4. Create IDs
    # --------------------------------------------------

    ids = [

        f"vector_store_chunk_{i}"

        for i in range(
            len(chunks)
        )

    ]


    # --------------------------------------------------
    # 5. Metadata
    # --------------------------------------------------

    metadatas: list[dict[str, str]] = [

        {
            "source": "cloud_computing.txt",
            "chunk_id": str(i)
        }

        for i in range(
            len(chunks)
        )

    ]


    # --------------------------------------------------
    # 6. Store vectors
    # --------------------------------------------------

    collection.upsert(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist(),

        metadatas=metadatas

    )


    # --------------------------------------------------
    # 7. Display result
    # --------------------------------------------------

    print(
        "Vector store created successfully."
    )

    print(
        "ChromaDB path:",
        CHROMA_DB_PATH
    )

    print(
        "Number of chunks stored:",
        collection.count()
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    build_vector_store()
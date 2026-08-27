import os
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


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
# RETRIEVAL SETTINGS
# ==================================================

# Smaller distance = more similar.
#
# We retrieve candidates first and allow the
# cross-encoder reranker to make the final
# relevance decision.

RELEVANCE_THRESHOLD = 1.5


# ==================================================
# RETRIEVE
# ==================================================

def retrieve(
    query: str,
    document_id: str,
    session_id: str,
    top_k: int = 8
) -> list[dict[str, Any]]:

    # --------------------------------------------------
    # 1. Create query embedding
    # --------------------------------------------------

    query_embedding = model.encode(
        [query]
    )


    # --------------------------------------------------
    # 2. Search only the selected document
    #    inside the current session
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=query_embedding.tolist(),

        n_results=top_k,

        where={
            "$and": [

                {
                    "document_id": document_id
                },

                {
                    "session_id": session_id
                }

            ]
        },

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    # --------------------------------------------------
    # 3. Validate results
    # --------------------------------------------------

    documents = results.get(
        "documents"
    )

    metadatas = results.get(
        "metadatas"
    )

    distances = results.get(
        "distances"
    )

    ids = results.get(
        "ids"
    )


    if not documents:
        return []


    if not metadatas:
        return []


    if not distances:
        return []


    if not ids:
        return []


    # --------------------------------------------------
    # 4. Chroma returns nested lists
    # --------------------------------------------------

    documents = documents[0]

    metadatas = metadatas[0]

    distances = distances[0]

    ids = ids[0]


    # --------------------------------------------------
    # 5. Build candidate chunks
    # --------------------------------------------------

    retrieved_chunks: list[
        dict[str, Any]
    ] = []


    for i in range(
        len(documents)
    ):

        distance = distances[i]


        # --------------------------------------------------
        # 6. Relevance filtering
        # --------------------------------------------------

        if distance > RELEVANCE_THRESHOLD:
            continue


        retrieved_chunks.append(

            {
                "id": ids[i],

                "text": documents[i],

                "metadata": metadatas[i],

                "distance": distance
            }

        )


    # --------------------------------------------------
    # 7. Return candidates
    # --------------------------------------------------

    return retrieved_chunks


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RETRIEVER MODULE")
    print("=" * 60)

    print()

    print(
        "Retriever loaded successfully."
    )

    print()

    print(
        "ChromaDB path:",
        CHROMA_DB_PATH
    )

    print()

    print(
        "Relevance threshold:",
        RELEVANCE_THRESHOLD
    )
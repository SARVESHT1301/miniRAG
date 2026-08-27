from typing import Any

from sentence_transformers import CrossEncoder


# ==================================================
# RERANKER MODEL
# ==================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


print("Loading reranker model...")

reranker = CrossEncoder(
    MODEL_NAME
)


print("Reranker loaded successfully.")


# ==================================================
# RERANK DOCUMENTS
# ==================================================

def rerank(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    top_k: int = 3
) -> list[dict[str, Any]]:
    """
    Rerank retrieved document chunks using a CrossEncoder.

    Parameters
    ----------
    query:
        User's question or rewritten retrieval query.

    retrieved_chunks:
        Chunks returned by the vector retriever.

    top_k:
        Number of best chunks to return.

    Returns
    -------
    list[dict[str, Any]]
        Reranked chunks ordered from most relevant
        to least relevant.
    """

    # --------------------------------------------------
    # No chunks
    # --------------------------------------------------

    if not retrieved_chunks:

        return []


    # --------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------

    pairs = [
        (
            query,
            chunk["text"]
        )
        for chunk in retrieved_chunks
    ]


    # --------------------------------------------------
    # Calculate reranker scores
    # --------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------
    # Attach reranker score
    # --------------------------------------------------

    reranked_chunks = []

    for chunk, score in zip(
        retrieved_chunks,
        scores
    ):

        updated_chunk = dict(
            chunk
        )

        updated_chunk["rerank_score"] = float(
            score
        )

        reranked_chunks.append(
            updated_chunk
        )


    # --------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------

    reranked_chunks.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )


    # --------------------------------------------------
    # Return top results
    # --------------------------------------------------

    return reranked_chunks[:top_k]


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RERANKER MODULE")
    print("=" * 60)


    # --------------------------------------------------
    # Example retrieved chunks
    # --------------------------------------------------

    test_chunks = [

        {
            "id": "chunk_1",
            "text": (
                "Supervised learning uses labeled "
                "training data."
            ),
            "metadata": {
                "source": "test_document.pdf"
            },
            "distance": 0.50
        },

        {
            "id": "chunk_2",
            "text": (
                "Unsupervised learning works with "
                "unlabeled data."
            ),
            "metadata": {
                "source": "test_document.pdf"
            },
            "distance": 0.60
        },

        {
            "id": "chunk_3",
            "text": (
                "Machine learning is a branch of "
                "artificial intelligence."
            ),
            "metadata": {
                "source": "test_document.pdf"
            },
            "distance": 0.70
        }
    ]


    # --------------------------------------------------
    # Test query
    # --------------------------------------------------

    query = (
        "What kind of data does supervised learning use?"
    )


    print("\nQUERY")
    print("-" * 40)

    print(query)


    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    results = rerank(
        query=query,
        retrieved_chunks=test_chunks,
        top_k=3
    )


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RERANKED RESULTS")
    print("=" * 60)


    for index, result in enumerate(
        results
    ):

        print(
            f"\nRank {index + 1}"
        )

        print("-" * 40)

        print(
            "ID:",
            result["id"]
        )

        print(
            "Vector distance:",
            result["distance"]
        )

        print(
            "Rerank score:",
            result["rerank_score"]
        )

        print(
            "Text:",
            result["text"]
        )
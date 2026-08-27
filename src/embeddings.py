from sentence_transformers import SentenceTransformer
from src.chunker import recursive_chunk
from src.loader import load_document

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text):
    return model.encode(text)


if __name__ == "__main__":

    document = load_document("data/cloud_computing.txt")

    chunks = recursive_chunk(document, max_words=40)

    print("=" * 60)
    print("Number of chunks:", len(chunks))
    print("=" * 60)

    embeddings = []

    for i, chunk in enumerate(chunks):
        vector = create_embedding(chunk)
        embeddings.append(vector)

        print(f"\nChunk {i+1}")
        print("-" * 30)
        print(chunk)
        print(f"Vector Dimension : {len(vector)}")
        print(f"First 5 Values   : {vector[:5]}")

    print("\nTotal Embeddings Generated:", len(embeddings))
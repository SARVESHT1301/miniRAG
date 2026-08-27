import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="mini_rag"
)


print("=" * 60)
print("VECTOR STORE INSPECTION")
print("=" * 60)

print("Collection name:", collection.name)
print("Number of items:", collection.count())


data = collection.get(
    include=["documents", "metadatas"]
)

ids = data["ids"]
documents = data["documents"]
metadatas = data["metadatas"]

assert ids is not None
assert documents is not None
assert metadatas is not None


for i in range(len(ids)):

    print(f"\nID: {ids[i]}")

    print("Document:")
    print(documents[i])

    print("Metadata:")
    print(metadatas[i])
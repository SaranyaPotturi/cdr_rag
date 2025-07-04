import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# Custom embedding wrapper class
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

# --- Configuration for ChromaDB ---
CHROMA_DB_PATH = "./chroma_db"

def initialize_chromadb(documents, metadatas, ids, embedding_model, embedding_model_name):
    """Initializes and populates ChromaDB."""
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    custom_embedding_function = SentenceTransformerEmbeddingFunction(
        model=embedding_model,
        model_name=embedding_model_name
    )

    collection = client.get_or_create_collection(
        name="call_metadata_collection",
        embedding_function=custom_embedding_function
    )

    if collection.count() == 0 or len(ids) != collection.count():
        print("Clearing existing documents in ChromaDB and adding new ones...")
        try:
            current_ids_in_db = collection.get(ids=collection.get()['ids'])['ids']
            if current_ids_in_db:
                collection.delete(ids=current_ids_in_db)
        except Exception as e:
            print(f"Could not clear ChromaDB collection (might be empty or other issue): {e}")

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"Added {collection.count()} documents to ChromaDB.")
    else:
        print(f"ChromaDB already contains {collection.count()} documents. Skipping re-population.")

    return collection

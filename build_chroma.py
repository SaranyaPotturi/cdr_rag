# build_chroma.py

import os, json, uuid
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "chroma_db"
ES_DATA_PATH = "es_data.json"
COLLECTION_NAME = "cdr_semantic_search"

print("\n--- Setting up Local Semantic RAG Pipeline ---")

# Delete old DB
if os.path.exists(CHROMA_DB_PATH):
    print(f"Removing existing ChromaDB at {CHROMA_DB_PATH}...")
    import shutil
    shutil.rmtree(CHROMA_DB_PATH)

# Load data
print(f"Loading CDR data from {ES_DATA_PATH}...")
with open(ES_DATA_PATH, 'r') as f:
    data = json.load(f)

records = []
seen_ids = set()

for i, hit in enumerate(data.get('hits', {}).get('hits', [])):
    source = hit.get('_source', {})
    fields = source.get('fields', {})
    comm_rec = fields.get('communicationRecord', {})

    # Base ID from comUUID or _id
    base_id = comm_rec.get('comUUID') or hit.get('_id') or str(uuid.uuid4())

    # Ensure uniqueness by appending a suffix if needed
    final_id = base_id
    suffix = 1
    while final_id in seen_ids:
        final_id = f"{base_id}_{suffix}"
        suffix += 1
    seen_ids.add(final_id)

    # Minimal text for testing
    text = f"Communication on platform {comm_rec.get('platform', {}).get('name', 'N/A')} - Type: {comm_rec.get('comType', 'N/A')} - Direction: {comm_rec.get('direction', 'N/A')}"

    records.append({
        "id": str(final_id),
        "document": text,
        "metadata": {
            "platform": comm_rec.get('platform', {}).get('name', 'N/A'),
            "com_type": comm_rec.get('comType', 'N/A'),
            "direction": comm_rec.get('direction', 'N/A'),
        }
    })

print(f"Prepared {len(records)} documents.")

# Init embedding
print("Loading embedding model...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Init Chroma
print("Creating ChromaDB client...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# Insert in batches
print("Adding documents to ChromaDB...")
batch_size = 500
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    collection.upsert(
        documents=[r['document'] for r in batch],
        ids=[r['id'] for r in batch],
        metadatas=[r['metadata'] for r in batch]
    )
    print(f"Added batch {i//batch_size + 1}")

print(f"ChromaDB setup complete. Total records: {collection.count()}")


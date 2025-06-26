import json
import pandas as pd
from pprint import pprint


def load_es_data(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def sanitize_metadata(metadata: dict) -> dict:
    """
    Recursively clean metadata by removing any keys with None or unsupported types.
    Supported types: str, int, float, bool
    """
    allowed_types = (str, int, float, bool)
    cleaned = {}

    for k, v in metadata.items():
        if isinstance(v, allowed_types):
            cleaned[k] = v
        elif isinstance(v, dict):
            nested_clean = sanitize_metadata(v)
            if nested_clean:
                cleaned[k] = nested_clean
        elif isinstance(v, list):
            # Only keep simple, non-None types inside lists
            filtered_list = [item for item in v if isinstance(item, allowed_types)]
            if filtered_list:
                cleaned[k] = filtered_list
        else:
            print(f"⚠️ Skipping key '{k}' with unsupported type: {type(v).__name__}")

    return cleaned


def prepare_documents_for_chroma_and_df(records):
    all_documents = []
    all_metadatas = []
    valid_records = []

    for idx, record in enumerate(records):
        text = record.get("text", "").strip()
        metadata = record.get("metadata", {})

        if not text:
            continue

        clean_metadata = sanitize_metadata(metadata)
        clean_metadata["id"] = f"doc_{idx}"

        all_documents.append(text)
        all_metadatas.append(clean_metadata)
        valid_records.append({**record, "metadata": clean_metadata})

    df = pd.DataFrame(valid_records)

    print(f"✅ Loaded {len(all_documents)} valid documents.")

    # Debug: print first 3 cleaned metadatas
    pprint(all_metadatas[:3])

    # Check for any remaining invalid metadata
    for i, meta in enumerate(all_metadatas):
        if any(v is None for v in meta.values()):
            print(f"❌ Invalid metadata at index {i}:")
            pprint(meta)

    return all_documents, all_metadatas, df
def load_structured_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_documents_and_metadata():
    """Loads raw docs + metadata for Chroma indexing."""
    data = load_es_data("sentence_data.json")  # Correct relative path
  # update path if needed
    docs, metadata, _ = prepare_documents_for_chroma_and_df(data)
    return docs, metadata

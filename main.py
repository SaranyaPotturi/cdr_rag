# main.py
from datetime import datetime
import json
import pandas as pd
from model_setup import initialize_models
from chroma_setup import initialize_chromadb
from rag_components import (
    load_es_data,
    prepare_documents_for_chroma_and_df,
    classify_query_intent,
    generate_aggregation_query,
    execute_aggregation_query,
    synthesize_aggregation_result,
    SCHEMA_FIELDS,
    ES_DATA_FILE,
    is_meaningless_query,
    process_query 
)

# Semantic search function using ChromaDB and embedding model
# Returns list of (doc, score, metadata)
def semantic_search_fn_chroma(user_query, chroma_collection, embedding_model):
    query_embedding = embedding_model.encode(user_query).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
    )
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    scores = results['distances'][0] if results['distances'] else []
    # Lower score = closer match for cosine, so invert for ranking
    return list(zip(docs, [1-s for s in scores], metadatas))


def rag_pipeline(user_query, df_all_records, chroma_collection, llm_pipeline, embedding_model, metadatas, cache=None):
    # Unified entry point: handles semantic, aggregation, and mixed queries
    return process_query(
        user_query=user_query,
        df=df_all_records,
        documents=None,  # Not used for ChromaDB search
        metadatas=metadatas,
        llm_pipeline=llm_pipeline
    )

if __name__ == "__main__":
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("Loading and preparing data...")
    es_raw_hits = load_es_data(ES_DATA_FILE)
    if not es_raw_hits:
        print("No data loaded. Exiting.")
        exit()

    documents, metadatas, ids, df_all_records = prepare_documents_for_chroma_and_df(es_raw_hits)
    embedding_model, llm_pipeline, _, _ = initialize_models()
    chroma_collection = initialize_chromadb(documents, metadatas, ids, embedding_model, "sentence-transformers")

    while True:
        user_input = input("\nEnter your query: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        response = rag_pipeline(user_input, df_all_records, chroma_collection, llm_pipeline, embedding_model, metadatas)
        print(f"\nResponse:\n{response}\n{'-'*80}")

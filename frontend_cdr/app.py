# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rag_pipeline
from rag_components import load_es_data, prepare_documents_for_chroma_and_df, ES_DATA_FILE
from model_setup import initialize_models
from chroma_setup import initialize_chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# --- Global objects for pipeline ---
es_raw_hits = load_es_data(ES_DATA_FILE)
documents, metadatas, ids, df_all_records = prepare_documents_for_chroma_and_df(es_raw_hits)
embedding_model, llm_pipeline, _, embedding_model_name = initialize_models()
chroma_collection = initialize_chromadb(documents, metadatas, ids, embedding_model, embedding_model_name)

# --- Simple in-memory cache for LLM and retrieval results ---
query_cache = {}

@app.post("/query")
async def handle_query(data: QueryRequest):
    query = data.query
    print(f"[API] Received query: {query}")
    print("[API] Calling rag_pipeline...")
    answer = rag_pipeline(
        query,  # Pass the correct variable
        df_all_records,
        chroma_collection,
        llm_pipeline,
        embedding_model,
        metadatas
    )
    print(f"[API] Final answer: {answer}")
    return answer
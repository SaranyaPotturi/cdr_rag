# rag_core.py

import json
import os
from semantic_pipeline import semantic_search
from aggregation_pipeline import aggregate_cdrs
from query_classification import classify_query

# === Load the preprocessed CDRs from sentence_data.json ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCE_DATA_PATH = os.path.join(BASE_DIR, "sentence_data.json")

with open(SENTENCE_DATA_PATH, "r") as f:
    preprocessed_cdrs = json.load(f)

# === Function to filter relevant structured CDRs ===
def filter_relevant_structured_cdrs(query: str, cdrs: list) -> list:
    """
    Very basic filter for relevant CDRs based on keyword presence.
    For now, returns all CDRs. You can improve this with proper filters.
    """
    query_keywords = query.lower().split()
    filtered = []

    for cdr in cdrs:
        cdr_text = json.dumps(cdr).lower()
        if any(kw in cdr_text for kw in query_keywords):
            filtered.append(cdr)

    return filtered

# === Main RAG system class ===
class RAGSystem:
    def __init__(self):
        self.cdr_data = preprocessed_cdrs  # sentence_data.json

    def process_query(self, query: str) -> dict:
        query_type = classify_query(query)

        if query_type == "semantic":
            result = semantic_search(query)
            return {
                "answer": result["answer"],
                "sources": result["sources"]
            }

        elif query_type == "aggregation":
            structured_results = filter_relevant_structured_cdrs(query, self.cdr_data)
            answer = aggregate_cdrs(structured_results, query)
            return {
                "answer": answer,
                "sources": structured_results  # optional: can be removed
            }

        else:
            return {
                "answer": "Sorry, I couldn't understand your query.",
                "sources": []
            }

# === Export the instance ===
rag_system_instance = RAGSystem()

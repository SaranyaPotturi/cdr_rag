# main.py

# No need to import transformers or torch since this version is rule-based
# Just keep it clean and simple

def classify_query(query: str):
    semantic_keywords = ["details", "list", "show", "records", "missed", "who", "when"]
    aggregation_keywords = ["how many", "total", "count", "number of", "sum"]

    query_lower = query.lower()

    if any(word in query_lower for word in aggregation_keywords):
        return "aggregation"
    elif any(word in query_lower for word in semantic_keywords):
        return "semantic"
    else:
        return "semantic"  # fallback


# ✅ CLI testing
if __name__ == "__main__":
    query = input("Enter a query: ")
    label = classify_query(query)
    print("This query is classified as:", label)

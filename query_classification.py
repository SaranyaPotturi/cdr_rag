# query_classification.py

def classify_query(query: str) -> str:
    """
    A simple keyword-based classifier for demo purposes.
    You can replace this later with an ML model or fine-tuned logic.
    """
    aggregation_keywords = ["how many", "total", "average", "count", "maximum", "minimum", "number of"]

    if any(keyword in query.lower() for keyword in aggregation_keywords):
        return "aggregation"
    else:
        return "semantic"

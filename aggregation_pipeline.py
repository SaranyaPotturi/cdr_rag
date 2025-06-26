# aggregation_pipeline.py

import pandas as pd
from typing import List, Dict

def aggregate_cdrs(cdrs: List[Dict], query: str) -> str:
    """
    Perform basic aggregation over structured CDRs.
    Currently supports: count-based queries.
    """

    query = query.lower()
    df = pd.DataFrame(cdrs)

    if df.empty:
        return "No relevant call records found."

    # Handle count-based queries
    if any(kw in query for kw in ["how many", "count", "total", "number of"]):
        call_count = len(df)
        return f"There were {call_count} calls matching your query."

    # Future: Add average duration, max call time, etc.
    return "Aggregation type not supported yet."


# chart_data_sample.py
# Example FastAPI endpoint to generate and return Chart.js-compatible chart_data for the frontend

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query = body.get("query", "")
    # Generate chart data based on the query
    if "duration" in query.lower():
        chart_type = "bar"
        labels = ["Device A", "Device B", "Device C"]
        data_points = [120, 90, 60]
        label = "Call Duration (min)"
    elif "hour" in query.lower():
        chart_type = "line"
        labels = [str(h) + ":00" for h in range(24)]
        data_points = [abs(50 + 30 * ((h-12)/12)**2 - 2*h) for h in range(24)]
        label = "Calls per Hour"
    elif "trend" in query.lower():
        chart_type = "line"
        labels = [f"Day {i+1}" for i in range(7)]
        data_points = [60 + 10*i + (i%2)*15 for i in range(7)]
        label = "Call Trend"
    else:
        chart_type = "pie"
        labels = ["Category 1", "Category 2", "Category 3"]
        data_points = [40, 35, 25]
        label = "Sample Distribution"

    chart_data = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": [{
                "label": label,
                "data": data_points,
                "backgroundColor": ["#e63946", "#f1c40f", "#457b9d", "#a8dadc", "#22223b", "#f7b801", "#43aa8b"][:len(labels)]
            }]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"display": True}
            }
        }
    }
    return JSONResponse({
        "llm_summary": f"Chart generated for query: {query}",
        "chart_data": chart_data
    })

# To run: uvicorn frontend_cdr.chart_data_sample:app --reload

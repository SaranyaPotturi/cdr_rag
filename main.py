# main.py
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rag_core import rag_system_instance

app = FastAPI(
    title="CDR GenAI Hybrid RAG Backend",
    description="Backend APIs for hybrid CDR analysis using ChromaDB and Neo4j",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
async def on_startup():
    try:
        logger.info("✅ RAGSystem ready. No async init required.")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise RuntimeError(f"Startup failed: {str(e)}")

@app.post("/query")
async def query_handler(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return {"error": "Query parameter is required"}
    result = rag_system_instance.process_query(query)
    return result

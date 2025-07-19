A modern dashboard for Call Detail Records (CDR) analytics, combining semantic search, aggregation queries, and dynamic charting using Python, Pandas, ChromaDB, and an LLM (Mistral via Ollama).

## Features

- **Semantic Search:** Find relevant call records using natural language queries.
- **Aggregation Analytics:** Get counts, trends, and statistics (e.g., total calls, average duration, calls per day/month/hour).
- **Dynamic Chart Generation:** Visualize results as charts (bar, line, etc.) returned as base64 images.
- **LLM Integration:** Uses Mistral (via Ollama) for query intent classification and natural language summaries.
- **Frontend Dashboard:** Modern UI with responsive design.

## Technologies Used

- Python 3.x
- Pandas
- ChromaDB
- Sentence Transformers (MiniLM)
- Matplotlib
- Ollama (Mistral LLM)
- HTML/CSS (frontend)

## Folder Structure

```
├── build_chroma.py
├── check.py
├── chroma_setup.py
├── es_data.json
├── main.py
├── model_setup.py
├── rag_components.py
├── requirements.txt
├── data/
│   └── sentence_data.json
├── frontend_cdr/
│   ├── app.py
│   ├── backup.html
│   ├── backupstyles.css
│   ├── chart_data_sample.py
│   ├── index.html
│   ├── main.py
│   └── style.css
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repo-url>
   cd cdr_rag
   ```
2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Install Ollama and run Mistral model:**
   - [Ollama installation guide](https://ollama.com/download)
   - Start Ollama server and pull Mistral:
     ```
     ollama pull mistral
     ollama run mistral
     ```
4. **Run the backend:**
   ```
   python main.py
   ```
5. **Run the frontend:**
   - Open `frontend_cdr/index.html` in your browser.

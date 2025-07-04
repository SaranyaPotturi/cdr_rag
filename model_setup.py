# model_setup.py
from sentence_transformers import SentenceTransformer
import requests

def ollama_generate(prompt, model='mistral', max_new_tokens=60, temperature=0.1, **kwargs):
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': max_new_tokens,
            'temperature': temperature
        }
    }
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()['response'].strip()

def initialize_models():
    print("Loading embedding model (MiniLM)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Using Ollama for LLM (Mistral)")
    return embedding_model, ollama_generate, None, "sentence-transformers"

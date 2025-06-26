import json
import os
from typing import List, Dict

from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# === Set HuggingFace API token ===
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your actual HF token

# === Load your HuggingFace model via LangChain ===
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

# === Define custom prompt ===
prompt_template = """
You are an expert CDR analyst. Use the below call records to answer the user's question.

Call Records:
{context}

Question:
{question}

Answer concisely and only based on the records.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# === Load the preprocessed sentence data ===
def load_documents(filepath: str) -> List[Document]:
    with open(filepath, "r") as f:
        data = json.load(f)

    documents = [Document(page_content=sentence) for sentence in data]
    return documents

# === Set up vector database using Chroma ===
def create_chroma_index(documents: List[Document]) -> Chroma:
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding_function, persist_directory="chroma_db")
    return vectordb

# === Perform semantic search and return answer ===
def semantic_search(question: str) -> Dict:
    # Load docs and build index
    docs = load_documents("sentence_data.json")
    vectordb = create_chroma_index(docs)

    # Perform similarity search
    relevant_docs = vectordb.similarity_search(question, k=5)

    # Load QA chain and run
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    answer = chain.run(input_documents=relevant_docs, question=question)

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in relevant_docs]
    }

# RAG-Based Question Answering System

A professional AI system built with **FastAPI**, **Pinecone**, and **Groq (Llama 3.3)** to ask questions and get replies with your documents.

## Tech Stack
- **Backend:** FastAPI
- **Vector DB:** Pinecone (Serverless)
- **Embeddings:** `all-MiniLM-L6-v2` (Local via Sentence-Transformers)
- **LLM:** Llama 3.3 70B via Groq API
- **File Parsing:** PyPDF

## Setup Instructions
1. **Clone the repo:** `git clone [Your Link]`
2. **Create Environment:** `python3 -m venv venv && source venv/bin/activate`
3. **Install Tools:** `pip install -r requirements.txt`
4. **Environment Variables:** Create a `.env` file:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME` (Must be **384 dimensions**)
   - `GROQ_API_KEY`
5. **Run:** `python3 -m uvicorn app.main:app --reload`

##  Usage
- Open `http://127.0.0.1:8000` for the UI.
- Click **Clear Memory** to start fresh.
- Upload a PDF and wait for the terminal to say `SUCCESS`.
- Ask your question and get an AI-generated answer.
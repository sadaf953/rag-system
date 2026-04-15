import os
import uuid
import time
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

load_dotenv()

# --- INITIALIZATION ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
model = SentenceTransformer('all-MiniLM-L6-v2') 
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ingest_document_to_pinecone(file_path: str, filename: str):
    try:
        print(f"\n STARTING PROCESSING: {filename}")
        time.sleep(1) 
        
        text = ""
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            print("ERROR: No text found in document!")
            return

        # Split into chunks
        chunks = [text[i:i+600] for i in range(0, len(text), 500)]
        
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            vectors.append({
                "id": f"{filename}_{i}_{str(uuid.uuid4())[:8]}", 
                "values": embedding, 
                "metadata": {"text": chunk, "filename": filename}
            })
        
        # Upload
        index.upsert(vectors=vectors)
        
        print("******************************************")
        print(f"SUCCESS: {filename} IS FULLY PROCESSED!")
        print(f"Total Chunks Saved: {len(vectors)}")
        print("******************************************\n")

    except Exception as e:
        print(f"INGESTION CRASHED: {e}")

async def retrieve_and_generate_answer(question: str):
    try:
        # --- TRACK RETRIEVAL TIME ---
        start_retrieval = time.time() 
        
        query_vector = model.encode(question).tolist()
        results = index.query(
            vector=query_vector, 
            top_k=5, 
            include_metadata=True
        )
        
        contexts = [match['metadata']['text'] for match in results['matches'] if match['score'] > 0.1]
        
        retrieval_time = time.time() - start_retrieval
        print(f"RETRIEVAL (Pinecone): {retrieval_time:.4f} seconds") # <--- This prints the time

        if not contexts:
            return "No relevant info found."

        # --- TRACK GENERATION TIME ---
        start_gen = time.time() 

        context_str = "\n---\n".join(contexts)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer using the context."},
                {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {question}"}
            ]
        )
        
        gen_time = time.time() - start_gen
        print(f"GENERATION (Groq): {gen_time:.4f} seconds") # <--- This prints the time
        
        print(f"TOTAL QUERY TIME: {retrieval_time + gen_time:.4f} seconds\n")

        return completion.choices[0].message.content

    except Exception as e:
        print(f"ERROR: {e}")
        return "Error occurred."

async def clear_all_knowledge():
    try:
        # This deletes every single vector in your index
        index.delete(delete_all=True)
        print("Pinecone Index wiped clean.")
        return True
    except Exception as e:
        print(f"Error clearing index: {e}")
        return False
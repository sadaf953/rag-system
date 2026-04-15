import os
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse

# Make sure to import clear_all_knowledge here
from app.rag_processor import ingest_document_to_pinecone, retrieve_and_generate_answer, clear_all_knowledge
from app.models import QueryRequest, DocumentUploadResponse, QueryResponse

# --- FastAPI App Setup ---
app = FastAPI(
    title="RAG-Based Question Answering System",
    description="API to upload documents and ask questions using RAG with Pinecone and Groq.",
    version="1.0.0"
)

# --- Configuration ---
UPLOAD_DIR = Path("data") # Where temporary uploaded files are saved
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# --- HTML UI Interface ---
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Document AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/api/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1c1e21; }
            .container { max-width: 800px; margin-top: 60px; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
            .header { border-bottom: 2px solid #f0f2f5; margin-bottom: 30px; padding-bottom: 20px; }
            .section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 15px; color: #4b4f56; }
            .btn-upload { background-color: #007bff; color: white; border: none; }
            .btn-ask { background-color: #28a745; color: white; border: none; width: 100%; padding: 12px; font-weight: bold; }
            .btn-clear { background-color: #dc3545; color: white; border: none; font-size: 0.8rem; padding: 5px 15px; border-radius: 5px; }
            #answerBox { display: none; margin-top: 25px; padding: 20px; background-color: #f8f9fa; border-left: 5px solid #28a745; border-radius: 8px; }
            .status-msg { font-size: 0.85rem; margin-top: 8px; font-weight: 500; }
            .loading-spinner { display: none; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #28a745; border-radius: 50%; animation: spin 1s linear infinite; margin: 10px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header text-center position-relative">
                <h1>Document AI</h1>
                <p class="text-muted">Upload a paper and get instant answers</p>
                <button class="btn btn-clear position-absolute top-0 end-0" onclick="clearMemory()">🗑️ Clear AI Memory</button>
            </div>

            <!-- UPLOAD SECTION -->
            <div class="mb-5">
                <div class="section-title">Step 1: Ingest Document</div>
                <div class="input-group">
                    <input type="file" id="fileInput" class="form-control" accept=".pdf,.txt">
                    <button class="btn btn-upload" type="button" onclick="uploadFile()">Upload Now</button>
                </div>
                <div id="uploadStatus" class="status-msg"></div>
            </div>

            <!-- ASK SECTION -->
            <div>
                <div class="section-title">Step 2: Ask Anything</div>
                <textarea id="questionInput" class="form-control mb-3" rows="3" placeholder="e.g., Give me a detailed summary of this document."></textarea>
                <button id="askBtn" class="btn btn-ask" onclick="askQuestion()">Get Answer</button>
                <div id="loading" class="loading-spinner"></div>
            </div>

            <!-- ANSWER DISPLAY -->
            <div id="answerBox">
                <h6 class="text-uppercase text-muted small fw-bold">AI Response:</h6>
                <p id="answerText" class="mb-0"></p>
            </div>
        </div>

        <script>
            async function clearMemory() {
                if(!confirm("This will delete all previous document knowledge. Are you sure?")) return;
                try {
                    const response = await fetch('/clear', { method: 'POST' });
                    const data = await response.json();
                    alert(data.message);
                    location.reload();
                } catch (e) {
                    alert("Error clearing memory.");
                }
            }

            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const status = document.getElementById('uploadStatus');
                if (!fileInput.files[0]) { alert("Please select a file first!"); return; }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                status.style.color = "#007bff";
                status.innerText = "⏳ Processing document... please watch the terminal.";

                try {
                    const response = await fetch('/upload-document', { method: 'POST', body: formData });
                    const data = await response.json();
                    status.style.color = "#28a745";
                    status.innerText = data.message;
                } catch (e) {
                    status.style.color = "#dc3545";
                    status.innerText = "Upload failed.";
                }
            }

            async function askQuestion() {
                const question = document.getElementById('questionInput').value;
                const answerBox = document.getElementById('answerBox');
                const answerText = document.getElementById('answerText');
                const loading = document.getElementById('loading');
                const askBtn = document.getElementById('askBtn');

                if (!question) { alert("Please enter a question!"); return; }

                loading.style.display = "block";
                answerBox.style.display = "none";
                askBtn.disabled = true;

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    const data = await response.json();
                    
                    loading.style.display = "none";
                    askBtn.disabled = false;
                    answerBox.style.display = "block";
                    answerText.innerText = data.answer;
                } catch (e) {
                    loading.style.display = "none";
                    askBtn.disabled = false;
                    alert("Error getting answer. Ensure document is processed.");
                }
            }
        </script>
    </body>
    </html>
    """

# --- BACKEND ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    app.state.background_tasks = BackgroundTasks()

@app.on_event("startup")
async def startup_event():
    print("✨ Server is starting up...")

@app.post("/clear")
async def clear_index():
    success = await clear_all_knowledge()
    if success:
        return {"message": "AI memory cleared! You can now start with a fresh document."}
    raise HTTPException(status_code=500, detail="Failed to clear memory.")

@app.post("/upload-document", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed_content_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT allowed.")
    
    file_location = UPLOAD_DIR / file.filename
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # We pass background_tasks directly into the function
        background_tasks.add_task(ingest_document_to_pinecone, str(file_location), file.filename)
        
        return DocumentUploadResponse(
            message=f"Processing '{file.filename}'... Watch your terminal for progress.",
            filename=file.filename
        )
    except Exception as e:
        if file_location.exists(): os.remove(file_location)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    answer = await retrieve_and_generate_answer(query.question)
    return QueryResponse(question=query.question, answer=answer)
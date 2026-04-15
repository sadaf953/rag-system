from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, description="The question to ask based on the documents.")

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    # You could add retrieved_contexts: List[str] here for debugging
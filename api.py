from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import run_rag

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str

# Health check
@app.get("/")
def home():

    return {
        "message": "MediGest RAG API Running"
    }

# Main RAG endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):

    response = run_rag(request.query)

    return {
        "query": request.query,
        "response": response
    }
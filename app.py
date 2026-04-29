# FastAPI endpoint for RAG pipeline
# basic version working, will add streaming later
# TODO: add auth middleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingest import load_documents, split_documents
from retrieve import create_vectorstore, retrieve_docs
from generate import build_prompt, generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    file_path: str

class QueryResponse(BaseModel):
    answer: str
    sources: int

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        # load and chunk docs
        docs = load_documents(request.file_path)
        chunks = split_documents(docs)

        # retrieve relevant chunks
        vectorstore = create_vectorstore(chunks)
        context_docs = retrieve_docs(vectorstore, request.question)
        context = "\n".join([d.page_content for d in context_docs])

        # generate answer
        prompt = build_prompt()
        answer = generate_answer(prompt, context, request.question)

        return QueryResponse(answer=answer, sources=len(context_docs))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-pipeline"}

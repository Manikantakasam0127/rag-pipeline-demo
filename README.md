# RAG Pipeline Demo

Built this as a clean reference implementation of a 
RAG system for enterprise document question answering.

Wanted a reusable baseline I could iterate on — 
the kind of thing that takes a week to get right 
in production but is worth having as a solid foundation.

## What it does

Takes documents, chunks them, embeds them into a 
FAISS vector index, retrieves relevant context, 
and passes it to GPT-4o for generation. 
Added a RAGAS evaluation layer to track 
faithfulness and relevancy scores over time.

## Tech Stack

- Python, LangChain, FAISS, Azure OpenAI (GPT-4o)
- FastAPI for serving
- RAGAS for evaluation
- Docker for containerization

## Files

- ingest.py — document loading and chunking
- retrieve.py — FAISS vector search
- generate.py — LLM generation with prompt template
- app.py — FastAPI REST endpoint
- evaluate.py — RAGAS scoring framework
- Dockerfile — containerized deployment

## How to run

pip install -r requirements.txt
uvicorn app:app --reload

## API

POST /query
{
  "question": "what is the refund policy?",
  "file_path": "docs/sample.pdf"
}

GET /health

## Key learnings

- chunk_size=500 worked better than 1000 for our docs
- k=5 retrieval gave more complete context than k=3
- GPT-4o gave noticeably better faithfulness 
  scores than GPT-3.5 on domain-specific questions
- RAGAS evaluation caught regressions early 
  before they hit production

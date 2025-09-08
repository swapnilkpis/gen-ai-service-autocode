from fastapi import APIRouter
from schemas.request import IngestRequest, QueryRequest
from services.rag_service import embed_and_store, query_data, ask_question

router = APIRouter()

@router.post("/embed-and-store")
def embed(req: IngestRequest):
    return embed_and_store(req)

@router.post("/query")
def query(req: QueryRequest):
    return query_data(req)

@router.post("/ask")
def ask(req: QueryRequest):
    return ask_question(req)

from fastapi import FastAPI
from routes.rag import router as rag_router

app = FastAPI(
    title="Qdrant RAG Service",
    description="RAG service with dynamic LLM and Qdrant integration",
    version="1.0.0"
)

# Register routes from controller
app.include_router(rag_router, prefix="/api")

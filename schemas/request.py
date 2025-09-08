from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    text: str
    metadata: Optional[dict] = {}
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    provider: str = "groq"
    api_key: str
    model: str = "mixtral-8x7b-32768"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str

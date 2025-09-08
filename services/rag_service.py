from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from embedding.model import embedding_model
from llms.factory import call_llm
import uuid

def embed_and_store(req):
    qdrant = QdrantClient(host=req.qdrant_host, port=req.qdrant_port)

    try:
        qdrant.get_collection(req.collection_name)
    except:
        qdrant.recreate_collection(
            collection_name=req.collection_name,
            vector_size=384,
            distance="Cosine"
        )

    vector = embedding_model.encode(req.text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=req.metadata | {"text": req.text}
    )
    qdrant.upsert(req.collection_name, points=[point])
    return {"status": "stored", "id": point.id}

def query_data(req):
    qdrant = QdrantClient(host=req.qdrant_host, port=req.qdrant_port)
    query_vector = embedding_model.encode(req.question).tolist()
    search_result = qdrant.search(
        collection_name=req.collection_name,
        query_vector=query_vector,
        limit=req.top_k
    )
    context = "\n".join([hit.payload.get("text", "") for hit in search_result])
    return {"context": context}

def ask_question(req):
    qdrant = QdrantClient(host=req.qdrant_host, port=req.qdrant_port)
    query_vector = embedding_model.encode(req.question).tolist()
    search_result = qdrant.search(
        collection_name=req.collection_name,
        query_vector=query_vector,
        limit=req.top_k
    )
    context = "\n".join([hit.payload.get("text", "") for hit in search_result])

    prompt = f"""Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {req.question}\nAnswer:"""

    answer = call_llm(req.provider, req.api_key, req.model, prompt)
    return {"answer": answer, "context": context}

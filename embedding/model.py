from sentence_transformers import SentenceTransformer

# Load the embedding model once during startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "app/rag/faiss_index.index"
DOCS_PATH = "app/rag/faiss_index.pkl"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load stored documents
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve_docs(query, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)

    results = [documents[i] for i in indices[0] if i < len(documents)]
    return results

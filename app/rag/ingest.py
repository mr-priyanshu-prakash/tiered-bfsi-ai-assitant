import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_PATH = "knowledge_base"
INDEX_PATH = "app/rag/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def build_index():
    model = SentenceTransformer(MODEL_NAME)
    docs = load_documents()

    embeddings = model.encode(docs)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_PATH}.index")

    with open(f"{INDEX_PATH}.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("FAISS index created")


if __name__ == "__main__":
    build_index()

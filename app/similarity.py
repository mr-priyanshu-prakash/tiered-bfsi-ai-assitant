import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bfsi_alpaca_expanded.json")

SIMILARITY_THRESHOLD = 0.85

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

instructions = [item["instruction"] for item in dataset]
embeddings = model.encode(instructions, convert_to_numpy=True)

print("Similarity layer ready.")

def check_similarity(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        return dataset[best_idx]["output"], best_score

    return None, best_score

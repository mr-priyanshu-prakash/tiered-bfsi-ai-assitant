A multi-tier AI assistant for BFSI customer support that runs fully locally using semantic search, a lightweight LLM, and Retrieval-Augmented Generation (RAG).This system intelligently routes queries through three decision layers to balance latency, cost, and accuracy.

Key Features:
	•Instant FAQ responses via semantic similarity
	•On-device reasoning with TinyLlama (1.1B)
	•Grounded answers using RAG + FAISS
	•Fully offline — no external API calls
	•Optimized for low-resource machines (M1, 8GB RAM)

Architecture:

User Query
   │
   ▼
Embedding (MiniLM)
   │
   ▼
Similarity Check
   ├── Tier 1: FAQ Retrieval
   ├── Tier 2: TinyLlama Response
   └── Tier 3: RAG (FAISS + Context)
   │
   ▼
Final Answer + Confidence + Tier

System Design:

Tier 1 — FAQ Semantic Retrieval
	•	Model: sentence-transformers/all-MiniLM-L6-v2
	•	Purpose: Fast intent matching
	•	Threshold: 0.75

Returns stored answer without generation.


Tier 2 — Small Language Model
	•	Model: TinyLlama-1.1B-Chat (GGUF Q4)
	•	Engine: llama-cpp-python
	•	Purpose: Generate responses for general queries
	•	Threshold: 0.45 – 0.75

Optimized for low latency local inference.


Tier 3 — Retrieval-Augmented Generation
	•	Vector Store: FAISS
	•	Embeddings: MiniLM
	•	LLM: TinyLlama

Used when query requires policy or knowledge grounding.


Tech Stack:
	•Python 3.10
	•Sentence Transformers
	•FAISS
	•llama-cpp-python
	•NumPy / scikit-learn


Project Structure:

BFSI_AI_ASSISTANT
│
├── app
│   ├── pipeline.py
│   ├── similarity.py
│   ├── slm.py
│   └── rag
│       ├── ingest.py
│       ├── retrieve.py
│       └── generate.py
│
├── dataset
├── knowledge_base
├── models
└── run_local.py

Running Locally:

1. Create environment:

	conda create -n bfsi_ai python=3.10
	conda activate bfsi_ai

2. Install dependencies:

	pip install -r requirements.txt

3. Build vector index:

	python app/rag/ingest.py

4. Start assistant:

	python run_local.py

Decision Logic:

Similarity Score	Action
≥ 0.75			FAQ response
0.45 – 0.75		TinyLlama
< 0.45			RAG pipeline



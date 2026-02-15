from app.slm import llm
from app.rag.retrieve import retrieve_docs

SYSTEM_PROMPT = "You are a BFSI assistant. Answer using the context."

def generate_rag_response(query):

    docs = retrieve_docs(query, top_k=3)
    context = "\n".join(docs)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

User: {query}
Assistant:"""

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        stop=["User:", "</s>"]
    )

    return output["choices"][0]["text"].strip()

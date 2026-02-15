from app.similarity import check_similarity
from app.slm import generate_response
from app.rag.generate import generate_rag_response

SIM_THRESHOLD = 0.75
SLM_THRESHOLD = 0.45

def run_pipeline(query):

    faq_answer, score = check_similarity(query)

    # Tier 1 — FAQ match
    if faq_answer is not None and score >= SIM_THRESHOLD:
        return {
            "tier": "faq",
            "response": faq_answer,
            "confidence": score
        }
    # Tier 2 — SLM answer
    if score >= SLM_THRESHOLD:
        response = generate_response(query)
        return {
            "tier": "slm",
            "response": response,
            "confidence": score
        }
    # Tier 3 — RAG fallback
    response = generate_rag_response(query)

    return {
        "tier": "rag",
        "response": response,
        "confidence": score
    }

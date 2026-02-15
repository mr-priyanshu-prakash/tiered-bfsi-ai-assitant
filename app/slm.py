from llama_cpp import Llama
import os

SYSTEM_PROMPT = """
You are a BFSI banking assistant.

Provide a clear, concise answer to the user's question.
Do not generate another question.
Do not continue the conversation.
Keep the answer under 5 sentences.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

print("Loading GGUF model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=0 # Adjust it based on your GPU capabilities
)

print("SLM ready.")

def generate_response(query, max_tokens=200):
    
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        stop=["User:", "###", "</s>"]
    )

    text = output["choices"][0]["text"].strip()

    for phrase in [
        "Provide a clear",
        "Do not generate",
        "Keep the answer",
        "Do not continue"
    ]:
        if phrase in text:
            text = text.split(phrase)[0].strip()

    return text

from typing import List, Dict
import google.generativeai as genai
from app.core.config import settings

# Configure once at import time
genai.configure(api_key=settings.google_api_key)
_model = genai.GenerativeModel("gemini-2.5-flash")


# ── Prompt template ──────────────────────────────────────────────────────────
# This is called a "grounded" prompt — the LLM is explicitly told to
# ONLY use the provided context, not its training data.
# The refusal instruction ("I couldn't find the answer") is what reduces
# hallucination. In Phase 4 you'll make this template configurable.
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the provided context.

Context:
{context}

Question: {question}

Rules:
- Answer ONLY using information from the context above.
- If the answer is not present in the context, respond with: "I couldn't find the answer in the provided documents."
- Be concise and factual. Do not speculate or add information from outside the context.
- If the context is relevant but incomplete, say what you found and note the limitation.

Answer:"""


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """
    Build a grounded prompt from retrieved chunks and call the LLM.

    The chunks are separated by a divider so the model can clearly
    see where one chunk ends and another begins.
    """
    context = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = _model.generate_content(prompt)
    return response.text
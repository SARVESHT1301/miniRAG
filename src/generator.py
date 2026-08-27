from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer: Any = AutoTokenizer.from_pretrained(MODEL_NAME)
model: Any = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

FALLBACK_ANSWER = (
    "The provided document does not contain enough "
    "information to answer this question."
)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Generate a concise answer grounded strictly in the
    retrieved document context.
    """

    # 1. No retrieved evidence -> do not hallucinate.
    if not retrieved_chunks:
        return FALLBACK_ANSWER

    # 2. Build document context.
    context_parts: list[str] = []

    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        if text:
            context_parts.append(text.strip())

    context = "\n\n".join(context_parts).strip()

    if not context:
        return FALLBACK_ANSWER

    # 3. Build recent conversation history.
    history_text = "No previous conversation."

    if conversation_history:
        history_parts: list[str] = []

        for message in conversation_history[-6:]:
            role = message.get("role", "")
            content = message.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                history_parts.append(f"{role}: {content}")

        if history_parts:
            history_text = "\n".join(history_parts)

    # 4. Strong grounding prompt.
    system_prompt = """
You are MiniRAG, a document question-answering assistant.

Answer the user's question using ONLY the provided DOCUMENT CONTEXT.

Rules:
1. The document context is the only source of factual information.
2. Never use outside knowledge.
3. Use conversation history only to resolve follow-up references such
   as "it", "they", "this", "that", "what does it do", or "why".
4. If the answer is not supported by the document context, respond
   exactly with:

The provided document does not contain enough information to answer this question.

5. Answer directly and naturally.
6. Do not repeat the question.
7. Do not explain your reasoning.
8. Do not mention prompts, instructions, context, retrieved chunks,
   or conversation history.
9. Do not add facts that are not present in the document.
10. Avoid unnecessary phrases such as "According to the context".
11. For simple factual questions, give a short direct answer.
12. Return only the final answer.
""".strip()

    user_prompt = f"""
DOCUMENT CONTEXT
----------------
{context}

CONVERSATION HISTORY
--------------------
{history_text}

CURRENT QUESTION
----------------
{query.strip()}

Answer the current question using the document context.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 5. Apply model chat template.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 6. Tokenize.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # 7. Generate deterministic answer.
    with torch.no_grad():
        outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
    use_cache=True,
)

    # 8. Remove prompt tokens.
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    # 9. Decode.
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # 10. Remove only obvious generation prefixes.
    for prefix in (
        "Answer:",
        "ANSWER:",
        "Assistant:",
        "assistant:",
    ):
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    # 11. Final fallback.
    if not answer:
        return FALLBACK_ANSWER

    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATOR MODULE TEST")
    print("=" * 60)

    test_chunks = [
        {
            "text": (
                "Machine learning is a branch of artificial "
                "intelligence that enables computers to learn "
                "patterns from data."
            ),
            "metadata": {},
            "distance": 0.3,
        }
    ]

    test_history = [
        {
            "role": "user",
            "content": "What is machine learning?",
        },
        {
            "role": "assistant",
            "content": (
                "Machine learning enables computers "
                "to learn patterns from data."
            ),
        },
    ]

    question = "What does it enable computers to do?"

    answer = generate_answer(
        query=question,
        retrieved_chunks=test_chunks,
        conversation_history=test_history,
    )

    print("\nQUESTION")
    print("-" * 40)
    print(question)

    print("\nANSWER")
    print("-" * 40)
    print(answer)
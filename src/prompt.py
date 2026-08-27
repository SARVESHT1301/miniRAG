def build_prompt(query, retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(chunk["text"])

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Use only information explicitly supported by the provided context.

If the context contains only part of the information needed
to answer the question, answer the supported part and clearly
state what cannot be determined from the provided documents.

Do not use outside knowledge.
Do not make unsupported assumptions.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


if __name__ == "__main__":

    example_chunks = [
        {
            "text": "AWS Lambda is a serverless compute service that allows developers to run code without provisioning or managing servers."
        }
    ]

    query = "What is AWS Lambda?"

    prompt = build_prompt(
        query,
        example_chunks
    )

    print(prompt)
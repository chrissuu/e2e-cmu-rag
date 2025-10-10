"""
Prompt

Prompt generation function.

Intentionally modular.

This will collect chunks and a query and form
the relevant prompt.
"""

def prompt(query, chunks, k):
    chunk_str = "\n\n".join([f"[CHUNK {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    return f"""
You are a factual question-answering assistant.

You will receive a single user query and {k} retrieved document chunks that may contain the answer.
Read them carefully and extract the most relevant information to answer the query directly.

If the answer is not found in the documents, reply with: "Not found."

==================== DOCUMENTS ====================
{chunk_str}
==================== END DOCUMENTS ====================

Query: {query}

Instructions:
- Answer concisely in one sentence or less.
- Do not explain or justify your answer.
- Do not repeat or reference the documents.
- If uncertain or no evidence is present, say "Not found."

Answer:
"""
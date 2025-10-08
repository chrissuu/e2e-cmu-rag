'''
def prompt(query, chunks, k):
    chunk_str = '\n'.join([f"CHUNK {i}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""
You will be a question-answering system for ONE SINGLE query

A retriever will attempt to provide you with relevant documents
to aid in your answering.

Return a short, concise answer with no explanation.

Here are {k} document chunks that have been retrieved by the
retriever to aid you.

============== BEGIN DOCUMENT CHUNKS ==============
{chunk_str}
==============  END DOCUMENT CHUNKS  ==============

Here is the query you will answer:
{query}

REMEMBER: ONLY RETURN THE ANSWER, AND NOTHING ELSE!
========== ANSWER ===========
'''
def prompt(query, chunks, k):
    chunk_str = "\n".join([f"CHUNK {i}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""
You are an extraction model. First, search the context for the answer. If the answer is not in the context, answer from your general knowledge.

OUTPUT FORMAT (must follow all):
- Output **only** the answer text (a single short span). No explanations, no prefixes/suffixes, no quotes, no trailing punctuation.
- Do NOT write strings like "the answer is", "Answer:", or banners.
- Return the **minimal canonical span** (e.g., "UPMC", "1984", "Heinz Hall").
- For dates/times return just the date/time string (e.g., "Aug 17, 2025", "2:30 pm").
- If multiple valid spans exist, choose the most specific and widely accepted one.

EXAMPLES:
Q: What is 2 + 2?            → 4
Q: Who is the top employer?  → UPMC
Q: Where is it held?         → Heinz Hall

============== BEGIN CONTEXT ({k} chunks) ==============
{chunk_str}
==============  END CONTEXT  ==============

Question: {query}
Answer:
""".strip()

"""

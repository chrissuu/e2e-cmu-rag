def prompt(query, chunks, k):
    chunk_str = '\n'.join([f"CHUNK {i}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""
You will be a helpful question-answering system for ONE SINGLE query

Here are some facts that may be relevant for answering the question:
============== BEGIN FACTS ==============
{chunk_str}
==============  END FACTS  ==============
REMEMBER: Return only the answer, in the most concise form possible,
and nothing else.

{query}
"""
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
"""

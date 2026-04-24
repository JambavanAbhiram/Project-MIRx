def build_prompt(query, docs):

    context = "\n\n".join(docs)

    prompt = f"""
You are a medical retrieval assistant.

You are a medical retrieval assistant.

STRICT RULES:
1. Use ONLY retrieved context.
2. Do NOT invent medicines.
3. Do NOT prescribe medication.
4. Mention retrieved medicines only as informational references.
5. If unclear, say:
   "I could not find reliable information."

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""

    return prompt
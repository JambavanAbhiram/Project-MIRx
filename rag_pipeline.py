from retriever import retrieve
from reranker import rerank

from prompt import build_prompt
from generator import generate_response

def run_rag(query):

    # Retrieve more docs initially
    retrieved = retrieve(query, k=20)

    # Re-rank them
    reranked = rerank(
        query,
        retrieved,
        top_k=5
    )

    best_score = reranked[0]["rerank_score"]

    print("\nRERANKED RESULTS:\n")

    for item in reranked:

        print(f"Rank: {item['rank']}")
        print(f"FAISS Distance: {item['distance']:.4f}")
        print(f"Rerank Score: {item['rerank_score']:.4f}")

        print(f"Medicine: {item['name']}")
        print(f"Uses: {item['uses']}")

        print("=" * 80)

    docs = [item["document"] for item in reranked]

    prompt = build_prompt(query, docs)

    response = generate_response(prompt)

    return response
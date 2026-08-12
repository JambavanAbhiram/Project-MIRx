from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, retrieved_docs, top_k=5):

    pairs = [(query, item["document"]) for item in retrieved_docs]

    scores = reranker_model.predict(pairs)

    for item, score in zip(retrieved_docs, scores):

        item["rerank_score"] = float(score)

    reranked = sorted(
        retrieved_docs,
        key = lambda x: x["rerank_score"],
        reverse = True
    )

    return reranked[:top_k]
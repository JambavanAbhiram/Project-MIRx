import faiss
import pickle
import numpy as np

from embeddings import embed_texts

index = faiss.read_index('data/medicine_index.faiss')

with open('data/records.pkl', 'rb') as f:
    records = pickle.load(f)

def retrieve(query, k=5):

    query_emb = embed_texts([query])

    query_emb = np.array(query_emb).astype('float32')

    distances, indices = index.search(
        query_emb,
        k
    )

    retrieved = []

    for rank, idx in enumerate(indices[0]):

        record = records[idx]

        retrieved.append({

            "rank": rank + 1,

            "distance": float(distances[0][rank]),

            "name": record["name"],

            "uses": record["uses"],

            "therapeutic_class":
                record["therapeutic_class"],

            "confidence_score":
                record["confidence_score"],

            "document": record["document"]
        })

    return retrieved
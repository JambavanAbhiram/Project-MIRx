import json

from retriever import retrieve
from reranker import rerank
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# Load test queries
with open('sample_queries.json', 'r') as f:
    test_queries = json.load(f)

correct = 0

total = len(test_queries)

for sample in test_queries:

    query = sample['query']

    expected_keyword = sample['expected_keyword']

    # Retrieve
    retrieved = retrieve(query, k=10)

    # Rerank
    reranked = rerank(query, retrieved, top_k=5)

    # Top result
    top_doc = reranked[0]['document'].lower()

    print("\n" + "=" * 80)

    print(f"Query: {query}")

    print(f"Expected Keyword: {expected_keyword}")

    if expected_keyword.lower() in top_doc:

        print("Result: CORRECT")

        correct += 1

    else:

        print("Result: INCORRECT")

    print("Top Retrieved Medicine:")

    print(reranked[0]['name'])

accuracy = correct / total

print("\n" + "=" * 80)

print(f"Retrieval Accuracy: {accuracy:.2f}")
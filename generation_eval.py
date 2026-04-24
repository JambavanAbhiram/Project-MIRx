
import json
from rag_pipeline import run_rag

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# Load queries
with open('sample_queries.json', 'r') as f:
    test_queries = json.load(f)

hallucinations = 0

for sample in test_queries:

    query = sample['query']

    expected_keyword = sample['expected_keyword']

    response = run_rag(query)

    print("\n" + "=" * 80)

    print(f"Query: {query}")

    print(f"Response: {response}")

    if expected_keyword.lower() not in response.lower():

        hallucinations += 1

        print("Potential Hallucination Detected")

hallucination_rate = hallucinations / len(test_queries)

print("\n" + "=" * 80)

print(f"Hallucination Rate: {hallucination_rate:.2f}")
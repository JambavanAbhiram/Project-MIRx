import time

from rag_pipeline import run_rag

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

queries = [
    "medicine for fever",
    "medicine for bacterial infection",
    "pain relief medicine"
]

latencies = []

for query in queries:

    start = time.time()

    response = run_rag(query)

    end = time.time()

    latency = end - start

    latencies.append(latency)

    print("\n" + "=" * 80)

    print(f"Query: {query}")

    print(f"Latency: {latency:.2f} seconds")

average_latency = sum(latencies) / len(latencies)

print("\n" + "=" * 80)

print(f"Average Latency: {average_latency:.2f} seconds")
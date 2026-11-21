
import json
import faiss
import numpy as np
import time

from pathlib import Path
from sentence_transformers import SentenceTransformer

index_dir = Path("ml/index")

# Load documents
with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Load both indexes
exact_index = faiss.read_index(str(index_dir / "faiss.index"))
hnsw_index = faiss.read_index(str(index_dir / "faiss_hnsw.index"))

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

queries = [
    "How do machine learning models learn?",
    "Explain neural networks and deep learning.",
    "How do search engines retrieve documents?",
    "What is information retrieval?",
    "Which programming language is Python?",
    "What is Java used for?",
    "Which sport is played with a bat and ball?",
    "Explain football as a team sport.",
    "How do databases store information?",
    "What is artificial intelligence?"
]

TOP_K = 10
REPEATS = 100

# Generate query embeddings only once
query_embeddings = model.encode(queries)
query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
faiss.normalize_L2(query_embeddings)


def benchmark(index, embeddings, repeats):
    # Warm-up search
    index.search(embeddings, TOP_K)

    start = time.perf_counter()

    for _ in range(repeats):
        index.search(embeddings, TOP_K)

    end = time.perf_counter()

    total_queries = len(embeddings) * repeats

    return (end - start) * 1000 / total_queries


# Exact search: ground truth
exact_results = []

for embedding in query_embeddings:
    scores, indices = exact_index.search(
        embedding.reshape(1, -1), TOP_K
    )
    exact_results.append(set(indices[0]))

exact_latency = benchmark(
    exact_index, query_embeddings, REPEATS
)

print("\nEXACT SEARCH")
print(f"Average latency: {exact_latency:.4f} ms/query")

# Test different HNSW search efforts
print("\nHNSW RESULTS")

for ef in [8, 16, 32, 64, 128]:
    hnsw_index.hnsw.efSearch = ef

    hnsw_results = []

    for embedding in query_embeddings:
        scores, indices = hnsw_index.search(
            embedding.reshape(1, -1), TOP_K
        )
        hnsw_results.append(set(indices[0]))

    # Calculate average Recall@10
    recall = np.mean([
        len(exact & approximate) / TOP_K
        for exact, approximate in zip(
            exact_results, hnsw_results
        )
    ])

    latency = benchmark(
        hnsw_index, query_embeddings, REPEATS
    )

    print(
        f"efSearch={ef:3} | "
        f"Latency={latency:.4f} ms/query | "
        f"Recall@10={recall:.2%}"
    )

import json
import faiss
import numpy as np
import time

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load documents and indexes
# --------------------------------------------------

index_dir = Path("ml/index")

with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Load the exact and approximate FAISS indexes.
exact_index = faiss.read_index(str(index_dir / "faiss.index"))
hnsw_index = faiss.read_index(str(index_dir / "faiss_hnsw.index"))


# --------------------------------------------------
# 2. Load embedding model and prepare queries
# --------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

queries = [
    "How do machine learning models learn?",
    "Explain neural networks and deep learning.",
    "How do search engines retrieve documents?",
    "What is information retrieval?",
    "Which programming language is Python?",
    "What is Java used for?",
    "Which sport is played with a bat and ball?",
    "Explain football as a team sport.",
    "How do databases store information?",
    "What is artificial intelligence?"
]

TOP_K = 10
REPEATS = 100

# Generate query embeddings only once.
query_embeddings = model.encode(queries)
query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

# Normalize query vectors because our indexes use inner product
# on normalized vectors to calculate cosine similarity.
faiss.normalize_L2(query_embeddings)


# --------------------------------------------------
# 3. Benchmark function
# --------------------------------------------------

def benchmark(index, embeddings, repeats, top_k):

    # Warm-up search: run once before measuring latency.
    # This helps avoid including initial execution overhead.
    index.search(embeddings, top_k)

    start = time.perf_counter()

    # Repeat the same batch of queries to get a more
    # stable latency measurement.
    for _ in range(repeats):
        scores, indices = index.search(embeddings, top_k)

    end = time.perf_counter()

    # Total number of individual queries executed.
    total_queries = len(embeddings) * repeats

    # Calculate average latency per query in milliseconds.
    avg_latency = (end - start) * 1000 / total_queries

    # Return latency and document IDs from the last iteration.
    return avg_latency, indices


# --------------------------------------------------
# 4. Exact search: establish ground truth
# --------------------------------------------------

# IndexFlatIP checks every vector, giving us exact top-K results.
# We also measure its average search latency.
exact_latency, exact_indices = benchmark(
    exact_index,
    query_embeddings,
    REPEATS,
    TOP_K
)

# Convert each query's document IDs into a set.
# These are the ground-truth results for Recall@10.
exact_results = [
    set(row) for row in exact_indices
]

print("\nEXACT SEARCH")
print(f"Average latency: {exact_latency:.4f} ms/query")


# --------------------------------------------------
# 5. HNSW: test different efSearch values
# --------------------------------------------------

print("\nHNSW RESULTS")

for ef in [8, 16, 32, 64, 128]:

    # efSearch controls how many candidates HNSW explores
    # during search. Higher values generally improve recall
    # but can increase latency.
    hnsw_index.hnsw.efSearch = ef

    # Benchmark HNSW and retrieve its search results.
    # No separate search loop is needed.
    latency, hnsw_indices = benchmark(
        hnsw_index,
        query_embeddings,
        REPEATS,
        TOP_K
    )

    # Convert approximate results into sets for comparison.
    hnsw_results = [
        set(row) for row in hnsw_indices
    ]

    # Calculate Recall@10 for each query:
    # Number of overlapping document IDs / exact top-10 count.
    recalls = [
        len(exact & approximate) / TOP_K
        for exact, approximate in zip(
            exact_results,
            hnsw_results
        )
    ]

    # Average recall across all queries.
    recall = np.mean(recalls)

    print(
        f"efSearch={ef:3} | "
        f"Latency={latency:.4f} ms/query | "
        f"Recall@10={recall:.2%}"
    )

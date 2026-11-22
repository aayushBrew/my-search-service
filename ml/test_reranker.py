
from pathlib import Path
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. Load both models
# --------------------------------------------------

script_dir = Path(__file__).resolve().parent

pretrained_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

finetuned_model = CrossEncoder(
    str(script_dir / "models" / "reranker-finetuned")
)


# --------------------------------------------------
# 2. Prepare unseen test examples
# --------------------------------------------------

test_pairs = [
    (
        "What is vector search?",
        "Vector search finds similar items using numerical embeddings."
    ),
    (
        "What is vector search?",
        "Java is a programming language used for backend development."
    ),
    (
        "How does HNSW work?",
        "HNSW uses a hierarchical graph to perform approximate nearest neighbor search."
    ),
    (
        "How does HNSW work?",
        "BM25 ranks documents based on lexical term matching."
    )
]


# --------------------------------------------------
# 3. Get predictions from both models
# --------------------------------------------------

pretrained_scores = pretrained_model.predict(test_pairs)
finetuned_scores = finetuned_model.predict(test_pairs)


# --------------------------------------------------
# 4. Compare predictions
# --------------------------------------------------

print("\nMODEL COMPARISON")

for i, (query, document) in enumerate(test_pairs):

    print(f"\nQuery: {query}")
    print(f"Document: {document}")

    print(
        f"Pretrained score: {pretrained_scores[i]:.4f}"
    )

    print(
        f"Fine-tuned score: {finetuned_scores[i]:.4f}"
    )


from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Machine learning helps computers learn from data.",
    "AI systems learn patterns from data.",
    "Cricket is a bat and ball sport played between two teams.",
    "Virat Kohli is an Indian international cricketer known for batting.",
    "Python is a popular programming language."
]

query = "Which sport does Virat Kohli play?"

# Generate document embeddings
doc_embeddings = model.encode(documents)

# Generate query embedding
query_embedding = model.encode(query)

# Calculate similarity between query and every document
similarities = cos_sim(query_embedding, doc_embeddings)[0]

# Rank documents by similarity (highest first)
ranked_results = sorted(
    enumerate(similarities.tolist()),
    key=lambda x: x[1],
    reverse=True
)

# Display ranked results
for index, score in ranked_results:
    print(f"Score: {score:.4f} | Document: {documents[index]}")

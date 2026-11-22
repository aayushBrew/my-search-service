
import os

from google import genai

from hybrid_search import search as hybrid_search
from rerank_search import rerank


# --------------------------------------------------
# 1. Initialize Gemini
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Build context from retrieved documents
# --------------------------------------------------

def build_context(documents):

    context_parts = []

    for document in documents:

        context_parts.append(
            f"[Document ID: {document['doc_id']}]\n"
            f"{document['document']}"
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# 3. RAG pipeline
# --------------------------------------------------

def answer_question(query):

    # Stage 1: Retrieve candidates using BM25 + FAISS + RRF
    candidates = hybrid_search(query, top_k=20)

    # Stage 2: Rerank candidates using cross-encoder
    relevant_documents = rerank(
        query,
        candidates,
        top_k=5
    )

    # Stage 3: Construct context
    context = build_context(relevant_documents)

    # Stage 4: Ask Gemini to answer using retrieved context
    prompt = f"""
You are a question-answering assistant.

Answer the user's question using only the context below.

If the context does not contain enough information,
clearly say that you don't have enough information.

Cite supporting documents using their IDs in this format:
[Document ID: 123]

Context:
{context}

User question:
{query}

Answer:
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text, relevant_documents


# --------------------------------------------------
# 4. Test
# --------------------------------------------------

if __name__ == "__main__":

    query = "How do machine learning models learn?"

    answer, sources = answer_question(query)

    print("\nQUESTION:")
    print(query)

    print("\nRETRIEVED SOURCES:")

    for source in sources:
        print(
            f"\nID: {source['doc_id']} | "
            f"Score: {source['score']:.4f}"
        )
        print(source["document"])

    print("\nGENERATED ANSWER:")
    print(answer)

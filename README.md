# Search Engineering Playground

A hands-on search engineering project exploring modern information retrieval systems, from traditional keyword search to dense vector retrieval, approximate nearest-neighbor search, hybrid search, and Retrieval-Augmented Generation (RAG).

The project combines a **Java search service** with a **Python-based ML retrieval pipeline** to understand and implement production-oriented search architectures.

## Project Architecture

- **Java:** Search service and backend integration.
- **Python:** Embedding generation, vector indexing, retrieval experiments, and model evaluation.
- **FAISS:** Exact and approximate vector similarity search.
- **Sentence Transformers:** Text embedding generation.
- **GitHub Actions:** CI pipeline for Java build and tests.

---

## 1. CI/CD – Java Service

**Technology:** Java 17, Maven, GitHub Actions

- Created a Java Maven-based search service.
- Configured GitHub Actions for automated builds and tests.
- CI workflow runs on pull requests targeting the `main` branch.
- Uses Java 17 for compilation and test execution.
- Includes unit tests and integration tests.

**Workflow:** `.github/workflows/ci.yml`

**To run locally:**

```bash
mvn clean test
```

---

## 2. Dense Retrieval – Semantic Search

**Technology:** Python, Sentence Transformers, NumPy

Implemented semantic retrieval using dense vector embeddings.

- Integrated `sentence-transformers/all-MiniLM-L6-v2`.
- Generated 384-dimensional embeddings for documents and queries.
- Implemented cosine similarity-based document retrieval.
- Explored document embedding generation and persistence.
- Compared semantic similarity scores across different documents.

**Key concepts:**
- Dense vector representations
- Embedding models
- Cosine similarity
- Semantic similarity vs keyword matching

**Implementation:**
- `ml/test_embeddings.py`
- `ml/similarity.py`

---

## 3. Vector Search – FAISS Exact Search

**Technology:** FAISS, NumPy

Implemented exact nearest-neighbor search using FAISS.

- Generated document embeddings and normalized vectors.
- Built an exact vector index using `IndexFlatIP`.
- Used inner product on normalized vectors to perform cosine similarity search.
- Persisted the FAISS index to disk.
- Loaded the index independently during query execution.

**Implementation:**
- `ml/build_index.py`
- `ml/search_engine.py`

**Index artifact:**
- `ml/index/faiss.index`

---

## 4. Approximate Nearest-Neighbor Search – HNSW

**Technology:** FAISS HNSW

Implemented approximate vector retrieval using Hierarchical Navigable Small World (HNSW) graphs.

Explored the following parameters:

| Parameter | Description |
|---|---|
| `M` | Controls graph connectivity |
| `efConstruction` | Controls candidate exploration during graph construction |
| `efSearch` | Controls candidate exploration during query execution |

**Current configuration:**
- `M = 16`
- `efConstruction = 100`
- `efSearch` tested at 8, 16, 32, 64 and 128.

Built and persisted a separate HNSW index for approximate retrieval.

**Implementation:**
- `ml/build_hnsw_index.py`

**Index artifact:**
- `ml/index/faiss_hnsw.index`

---

## 5. Exact vs Approximate Search – Benchmarking

Compared FAISS exact search (`IndexFlatIP`) against HNSW approximate search.

**Evaluation metrics:**

- **Recall@K:** Measures how many of the exact top-K results are retrieved by approximate search.
- **Average Query Latency:** Measures average search execution time per query.

**Benchmark methodology:**

1. Use exact search to establish ground-truth top-K document IDs.
2. Measure exact search latency.
3. Execute HNSW search with different `efSearch` values.
4. Calculate Recall@K by comparing HNSW results against exact results.
5. Measure average HNSW query latency.
6. Compare the speed–recall trade-off.

**Benchmark dataset:**
- 5,000 synthetic documents.
- 10 evaluation queries.
- 384-dimensional embeddings.
- Top-K = 10.
- 100 benchmark repetitions.

**Sample benchmark results:**

| Search configuration | Latency (ms/query) | Recall@10 |
|---|---:|---:|
| Exact Search | 0.1412 | 100% |
| HNSW – efSearch 8 | 0.0055 | 80% |
| HNSW – efSearch 16 | 0.0110 | 87% |
| HNSW – efSearch 32 | 0.0149 | 97% |
| HNSW – efSearch 64 | 0.0186 | 99% |
| HNSW – efSearch 128 | 0.0388 | 100% |

*Results are from one local benchmark run. Latencies can vary depending on hardware and system load. The synthetic dataset is intended for experimentation, not production-quality retrieval evaluation.*

**To run the benchmark:**

From the project root:

```bash
python ml/compare_indexes.py
```

**Prerequisites:**

The following index artifacts must exist:

- `ml/index/documents.json`
- `ml/index/faiss.index`
- `ml/index/faiss_hnsw.index`

Generate the dataset and indexes in this order:

```bash
python ml/generate_dataset.py
python ml/build_index.py
python ml/build_hnsw_index.py
python ml/compare_indexes.py
```

**Output:** Benchmark results are printed directly to the terminal, including latency and Recall@10 for each HNSW configuration.

---

## 6. Project Structure

```text
my-search-service/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── main/java/com/example/search/
│   │   └── SearchService.java
│   │
│   └── test/java/com/example/search/
│       ├── SearchServiceTest.java
│       └── SearchIntegrationTest.java
│
├── ml/
│   ├── generate_dataset.py
│   ├── test_embeddings.py
│   ├── similarity.py
│   ├── build_index.py
│   ├── build_hnsw_index.py
│   ├── search_engine.py
│   ├── compare_indexes.py
│   │
│   └── index/
│       ├── documents.json
│       ├── faiss.index
│       └── faiss_hnsw.index
│
├── requirements.txt
├── pom.xml
├── README.md
└── .gitignore
```

*The `ml/index/` directory contains generated artifacts and is excluded from Git.*

---

## 7. Dependencies and Setup

### Java

- Java 17
- Apache Maven

### Python

- Python 3.12
- sentence-transformers
- FAISS CPU
- NumPy

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The Python environment is kept separate using a virtual environment.

### Current Python dependencies

```text
sentence-transformers==6.3.0
faiss-cpu==1.16.0
numpy==2.5.5
```

---

## 8. Upcoming Work / Roadmap

The project will progressively evolve into a complete retrieval and RAG system.

- [ ] BM25 keyword retrieval.
- [ ] Hybrid retrieval combining BM25 and dense vector search.
- [ ] Reciprocal Rank Fusion (RRF).
- [ ] Cross-encoder reranking.
- [ ] Retrieval evaluation: Precision@K, Recall@K, MRR and nDCG.
- [ ] Document chunking and metadata filtering.
- [ ] RAG pipeline: retrieval, context construction and LLM generation.
- [ ] RAG evaluation and grounded answer generation.
- [ ] GraphRAG: entity extraction, knowledge graphs and graph-based retrieval.
- [ ] Fine-tuning embedding models and rerankers.
- [ ] Model evaluation, export and versioning.
- [ ] Python inference service.
- [ ] Java integration with the Python retrieval service.
- [ ] Docker-based deployment and monitoring.

---

## Objective

Build a practical understanding of modern search systems by implementing each component, evaluating its impact independently, and progressively integrating the components into a production-oriented retrieval architecture.

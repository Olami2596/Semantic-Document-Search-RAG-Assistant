# Semantic Document Search & RAG Assistant

A local document question-answering system that uses semantic search and Retrieval-Augmented Generation (RAG) to answer questions from a collection of text documents.

The system converts documents into vector embeddings, stores them in ChromaDB, retrieves relevant sections using semantic similarity, and uses a locally running LLM through Ollama to generate context-grounded answers.

## Features

- Document ingestion from local TXT files
- Section-aware document chunking
- Semantic embeddings using Sentence Transformers
- Persistent vector storage with ChromaDB
- Cosine similarity search
- RAG-based answer generation
- Local LLM inference using Ollama
- Context-grounded answers with document references
- Abstention when information is unavailable
- Incremental document synchronisation using SHA-256 hashes
- FastAPI REST API
- Automated API tests
- Retrieval evaluation using Recall@K

## Architecture

```text
                 Documents
                     |
                     v
              Document Ingestion
                     |
                     v
             Section-Aware Chunking
                     |
                     v
           Sentence Transformer Model
                     |
                     v
                  ChromaDB
                     |
                     |
User Question ------+
       |
       v
Question Embedding
       |
       v
Semantic Retrieval
       |
       v
Relevant Document Chunks
       |
       v
Context Construction
       |
       v
       Ollama
       |
       v
   Generated Answer
       |
       v
    FastAPI JSON
```

## Technology Stack

| Component       | Technology       |
| --------------- | ---------------- |
| Language        | Python           |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB         |
| LLM             | Qwen 2.5 1.5B    |
| LLM runtime     | Ollama           |
| API framework   | FastAPI          |
| Data validation | Pydantic         |
| Testing         | Pytest           |
| API testing     | HTTPX            |

## Project Structure

```text
semantic-document-search/
│
├── data/
│   ├── ai_healthcare.txt
│   ├── customer_support.txt
│   ├── hr_policy.txt
│   └── product_policy.txt
│
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── search.py
│   ├── rag.py
│   ├── api.py
│   ├── schemas.py
│   └── evaluate_retrieval.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone Semantic-Document-Search-RAG-Assistant
cd semantic-document-search
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama from:

https://ollama.com/

Download the model:

```bash
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running before making API requests.

### 5. Start the API

From the project root:

```bash
uvicorn src.api:app --reload --reload-dir src --reload-dir data
```

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

## API Usage

### Health Check

**GET** `/health`

Example response:

```json
{
  "status": "healthy"
}
```

### Ask a Question

**POST** `/ask`

Request:

```json
{
  "question": "How many working days of annual leave do full-time employees receive?"
}
```

Example response:

```json
{
  "question": "How many working days of annual leave do full-time employees receive?",
  "answer": "Full-time employees receive 25 working days of paid annual leave every year.",
  "references": [
    {
      "source": "hr_policy.txt",
      "section": "Annual Leave",
      "chunk_index": 1,
      "distance": 0.145
    }
  ]
}
```

The reference distance represents the cosine distance returned by ChromaDB. Lower values indicate greater similarity.

## Retrieval Evaluation

Run:

```bash
python -m src.evaluate_retrieval
```

The evaluation checks whether the expected document and section appear within the top K retrieved results.

Initial evaluation:

- Test cases: 5
- Correct retrievals: 5
- Recall@2: 100%

Note: This is a small development evaluation dataset, not a comprehensive benchmark.

## Automated Tests

Run:

```bash
python -m pytest -v
```

The API tests cover:

- Health endpoint
- Successful question submission
- Rejection of questions that are too short
- Rejection of requests with missing questions

The API tests mock ingestion and answer generation to isolate API behaviour.

## Incremental Document Synchronisation

Documents are fingerprinted using SHA-256 hashes.

The ingestion pipeline:

- Skips unchanged documents.
- Re-indexes modified documents.
- Indexes newly added documents.
- Removes records belonging to deleted documents.

This avoids unnecessary embedding generation during subsequent application startups.

## Limitations

- Currently supports TXT documents.
- Uses a small local language model.
- Runs on a local ChromaDB instance.
- Retrieval evaluation uses a small test dataset.
- No authentication or rate limiting is implemented.
- API is intended for local development and demonstration, not unrestricted public deployment.

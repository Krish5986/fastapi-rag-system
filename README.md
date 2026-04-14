# 📘 RAG-Based Question Answering System

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based Question Answering API using FastAPI.

The system allows users to:

* Upload documents (PDF/TXT)
* Ask questions based on those documents
* Receive context-grounded answers using an LLM

The goal is to build a **real-world applied AI system**, focusing on retrieval quality, latency, and hallucination control.

---

## 🏗️ System Architecture

1. **Document Upload**

   * Users upload PDF or TXT files via `/upload`
   * Files are stored locally

2. **Background Processing**

   * Text is extracted from documents
   * Text is split into chunks
   * Embeddings are generated
   * Stored in FAISS vector index

3. **Query Flow**

   * User sends a question via `/query`
   * Question is embedded
   * FAISS retrieves top-k similar chunks
   * Retrieved context is passed to LLM (Gemini)
   * LLM generates answer strictly based on context

---

## ⚙️ Tech Stack

* **Framework**: FastAPI
* **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector DB**: FAISS
* **LLM**: Google Gemini (2.5 Flash)
* **Rate Limiting**: slowapi
* **PDF Parsing**: PyMuPDF (fitz)

---

## ✂️ Chunking Strategy (Important)

* **Chunk Size**: 300 words
* **Overlap**: 50 words

### Why this choice?

* 300 words provides enough semantic context for meaningful embeddings
* Smaller chunks → better retrieval precision
* Overlap (50 words) ensures:

  * continuity across chunks
  * prevents context loss at boundaries

👉 This balances **retrieval accuracy vs embedding efficiency**

---

## 🔍 Retrieval Strategy

* FAISS with L2 similarity search
* Top **k = 3** chunks retrieved per query
* Context is constructed by combining retrieved chunks

---

## ❌ Retrieval Failure Case

### Example:

**Query**: “What is the final conclusion of the document?”

### Issue:

* The conclusion was split across multiple chunks
* Retrieved chunks did not contain complete information

### Result:

* Model could not answer accurately

### Solution implemented:

```python
if not retrieved_chunks:
    return "I cannot answer this based on the provided documents."
```

👉 This prevents hallucination and ensures reliability

---

## 📊 Metric Tracked

### Latency

We measure **end-to-end query latency**, including:

* embedding generation
* vector search
* LLM response time

```python
start_time = time.time()
...
latency = time.time() - start_time
```

### Why latency?

* Critical for real-world systems
* Directly impacts user experience
* Helps evaluate system performance

---

## 🛡️ Hallucination Control

The system enforces strict grounding:

* LLM is instructed:

  > "Answer ONLY from context. Do not guess."

* If no relevant chunks are found:

  * System returns fallback response
  * Avoids generating incorrect answers

---

## 📡 API Endpoints

### 1. Upload Document

```
POST /upload
```

### 2. Query

```
POST /query
```

**Request:**

```json
{
  "question": "Your question here"
}
```

**Response:**

```json
{
  "question": "...",
  "answer": "...",
  "sources_used": 3,
  "latency": 1.25
}
```

---

## ⚡ Key Features

* ✅ Background document processing
* ✅ Fast semantic search using FAISS
* ✅ Lightweight embedding model
* ✅ Rate limiting for API safety
* ✅ Latency tracking
* ✅ Hallucination prevention

---

## 🧠 Design Decisions

* Used **MiniLM** → fast + efficient embeddings
* Used **FAISS** → simple, local, high performance
* Used **Gemini Flash** → low latency LLM
* Implemented **background tasks** → non-blocking uploads

---

## 🚧 Limitations

* FAISS index is in-memory (no persistence)
* No metadata tracking for chunks
* Retrieval may fail for long-context queries

---

## 🔮 Future Improvements

* Add persistent vector storage
* Add metadata (source, page number)
* Improve chunking (semantic chunking)
* Add hybrid search (keyword + vector)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🎯 Conclusion

This project demonstrates:

* End-to-end RAG pipeline
* Practical trade-offs in chunking & retrieval
* Importance of latency and hallucination control

It is designed as a **production-aware AI system**, not just a prototype.

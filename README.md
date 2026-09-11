# MIRx

### A Scalable Retrieval-Augmented Generation (RAG) System

MIRx is a Retrieval-Augmented Generation (RAG) based question-answering system designed to generate **relevant, context-aware, and user-friendly responses** from a large domain-specific dataset.

The current implementation is a **Proof of Concept (PoC)** that validates the complete RAG pipeline:

> **User Query → Embedding → Vector Search → Context Retrieval → LLM → Response**

The system currently works with a dataset containing **240K+ samples** and is being designed with future scalability and extensibility in mind.

---

## Current PoC

The current version implements a simple end-to-end RAG architecture with a frontend through which users can submit queries.

```text
                         ┌──────────────────┐
                         │       User       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Frontend     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Query Embedding  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Qdrant      │
                         │    Vector DB     │
                         └────────┬─────────┘
                                  │
                             Top-K Context
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Open-Weight LLM │
                         │ Response Gen.    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Final Response  │
                         └──────────────────┘

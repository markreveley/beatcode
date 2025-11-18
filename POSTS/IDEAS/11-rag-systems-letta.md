---
title: "Building RAG Systems with Letta: Retrieval-Augmented Generation"
date: "2025-02-12"
description: "Build sophisticated retrieval-augmented generation systems that combine Letta agents with external knowledge bases, vector databases, and semantic search."
tags: ["Letta", "RAG", "Vector DB", "Advanced", "Knowledge Retrieval"]
---

# Building RAG Systems with Letta: Retrieval-Augmented Generation

Letta agents are powerful, but they're limited by their training data. **RAG (Retrieval-Augmented Generation)** solves this by giving agents access to external knowledge bases. In this advanced tutorial, we'll build production RAG systems with Letta.

## What is RAG?

```
User Question
      ↓
┌─────────────────┐
│  1. Retrieve    │ ← Search knowledge base
│  Relevant Docs  │   for relevant information
└────────┬────────┘
         ↓
┌─────────────────┐
│  2. Augment     │ ← Inject retrieved docs
│  Prompt with    │   into agent context
│  Retrieved Info │
└────────┬────────┘
         ↓
┌─────────────────┐
│  3. Generate    │ ← Agent generates response
│  Response with  │   using retrieved knowledge
│  Letta Agent    │
└─────────────────┘
```

## Basic RAG with Letta

Let's start simple:

```python
from letta import create_client
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class SimpleRAG:
    """Basic RAG implementation with Letta."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Knowledge base: text chunks with embeddings
        self.knowledge_base = []
        self.embeddings = []

    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base."""
        for doc in documents:
            # Store document
            self.knowledge_base.append(doc)

            # Generate and store embedding
            embedding = self.encoder.encode(doc)
            self.embeddings.append(embedding)

        print(f"Added {len(documents)} documents to knowledge base")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant documents for query."""
        if not self.knowledge_base:
            return []

        # Encode query
        query_embedding = self.encoder.encode(query)

        # Calculate cosine similarity with all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in similarities[:top_k]]

        return [self.knowledge_base[i] for i in top_indices]

    def query(self, question: str) -> Dict:
        """Query with RAG: retrieve + generate."""

        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve(question, top_k=3)

        if not relevant_docs:
            # No relevant docs, ask agent directly
            context = ""
        else:
            # Build context from retrieved documents
            context = "Relevant information:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"{i}. {doc}\n\n"

        # Step 2: Augment prompt with context
        augmented_prompt = f"""{context}

Based on the information above, please answer this question:
{question}

If the information provided isn't sufficient, please say so and provide your best answer based on general knowledge."""

        # Step 3: Generate response with Letta agent
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=augmented_prompt,
            role="user"
        )

        response_text = self._extract_response(response)

        return {
            "question": question,
            "retrieved_docs": relevant_docs,
            "response": response_text,
            "num_docs_used": len(relevant_docs)
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
agent = create_client().create_agent(
    name="RAGAgent",
    persona="You are a helpful assistant that answers questions based on provided information."
)

rag = SimpleRAG(agent.id)

# Add knowledge base
documents = [
    "Letta is an open-source framework for building stateful AI agents with long-term memory.",
    "Letta agents can use tools and functions to interact with external systems and APIs.",
    "The Letta memory system has three tiers: core memory, recall memory, and archival memory.",
    "Letta was formerly known as MemGPT before being renamed in 2024.",
    "Letta supports multiple LLM providers including OpenAI, Anthropic, and Azure."
]

rag.add_documents(documents)

# Query with RAG
result = rag.query("What is Letta?")

print(f"Question: {result['question']}")
print(f"\nRetrieved {result['num_docs_used']} relevant documents:")
for i, doc in enumerate(result['retrieved_docs'], 1):
    print(f"{i}. {doc}")

print(f"\nResponse: {result['response']}")
```

## Advanced RAG with Vector Database

Using a proper vector database for scale:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorRAG:
    """Production RAG with Qdrant vector database."""

    def __init__(self, agent_id: str, collection_name: str = "knowledge_base"):
        self.agent_id = agent_id
        self.client = create_client()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Qdrant
        self.qdrant = QdrantClient(":memory:")  # Use ":memory:" for testing, URL for production
        self.collection_name = collection_name

        # Create collection
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding size
                distance=Distance.COSINE
            )
        )

    def add_documents(
        self,
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """Add documents to vector database."""
        if metadata is None:
            metadata = [{}] * len(documents)

        points = []

        for doc, meta in zip(documents, metadata):
            # Generate embedding
            embedding = self.encoder.encode(doc).tolist()

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": doc,
                    "metadata": meta
                }
            )
            points.append(point)

        # Upload to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Added {len(documents)} documents to vector database")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """Retrieve relevant documents with scores."""

        # Encode query
        query_embedding = self.encoder.encode(query).tolist()

        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )

        # Format results
        retrieved = []
        for result in results:
            retrieved.append({
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })

        return retrieved

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query with RAG."""

        # Retrieve relevant documents
        retrieved = self.retrieve(question, top_k=top_k)

        if not retrieved:
            context = "No relevant information found in the knowledge base.\n\n"
        else:
            context = "Retrieved Information:\n\n"
            for i, doc in enumerate(retrieved, 1):
                context += f"{i}. {doc['text']} (relevance: {doc['score']:.2f})\n\n"

        # Augment prompt
        augmented_prompt = f"""{context}

Question: {question}

Please provide a comprehensive answer based on the retrieved information.
If the retrieved information is insufficient, supplement with your general knowledge
but clearly indicate which parts are from the provided context vs. general knowledge."""

        # Generate response
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=augmented_prompt,
            role="user"
        )

        return {
            "question": question,
            "retrieved": retrieved,
            "response": self._extract_response(response)
        }

    def add_document_with_chunking(
        self,
        long_document: str,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """Add long document with chunking."""
        chunks = self._chunk_text(long_document, chunk_size, overlap)
        self.add_documents(chunks)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage with production vector DB
agent = create_client().create_agent(name="VectorRAGAgent")
vector_rag = VectorRAG(agent.id)

# Add documents with metadata
docs = [
    "Letta enables building AI agents with persistent memory.",
    "Vector databases store embeddings for semantic search."
]
metadata = [
    {"source": "letta_docs", "category": "features"},
    {"source": "tech_blog", "category": "databases"}
]

vector_rag.add_documents(docs, metadata)

# Query
result = vector_rag.query("How do I build AI agents?")
print(result['response'])
```

## Hybrid Search: Combining Dense and Sparse Retrieval

Best of both worlds:

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRAG:
    """RAG with hybrid dense + sparse retrieval."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Storage
        self.documents = []
        self.dense_embeddings = []
        self.bm25 = None

    def add_documents(self, documents: List[str]):
        """Add documents and build indices."""
        self.documents.extend(documents)

        # Build dense embeddings
        for doc in documents:
            embedding = self.encoder.encode(doc)
            self.dense_embeddings.append(embedding)

        # Build BM25 sparse index
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5  # Weight for dense vs sparse
    ) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse search.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight (0=only sparse, 1=only dense, 0.5=balanced)
        """
        if not self.documents:
            return []

        # Dense retrieval (semantic)
        query_embedding = self.encoder.encode(query)
        dense_scores = []

        for doc_embedding in self.dense_embeddings:
            score = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            dense_scores.append(score)

        # Sparse retrieval (keyword-based BM25)
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores(dense_scores)
        sparse_scores = self._normalize_scores(sparse_scores)

        # Combine scores
        hybrid_scores = [
            alpha * dense + (1 - alpha) * sparse
            for dense, sparse in zip(dense_scores, sparse_scores)
        ]

        # Get top_k results
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "score": hybrid_scores[idx],
                "dense_score": dense_scores[idx],
                "sparse_score": sparse_scores[idx]
            })

        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]."""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return [0.5] * len(scores)

        return ((scores - min_score) / (max_score - min_score)).tolist()

    def query(self, question: str) -> Dict:
        """Query with hybrid RAG."""
        retrieved = self.hybrid_retrieve(question, top_k=3)

        context = "Retrieved Information (using hybrid search):\n\n"
        for i, doc in enumerate(retrieved, 1):
            context += f"{i}. {doc['text']}\n"
            context += f"   (relevance: {doc['score']:.2f}, "
            context += f"semantic: {doc['dense_score']:.2f}, "
            context += f"keyword: {doc['sparse_score']:.2f})\n\n"

        prompt = f"{context}\nQuestion: {question}\n\nAnswer based on the retrieved information:"

        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )

        return {
            "question": question,
            "retrieved": retrieved,
            "response": self._extract_response(response)
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)
```

## Contextual RAG with Re-ranking

Improve retrieval quality with re-ranking:

```python
from sentence_transformers import CrossEncoder

class ContextualRAG:
    """RAG with contextual re-ranking for better results."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()

        # Bi-encoder for initial retrieval
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')

        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base."""
        self.documents.extend(documents)

        for doc in documents:
            embedding = self.retriever.encode(doc)
            self.embeddings.append(embedding)

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[Dict]:
        """
        Two-stage retrieval: fast retrieval + precise re-ranking.

        Args:
            initial_k: Number of candidates from initial retrieval
            final_k: Number of results after re-ranking
        """

        # Stage 1: Fast bi-encoder retrieval
        query_embedding = self.retriever.encode(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, sim))

        # Get top initial_k candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = [
            {"index": i, "text": self.documents[i], "retrieval_score": score}
            for i, score in similarities[:initial_k]
        ]

        # Stage 2: Precise cross-encoder re-ranking
        pairs = [[query, cand["text"]] for cand in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Combine and sort by rerank scores
        for cand, rerank_score in zip(candidates, rerank_scores):
            cand["rerank_score"] = float(rerank_score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:final_k]

    def query(self, question: str) -> Dict:
        """Query with contextual RAG."""
        retrieved = self.retrieve_and_rerank(question)

        context = "Top relevant passages (after re-ranking):\n\n"
        for i, doc in enumerate(retrieved, 1):
            context += f"{i}. {doc['text']}\n"
            context += f"   Retrieval: {doc['retrieval_score']:.3f}, "
            context += f"Relevance: {doc['rerank_score']:.3f}\n\n"

        prompt = f"{context}\nQuestion: {question}\n\nProvide a detailed answer:"

        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )

        return {
            "question": question,
            "retrieved": retrieved,
            "response": self._extract_response(response)
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)
```

## RAG with Citation Tracking

Track which sources inform the response:

```python
class CitationRAG:
    """RAG system that tracks and cites sources."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        self.documents = []
        self.embeddings = []
        self.sources = []  # Track source metadata

    def add_documents(
        self,
        documents: List[str],
        sources: List[Dict]
    ):
        """
        Add documents with source information.

        Args:
            documents: List of text documents
            sources: List of source metadata (title, author, url, etc.)
        """
        for doc, source in zip(documents, sources):
            self.documents.append(doc)
            self.sources.append(source)

            embedding = self.encoder.encode(doc)
            self.embeddings.append(embedding)

    def query_with_citations(self, question: str, top_k: int = 3) -> Dict:
        """Query and return response with citations."""

        # Retrieve relevant documents
        query_embedding = self.encoder.encode(question)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_docs = similarities[:top_k]

        # Build context with citation markers
        context = "Reference materials:\n\n"
        citations = []

        for idx, (doc_idx, score) in enumerate(top_docs, 1):
            doc = self.documents[doc_idx]
            source = self.sources[doc_idx]

            context += f"[{idx}] {doc}\n"
            context += f"    Source: {source.get('title', 'Unknown')}\n\n"

            citations.append({
                "id": idx,
                "text": doc,
                "source": source,
                "relevance": float(score)
            })

        # Enhanced prompt for citations
        prompt = f"""{context}

Question: {question}

Please answer the question using the reference materials above.
When using information from a reference, cite it using [1], [2], etc.
If you're unsure or the references don't contain the answer, please say so.

Answer:"""

        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )

        response_text = self._extract_response(response)

        # Extract which citations were used
        used_citations = []
        for citation in citations:
            if f"[{citation['id']}]" in response_text:
                used_citations.append(citation)

        return {
            "question": question,
            "response": response_text,
            "citations": citations,
            "used_citations": used_citations
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
agent = create_client().create_agent(name="CitationAgent")
citation_rag = CitationRAG(agent.id)

docs = [
    "Letta provides stateful AI agents with memory.",
    "RAG combines retrieval with generation for knowledge-grounded responses."
]

sources = [
    {"title": "Letta Documentation", "url": "https://docs.letta.ai"},
    {"title": "RAG Tutorial", "url": "https://example.com/rag"}
]

citation_rag.add_documents(docs, sources)

result = citation_rag.query_with_citations("What is Letta?")
print(f"Response: {result['response']}")
print(f"\nCitations used:")
for cit in result['used_citations']:
    print(f"[{cit['id']}] {cit['source']['title']}")
```

## Performance Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedRAG:
    """RAG with performance optimizations."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Cache for embeddings
        self._embedding_cache = {}

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def query_async(self, question: str) -> Dict:
        """Async query for better concurrency."""

        # Run retrieval and generation in parallel where possible
        loop = asyncio.get_event_loop()

        # Retrieve documents
        retrieved = await loop.run_in_executor(
            self._executor,
            self._retrieve,
            question
        )

        # Generate response
        response = await loop.run_in_executor(
            self._executor,
            self._generate,
            question,
            retrieved
        )

        return {
            "question": question,
            "retrieved": retrieved,
            "response": response
        }

    def _retrieve(self, query: str) -> List[Dict]:
        """Retrieval logic."""
        # Implementation here
        pass

    def _generate(self, question: str, context: List[Dict]) -> str:
        """Generation logic."""
        # Implementation here
        pass
```

## Next Steps

You've mastered RAG with Letta! Next: **Multi-Modal Agents** - working with images, audio, and video.

## Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)

Build knowledge-powered agents!

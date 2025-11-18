---
title: "Custom Memory Architectures in Letta: Building Advanced Memory Systems"
date: "2025-02-05"
description: "Deep dive into creating custom memory backends, implementing specialized memory structures, and optimizing memory performance for your agents."
tags: ["Letta", "Memory", "Advanced", "Architecture", "Performance"]
---

# Custom Memory Architectures in Letta: Building Advanced Memory Systems

Letta's default memory system is powerful, but what if you need something more specialized? In this advanced tutorial, we'll build **custom memory architectures** - from graph-based memory to time-series storage to domain-specific memory systems.

## Understanding Memory Backends

Letta's memory system has three pluggable components:

```
┌─────────────────────────────────────┐
│       Core Memory Backend           │
│  (Key-value storage: human/persona) │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      Recall Memory Backend          │
│    (Message history storage)        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│    Archival Memory Backend          │
│  (Vector database for knowledge)    │
└─────────────────────────────────────┘
```

Each can be customized or replaced entirely.

## Custom Archival Memory: Graph-Based Knowledge

Let's build a graph-based memory system where information is stored as connected nodes:

```python
from typing import List, Dict, Optional
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

class GraphMemoryBackend:
    """
    Graph-based memory where information is stored as connected nodes.
    Each node is a piece of information, edges represent relationships.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.graph = nx.DiGraph()
        self.encoder = SentenceTransformer(embedding_model)
        self.node_counter = 0

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        related_to: Optional[List[int]] = None
    ) -> int:
        """
        Add a memory node to the graph.

        Args:
            content: The memory content
            metadata: Optional metadata
            related_to: List of node IDs this memory relates to

        Returns:
            The new node ID
        """
        # Generate embedding
        embedding = self.encoder.encode(content)

        # Create node
        node_id = self.node_counter
        self.node_counter += 1

        self.graph.add_node(
            node_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time()
        )

        # Add edges to related nodes
        if related_to:
            for related_id in related_to:
                if self.graph.has_node(related_id):
                    # Calculate relationship strength based on embedding similarity
                    similarity = self._cosine_similarity(
                        embedding,
                        self.graph.nodes[related_id]['embedding']
                    )
                    self.graph.add_edge(node_id, related_id, weight=similarity)

        return node_id

    def search(
        self,
        query: str,
        limit: int = 5,
        include_neighbors: bool = True
    ) -> List[Dict]:
        """
        Search memory using semantic similarity and graph traversal.

        Args:
            query: Search query
            limit: Maximum results
            include_neighbors: Whether to include connected nodes

        Returns:
            List of matching memories with context
        """
        query_embedding = self.encoder.encode(query)

        # Calculate similarity scores for all nodes
        scores = []
        for node_id in self.graph.nodes():
            node_embedding = self.graph.nodes[node_id]['embedding']
            similarity = self._cosine_similarity(query_embedding, node_embedding)
            scores.append((node_id, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[:limit]

        # Build results
        results = []
        for node_id, score in top_nodes:
            node_data = self.graph.nodes[node_id]

            result = {
                "id": node_id,
                "content": node_data['content'],
                "score": float(score),
                "metadata": node_data['metadata']
            }

            # Include connected nodes if requested
            if include_neighbors:
                neighbors = list(self.graph.neighbors(node_id))
                result['related'] = [
                    {
                        "id": n,
                        "content": self.graph.nodes[n]['content'],
                        "relationship": self.graph[node_id][n]['weight']
                    }
                    for n in neighbors[:3]  # Top 3 neighbors
                ]

            results.append(result)

        return results

    def get_context_path(self, start_node: int, end_node: int) -> List[int]:
        """
        Find path between two memory nodes.

        Returns:
            List of node IDs forming the path
        """
        try:
            return nx.shortest_path(self.graph, start_node, end_node)
        except nx.NetworkXNoPath:
            return []

    def get_memory_cluster(self, node_id: int, depth: int = 2) -> List[int]:
        """
        Get cluster of related memories around a node.

        Args:
            node_id: Central node
            depth: How many hops to traverse

        Returns:
            List of related node IDs
        """
        cluster = set([node_id])

        for _ in range(depth):
            new_nodes = set()
            for n in cluster:
                new_nodes.update(self.graph.neighbors(n))
            cluster.update(new_nodes)

        return list(cluster)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def export_graph(self, filename: str):
        """Export graph to file for visualization."""
        nx.write_graphml(self.graph, filename)

# Usage
import time

graph_memory = GraphMemoryBackend()

# Add connected memories
python_id = graph_memory.add_memory("Python is a programming language")
django_id = graph_memory.add_memory("Django is a Python web framework", related_to=[python_id])
flask_id = graph_memory.add_memory("Flask is a Python web framework", related_to=[python_id])
react_id = graph_memory.add_memory("React is a JavaScript framework")

# Django and Flask are both Python frameworks
graph_memory.graph.add_edge(django_id, flask_id, weight=0.8)

# Search with context
results = graph_memory.search("Python web development", limit=3)

for result in results:
    print(f"\n📝 {result['content']} (score: {result['score']:.3f})")
    if 'related' in result:
        for rel in result['related']:
            print(f"  → Related: {rel['content']}")

# Get cluster
cluster = graph_memory.get_memory_cluster(python_id, depth=2)
print(f"\n🔗 Memory cluster around Python: {len(cluster)} related memories")
```

## Time-Series Memory Backend

Store memories with temporal relationships:

```python
from datetime import datetime, timedelta
from collections import defaultdict
import bisect

class TimeSeriesMemory:
    """
    Memory system optimized for time-series data.
    Efficient for "What happened on..." or "Events between..." queries.
    """

    def __init__(self):
        self.memories = []  # Sorted by timestamp
        self.indices = {
            'by_date': defaultdict(list),
            'by_tag': defaultdict(list),
            'by_type': defaultdict(list)
        }

    def add_memory(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None
    ) -> int:
        """Add time-stamped memory."""
        timestamp = timestamp or datetime.now()
        tags = tags or []

        memory = {
            'id': len(self.memories),
            'content': content,
            'timestamp': timestamp,
            'tags': tags,
            'type': memory_type
        }

        # Insert in sorted order
        bisect.insort(self.memories, memory, key=lambda x: x['timestamp'])

        # Update indices
        date_key = timestamp.date()
        self.indices['by_date'][date_key].append(memory['id'])

        for tag in tags:
            self.indices['by_tag'][tag].append(memory['id'])

        if memory_type:
            self.indices['by_type'][memory_type].append(memory['id'])

        return memory['id']

    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Get all memories in a date range."""
        results = []

        for memory in self.memories:
            if start_date <= memory['timestamp'] <= end_date:
                results.append(memory)
            elif memory['timestamp'] > end_date:
                break  # List is sorted, no need to continue

        return results

    def get_timeline(self, tags: Optional[List[str]] = None) -> List[Dict]:
        """Get chronological timeline, optionally filtered by tags."""
        if not tags:
            return self.memories

        # Find memories with any of the tags
        memory_ids = set()
        for tag in tags:
            memory_ids.update(self.indices['by_tag'].get(tag, []))

        return [m for m in self.memories if m['id'] in memory_ids]

    def get_recent(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get recent memories within specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [m for m in self.memories if m['timestamp'] >= cutoff]
        return recent[-limit:]  # Most recent first

    def get_summary_by_period(
        self,
        start: datetime,
        end: datetime,
        period: str = 'day'
    ) -> Dict[str, List[Dict]]:
        """
        Group memories by time period.

        Args:
            start, end: Date range
            period: 'hour', 'day', 'week', 'month'

        Returns:
            Dict mapping period -> memories
        """
        memories = self.get_by_date_range(start, end)
        grouped = defaultdict(list)

        for memory in memories:
            if period == 'hour':
                key = memory['timestamp'].strftime('%Y-%m-%d %H:00')
            elif period == 'day':
                key = memory['timestamp'].strftime('%Y-%m-%d')
            elif period == 'week':
                key = memory['timestamp'].strftime('%Y-W%W')
            elif period == 'month':
                key = memory['timestamp'].strftime('%Y-%m')
            else:
                key = str(memory['timestamp'])

            grouped[key].append(memory)

        return dict(grouped)

    def find_patterns(self, min_interval_hours: int = 24) -> List[Dict]:
        """Find recurring patterns in memories."""
        patterns = []

        # Group by content similarity
        content_groups = defaultdict(list)
        for memory in self.memories:
            # Simple grouping by first few words
            key = ' '.join(memory['content'].split()[:5])
            content_groups[key].append(memory)

        # Find recurring patterns
        for content, memories in content_groups.items():
            if len(memories) >= 2:
                intervals = []
                for i in range(len(memories) - 1):
                    interval = (memories[i+1]['timestamp'] - memories[i]['timestamp']).total_seconds() / 3600
                    intervals.append(interval)

                if intervals and max(intervals) >= min_interval_hours:
                    patterns.append({
                        'content_pattern': content,
                        'occurrences': len(memories),
                        'avg_interval_hours': sum(intervals) / len(intervals),
                        'memories': memories
                    })

        return patterns

# Usage
ts_memory = TimeSeriesMemory()

# Add time-series memories
ts_memory.add_memory(
    "User logged in",
    timestamp=datetime(2025, 1, 15, 9, 0),
    tags=['auth', 'user'],
    memory_type='event'
)

ts_memory.add_memory(
    "User completed tutorial",
    timestamp=datetime(2025, 1, 15, 9, 30),
    tags=['tutorial', 'user'],
    memory_type='event'
)

ts_memory.add_memory(
    "User logged in",
    timestamp=datetime(2025, 1, 16, 9, 5),
    tags=['auth', 'user'],
    memory_type='event'
)

# Query by date range
recent = ts_memory.get_recent(hours=48)
print(f"Recent memories: {len(recent)}")

# Get timeline
timeline = ts_memory.get_timeline(tags=['auth'])
print(f"Auth timeline: {len(timeline)} events")

# Find patterns
patterns = ts_memory.find_patterns()
for pattern in patterns:
    print(f"\nPattern: {pattern['content_pattern']}")
    print(f"  Occurs every {pattern['avg_interval_hours']:.1f} hours")
```

## Hierarchical Memory System

Multi-level memory with working, short-term, and long-term storage:

```python
class HierarchicalMemory:
    """
    Three-tier memory system mimicking human memory:
    - Working memory (immediate, limited capacity)
    - Short-term memory (recent, larger capacity)
    - Long-term memory (persistent, unlimited)
    """

    def __init__(
        self,
        working_capacity: int = 5,
        short_term_capacity: int = 50,
        consolidation_threshold: int = 3  # How many times to see before long-term
    ):
        self.working = []  # Most recent items
        self.short_term = []  # Recently accessed
        self.long_term = {}  # Persistent storage
        self.access_counts = defaultdict(int)

        self.working_capacity = working_capacity
        self.short_term_capacity = short_term_capacity
        self.consolidation_threshold = consolidation_threshold

    def add(self, content: str, metadata: Optional[Dict] = None):
        """Add new memory."""
        memory = {
            'content': content,
            'metadata': metadata or {},
            'id': hash(content),
            'timestamp': time.time()
        }

        # Add to working memory
        self.working.append(memory)

        # Maintain working memory capacity
        if len(self.working) > self.working_capacity:
            # Move oldest to short-term
            moved = self.working.pop(0)
            self.short_term.append(moved)

        # Maintain short-term capacity
        if len(self.short_term) > self.short_term_capacity:
            # Move least accessed to long-term
            self._consolidate_to_long_term()

    def recall(self, query: str) -> Optional[Dict]:
        """
        Recall memory, checking each tier.
        Accessing a memory increases its importance.
        """
        # Check working memory first (fastest)
        for memory in self.working:
            if query in memory['content']:
                self.access_counts[memory['id']] += 1
                return memory

        # Check short-term memory
        for memory in self.short_term:
            if query in memory['content']:
                self.access_counts[memory['id']] += 1
                # Move to working memory (recent access)
                self.short_term.remove(memory)
                self.working.append(memory)
                return memory

        # Check long-term memory
        for memory_id, memory in self.long_term.items():
            if query in memory['content']:
                self.access_counts[memory_id] += 1
                # Reactivate to working memory
                self.working.append(memory)
                return memory

        return None

    def _consolidate_to_long_term(self):
        """Move frequently accessed short-term memories to long-term."""
        # Sort by access count
        sorted_memories = sorted(
            self.short_term,
            key=lambda m: self.access_counts[m['id']],
            reverse=True
        )

        # Move frequently accessed to long-term
        for memory in sorted_memories[:10]:
            if self.access_counts[memory['id']] >= self.consolidation_threshold:
                self.long_term[memory['id']] = memory
                self.short_term.remove(memory)

        # Remove least accessed
        while len(self.short_term) > self.short_term_capacity:
            least_accessed = min(self.short_term, key=lambda m: self.access_counts[m['id']])
            self.short_term.remove(least_accessed)

    def get_status(self) -> Dict:
        """Get memory system status."""
        return {
            'working_memory': {
                'count': len(self.working),
                'capacity': self.working_capacity,
                'items': [m['content'][:50] for m in self.working]
            },
            'short_term_memory': {
                'count': len(self.short_term),
                'capacity': self.short_term_capacity
            },
            'long_term_memory': {
                'count': len(self.long_term)
            }
        }

# Usage
hmem = HierarchicalMemory()

# Add memories
for i in range(10):
    hmem.add(f"Memory item {i}")

# Access some memories repeatedly
for _ in range(4):
    hmem.recall("item 5")

# Check status
status = hmem.get_status()
print(f"Working: {status['working_memory']['count']}")
print(f"Short-term: {status['short_term_memory']['count']}")
print(f"Long-term: {status['long_term_memory']['count']}")
```

## Integrating Custom Memory with Letta

```python
from letta import create_client
from letta.schemas.memory import Memory

class CustomLettaMemory:
    """
    Adapter to use custom memory backends with Letta.
    """

    def __init__(self, custom_backend):
        self.backend = custom_backend
        self.client = create_client()

    def create_agent_with_custom_memory(self, name: str, persona: str):
        """Create Letta agent using custom memory backend."""

        # Create base agent
        agent = self.client.create_agent(
            name=name,
            persona=persona
        )

        # Store agent ID for memory association
        self.agent_id = agent.id

        return agent

    def enhanced_send_message(self, message: str, agent_id: str) -> Dict:
        """
        Send message with custom memory integration.
        """
        # Before sending: retrieve relevant custom memories
        relevant_memories = self.backend.search(message, limit=3)

        # Inject relevant memories into context
        context = "Relevant memories:\n"
        for mem in relevant_memories:
            context += f"- {mem['content']}\n"

        enhanced_message = f"{context}\n\nUser message: {message}"

        # Send to Letta agent
        response = self.client.send_message(
            agent_id=agent_id,
            message=enhanced_message,
            role="user"
        )

        # After response: store new memory
        response_text = self._extract_response(response)
        self.backend.add_memory(
            f"Q: {message} A: {response_text}",
            metadata={'agent_id': agent_id}
        )

        return {
            'response': response_text,
            'relevant_memories': relevant_memories
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Example: Using graph memory with Letta
graph_mem = GraphMemoryBackend()
custom_agent = CustomLettaMemory(graph_mem)

agent = custom_agent.create_agent_with_custom_memory(
    name="GraphMemoryAgent",
    persona="You are an assistant with graph-based memory"
)

# Use agent with enhanced memory
result = custom_agent.enhanced_send_message(
    "Tell me about Python",
    agent.id
)

print(result['response'])
print(f"\nRetrieved {len(result['relevant_memories'])} relevant memories")
```

## Performance Optimization

### 1. Memory Caching

```python
from functools import lru_cache
import hashlib

class CachedMemory:
    """Add caching layer to memory backend."""

    def __init__(self, backend):
        self.backend = backend
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=128)
    def _query_hash(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search with caching."""
        cache_key = f"{self._query_hash(query)}_{limit}"

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        results = self.backend.search(query, limit)
        self._cache[cache_key] = results

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self._cache.keys())[:100]
            for key in oldest_keys:
                del self._cache[key]

        return results

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }
```

### 2. Async Memory Operations

```python
import asyncio

class AsyncMemory:
    """Async memory operations for better performance."""

    def __init__(self, backend):
        self.backend = backend

    async def search_async(self, query: str) -> List[Dict]:
        """Async search operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.backend.search,
            query
        )

    async def batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """Search multiple queries in parallel."""
        tasks = [self.search_async(q) for q in queries]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    async_mem = AsyncMemory(graph_memory)

    results = await async_mem.batch_search([
        "Python programming",
        "Web frameworks",
        "Machine learning"
    ])

    for i, result_set in enumerate(results):
        print(f"Query {i}: {len(result_set)} results")

asyncio.run(main())
```

## Memory Compression

For large-scale deployments:

```python
import gzip
import pickle

class CompressedMemory:
    """Compress memories to save storage space."""

    def __init__(self):
        self.memories = {}

    def add_memory(self, content: str, metadata: Dict) -> int:
        """Add memory with compression."""
        memory_id = len(self.memories)

        # Serialize and compress
        data = {'content': content, 'metadata': metadata}
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)

        self.memories[memory_id] = compressed

        return memory_id

    def get_memory(self, memory_id: int) -> Dict:
        """Retrieve and decompress memory."""
        if memory_id not in self.memories:
            return None

        compressed = self.memories[memory_id]
        serialized = gzip.decompress(compressed)
        data = pickle.loads(serialized)

        return data

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if not self.memories:
            return 0

        total_compressed = sum(len(m) for m in self.memories.values())
        # Estimate original size
        sample = self.get_memory(0)
        original_size = len(pickle.dumps(sample))
        estimated_original = original_size * len(self.memories)

        return total_compressed / estimated_original if estimated_original > 0 else 0
```

## Next Steps

You've mastered custom memory architectures! Next, we'll explore **RAG systems with Letta** - building retrieval-augmented generation systems.

## Resources

- [Graph Databases in Python](https://networkx.org/)
- [Vector Similarity Search](https://www.pinecone.io/learn/vector-similarity/)
- [Memory-Augmented Neural Networks](https://arxiv.org/abs/1410.5401)

Build memory systems that scale!

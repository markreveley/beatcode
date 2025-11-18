---
title: "Performance Optimization for Letta Agents at Scale"
date: "2025-02-24"
description: "Advanced techniques for optimizing Letta agent performance, reducing latency, and scaling to millions of users."
tags: ["Letta", "Performance", "Optimization", "Scaling", "Production"]
---

# Performance Optimization for Letta Agents at Scale

When Letta agents serve millions of users, performance becomes critical. This tutorial covers optimization techniques for production-scale deployments.

## Caching Strategies

```python
from functools import lru_cache
import redis

class CachedLettaClient:
    """Letta client with multi-tier caching."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        
        # L1: In-memory cache
        self.memory_cache = {}
        
        # L2: Redis cache
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
    
    @lru_cache(maxsize=1000)
    def get_agent_cached(self, agent_id: str):
        """Cache agent metadata."""
        return self.client.get_agent(agent_id)
    
    def send_message_cached(self, message: str) -> Dict:
        """Send message with response caching."""
        
        # Generate cache key
        cache_key = f"response:{hash(message)}"
        
        # Check L1 cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check L2 cache
        cached = self.redis.get(cache_key)
        if cached:
            response = json.loads(cached)
            self.memory_cache[cache_key] = response
            return response
        
        # Cache miss - call agent
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=message,
            role="user"
        )
        
        # Store in caches
        self.memory_cache[cache_key] = response
        self.redis.setex(cache_key, 3600, json.dumps(response))
        
        return response
```

## Connection Pooling

```python
from queue import Queue
import threading

class ConnectionPool:
    """Pool of Letta client connections."""
    
    def __init__(self, pool_size: int = 10):
        self.pool = Queue(maxsize=pool_size)
        
        # Initialize pool
        for _ in range(pool_size):
            self.pool.put(create_client())
    
    def get_client(self):
        """Get client from pool."""
        return self.pool.get()
    
    def return_client(self, client):
        """Return client to pool."""
        self.pool.put(client)
    
    def execute(self, func, *args, **kwargs):
        """Execute function with pooled client."""
        client = self.get_client()
        try:
            result = func(client, *args, **kwargs)
            return result
        finally:
            self.return_client(client)
```

## Batch Processing

```python
class BatchProcessor:
    """Process multiple requests in batches."""
    
    def __init__(self, agent_id: str, batch_size: int = 10):
        self.agent_id = agent_id
        self.client = create_client()
        self.batch_size = batch_size
        self.queue = Queue()
    
    def process_batch(self, messages: List[str]) -> List[Dict]:
        """Process multiple messages efficiently."""
        
        # Process in batches
        results = []
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i+self.batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_single_batch(self, batch: List[str]) -> List[Dict]:
        """Process single batch in parallel."""
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(self._send_message, msg)
                for msg in batch
            ]
            return [f.result() for f in futures]
```

## Database Query Optimization

```python
class OptimizedMemoryAccess:
    """Optimized database access for agent memory."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
    
    def bulk_load_memories(self, memory_ids: List[str]) -> List[Dict]:
        """Load multiple memories in single query."""
        # Use database-specific bulk operations
        pass
    
    def lazy_load_messages(self, limit: int = 50):
        """Lazy load messages with pagination."""
        offset = 0
        while True:
            messages = self.client.get_messages(
                agent_id=self.agent_id,
                limit=limit,
                offset=offset
            )
            
            if not messages:
                break
            
            yield messages
            offset += limit
```

## Resource Monitoring

```python
import psutil
import time

class PerformanceMonitor:
    """Monitor agent performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "latencies": [],
            "errors": 0
        }
    
    def track_request(self, func):
        """Decorator to track request performance."""
        def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                self.metrics["requests"] += 1
                latency = (time.time() - start) * 1000
                self.metrics["latencies"].append(latency)
                return result
            except Exception as e:
                self.metrics["errors"] += 1
                raise e
        
        return wrapper
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        latencies = self.metrics["latencies"]
        
        return {
            "total_requests": self.metrics["requests"],
            "error_rate": self.metrics["errors"] / max(self.metrics["requests"], 1),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024
        }
```

## Load Balancing

```python
import random

class LoadBalancer:
    """Distribute load across multiple agent instances."""
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.client = create_client()
        self.request_counts = {aid: 0 for aid in agent_ids}
    
    def select_agent(self, strategy: str = "round_robin") -> str:
        """Select agent based on load balancing strategy."""
        
        if strategy == "round_robin":
            # Simple round-robin
            agent_id = self.agent_ids[sum(self.request_counts.values()) % len(self.agent_ids)]
        
        elif strategy == "least_connections":
            # Select agent with fewest active requests
            agent_id = min(self.request_counts, key=self.request_counts.get)
        
        elif strategy == "random":
            agent_id = random.choice(self.agent_ids)
        
        return agent_id
    
    def send_message(self, message: str) -> Dict:
        """Send message to least loaded agent."""
        agent_id = self.select_agent("least_connections")
        
        self.request_counts[agent_id] += 1
        try:
            response = self.client.send_message(
                agent_id=agent_id,
                message=message,
                role="user"
            )
            return response
        finally:
            self.request_counts[agent_id] -= 1
```

## Next Steps

Learn about **Testing and Quality Assurance** for Letta agents.

## Resources
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Redis Caching Best Practices](https://redis.io/docs/manual/patterns/)

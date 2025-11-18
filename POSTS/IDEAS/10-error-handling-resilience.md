---
title: "Error Handling and Resilience in Letta Agents: Building Robust Systems"
date: "2025-02-09"
description: "Master error handling, failover strategies, and resilience patterns to build production-ready Letta agents that gracefully handle failures."
tags: ["Letta", "Error Handling", "Resilience", "Production", "Best Practices"]
---

# Error Handling and Resilience in Letta Agents: Building Robust Systems

Production systems fail. APIs timeout. Databases disconnect. LLMs hallucinate. In this tutorial, we'll build **resilient Letta agents** that handle errors gracefully and recover from failures automatically.

## The Error Landscape

Common failures in Letta systems:

```
┌────────────────────────────────────┐
│  LLM Provider Issues               │
│  - Rate limits                     │
│  - Timeouts                        │
│  - Service outages                 │
│  - Invalid responses               │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│  Memory/Database Failures          │
│  - Connection errors               │
│  - Query failures                  │
│  - Disk space issues               │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│  Tool Execution Errors             │
│  - API failures                    │
│  - Permission errors               │
│  - Invalid parameters              │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│  Agent Logic Issues                │
│  - Infinite loops                  │
│  - Context overflow                │
│  - Hallucinations                  │
└────────────────────────────────────┘
```

## Layer 1: Graceful Degradation

Basic error handling with fallbacks:

```python
from letta import create_client
from typing import Optional, Dict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResilientAgent:
    """Agent wrapper with error handling and fallbacks."""

    def __init__(self, agent_id: str, fallback_responses: Optional[Dict] = None):
        self.client = create_client()
        self.agent_id = agent_id
        self.fallback_responses = fallback_responses or {
            "rate_limit": "I'm experiencing high load. Please try again in a moment.",
            "timeout": "That took longer than expected. Could you rephrase your question?",
            "generic": "I encountered an error. Let me try a different approach."
        }
        self.error_count = 0
        self.max_errors = 3

    def send_message(self, message: str, max_retries: int = 3) -> Dict:
        """Send message with retry logic and graceful degradation."""

        for attempt in range(max_retries):
            try:
                # Attempt to send message
                response = self.client.send_message(
                    agent_id=self.agent_id,
                    message=message,
                    role="user"
                )

                # Reset error count on success
                self.error_count = 0

                return {
                    "status": "success",
                    "response": self._extract_response(response),
                    "attempt": attempt + 1
                }

            except Exception as e:
                self.error_count += 1
                error_type = self._classify_error(e)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {error_type} - {str(e)}"
                )

                # If this isn't the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    wait_time = self._calculate_backoff(attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, use fallback
                    return self._handle_failure(error_type, message, e)

        # Should never reach here, but safety fallback
        return self._handle_failure("generic", message, None)

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling."""
        error_msg = str(error).lower()

        if "rate limit" in error_msg or "429" in error_msg:
            return "rate_limit"
        elif "timeout" in error_msg or "timed out" in error_msg:
            return "timeout"
        elif "connection" in error_msg:
            return "connection"
        elif "authentication" in error_msg or "401" in error_msg:
            return "auth"
        else:
            return "generic"

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time."""
        return min(2 ** attempt, 32)  # Max 32 seconds

    def _handle_failure(self, error_type: str, message: str, error: Exception) -> Dict:
        """Handle failure with appropriate fallback."""

        # Check if we should enter safe mode
        if self.error_count >= self.max_errors:
            logger.error("Too many errors, entering safe mode")
            return {
                "status": "safe_mode",
                "response": "I'm experiencing technical difficulties. Please contact support.",
                "error_count": self.error_count
            }

        # Return appropriate fallback response
        fallback = self.fallback_responses.get(
            error_type,
            self.fallback_responses["generic"]
        )

        return {
            "status": "fallback",
            "response": fallback,
            "error_type": error_type,
            "original_error": str(error) if error else None
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
agent = create_client().create_agent(name="ResilientBot")
resilient = ResilientAgent(agent.id)

result = resilient.send_message("Hello!")
print(f"Status: {result['status']}")
print(f"Response: {result['response']}")
```

## Layer 2: Circuit Breaker Pattern

Prevent cascading failures:

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject requests
    - HALF_OPEN: Testing recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")

        try:
            result = func(*args, **kwargs)

            # Success - reset if in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self._reset()

            return result

        except self.expected_exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (
            self.last_failure_time is not None and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )

    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        logger.info("Circuit breaker reset to CLOSED state")

    def get_state(self) -> Dict:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }

# Usage with Letta
class ProtectedLettaClient:
    """Letta client with circuit breaker protection."""

    def __init__(self):
        self.client = create_client()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30
        )

    def send_message(self, agent_id: str, message: str):
        """Send message with circuit breaker protection."""
        try:
            return self.circuit_breaker.call(
                self.client.send_message,
                agent_id=agent_id,
                message=message,
                role="user"
            )
        except Exception as e:
            state = self.circuit_breaker.get_state()
            logger.error(f"Circuit breaker state: {state}")

            if state['state'] == 'open':
                return {
                    "error": "Service temporarily unavailable",
                    "retry_after": 30
                }
            raise e

protected_client = ProtectedLettaClient()
```

## Layer 3: Bulkhead Pattern

Isolate failures to prevent system-wide impact:

```python
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

class BulkheadExecutor:
    """
    Isolate agent execution to prevent one failure from affecting others.
    Each agent gets its own resource pool.
    """

    def __init__(self, max_workers_per_agent: int = 3):
        self.max_workers = max_workers_per_agent
        self.executors = {}  # agent_id -> ThreadPoolExecutor
        self.locks = {}      # agent_id -> Lock
        self.main_lock = threading.Lock()

    def get_executor(self, agent_id: str) -> ThreadPoolExecutor:
        """Get or create executor for agent."""
        with self.main_lock:
            if agent_id not in self.executors:
                self.executors[agent_id] = ThreadPoolExecutor(
                    max_workers=self.max_workers
                )
                self.locks[agent_id] = threading.Lock()

            return self.executors[agent_id]

    def execute(self, agent_id: str, func, *args, **kwargs):
        """Execute function in isolated agent pool."""
        executor = self.get_executor(agent_id)

        future = executor.submit(func, *args, **kwargs)

        try:
            # Wait for result with timeout
            result = future.result(timeout=30)
            return {"status": "success", "result": result}
        except TimeoutError:
            logger.error(f"Agent {agent_id} timed out")
            return {"status": "timeout", "agent_id": agent_id}
        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            return {"status": "error", "error": str(e)}

    def shutdown(self, agent_id: Optional[str] = None):
        """Shutdown executor(s)."""
        if agent_id:
            if agent_id in self.executors:
                self.executors[agent_id].shutdown(wait=True)
                del self.executors[agent_id]
        else:
            for executor in self.executors.values():
                executor.shutdown(wait=True)
            self.executors.clear()

# Usage
bulkhead = BulkheadExecutor(max_workers_per_agent=2)

def send_message(agent_id, message):
    client = create_client()
    return client.send_message(agent_id=agent_id, message=message, role="user")

# Execute isolated by agent
result1 = bulkhead.execute("agent-1", send_message, "agent-1", "Hello")
result2 = bulkhead.execute("agent-2", send_message, "agent-2", "Hello")

# If agent-1 fails, agent-2 is unaffected
```

## Layer 4: Health Checks and Monitoring

Proactive failure detection:

```python
from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class HealthCheck:
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None

class AgentHealthMonitor:
    """Monitor agent health and detect issues early."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.health_history = []

    def check_health(self) -> HealthCheck:
        """Perform comprehensive health check."""
        checks = [
            self._check_agent_exists(),
            self._check_response_time(),
            self._check_memory_integrity(),
            self._check_tool_availability()
        ]

        # Aggregate results
        if all(c.status == "healthy" for c in checks):
            overall_status = "healthy"
        elif any(c.status == "unhealthy" for c in checks):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        health = HealthCheck(
            name="overall",
            status=overall_status,
            message=self._generate_health_message(checks),
            timestamp=datetime.now()
        )

        self.health_history.append(health)
        return health

    def _check_agent_exists(self) -> HealthCheck:
        """Check if agent exists and is accessible."""
        start = time.time()

        try:
            agent = self.client.get_agent(self.agent_id)
            response_time = (time.time() - start) * 1000

            return HealthCheck(
                name="agent_exists",
                status="healthy",
                message="Agent is accessible",
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
        except Exception as e:
            return HealthCheck(
                name="agent_exists",
                status="unhealthy",
                message=f"Cannot access agent: {str(e)}",
                timestamp=datetime.now()
            )

    def _check_response_time(self) -> HealthCheck:
        """Check agent response time."""
        start = time.time()

        try:
            response = self.client.send_message(
                agent_id=self.agent_id,
                message="Health check ping",
                role="user"
            )

            response_time = (time.time() - start) * 1000

            if response_time < 1000:
                status = "healthy"
            elif response_time < 3000:
                status = "degraded"
            else:
                status = "unhealthy"

            return HealthCheck(
                name="response_time",
                status=status,
                message=f"Response time: {response_time:.0f}ms",
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
        except Exception as e:
            return HealthCheck(
                name="response_time",
                status="unhealthy",
                message=f"Response check failed: {str(e)}",
                timestamp=datetime.now()
            )

    def _check_memory_integrity(self) -> HealthCheck:
        """Check if agent memory is intact."""
        try:
            agent = self.client.get_agent(self.agent_id)

            # Check if core memory exists
            if not agent.memory.human or not agent.memory.persona:
                return HealthCheck(
                    name="memory_integrity",
                    status="degraded",
                    message="Core memory is incomplete",
                    timestamp=datetime.now()
                )

            return HealthCheck(
                name="memory_integrity",
                status="healthy",
                message="Memory is intact",
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheck(
                name="memory_integrity",
                status="unhealthy",
                message=f"Memory check failed: {str(e)}",
                timestamp=datetime.now()
            )

    def _check_tool_availability(self) -> HealthCheck:
        """Check if agent tools are available."""
        try:
            # This is a simplified check - extend based on your tools
            return HealthCheck(
                name="tool_availability",
                status="healthy",
                message="Tools are available",
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheck(
                name="tool_availability",
                status="unhealthy",
                message=f"Tool check failed: {str(e)}",
                timestamp=datetime.now()
            )

    def _generate_health_message(self, checks: List[HealthCheck]) -> str:
        """Generate overall health message."""
        unhealthy = [c for c in checks if c.status == "unhealthy"]
        degraded = [c for c in checks if c.status == "degraded"]

        if unhealthy:
            return f"Unhealthy: {', '.join(c.name for c in unhealthy)}"
        elif degraded:
            return f"Degraded: {', '.join(c.name for c in degraded)}"
        else:
            return "All systems operational"

    def get_health_trend(self, hours: int = 24) -> Dict:
        """Analyze health trend over time."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [h for h in self.health_history if h.timestamp > cutoff]

        if not recent:
            return {"status": "no_data"}

        healthy_count = sum(1 for h in recent if h.status == "healthy")
        degraded_count = sum(1 for h in recent if h.status == "degraded")
        unhealthy_count = sum(1 for h in recent if h.status == "unhealthy")

        total = len(recent)

        return {
            "period_hours": hours,
            "total_checks": total,
            "healthy_pct": (healthy_count / total) * 100,
            "degraded_pct": (degraded_count / total) * 100,
            "unhealthy_pct": (unhealthy_count / total) * 100,
            "trend": "improving" if healthy_count > unhealthy_count else "degrading"
        }

# Usage
monitor = AgentHealthMonitor("agent-123")

# Perform health check
health = monitor.check_health()
print(f"Status: {health.status}")
print(f"Message: {health.message}")

# Run periodic health checks
import schedule

schedule.every(5).minutes.do(lambda: monitor.check_health())

# Analyze trend
trend = monitor.get_health_trend(hours=24)
print(f"Health trend: {trend}")
```

## Layer 5: Fallback Strategies

Multiple levels of fallbacks:

```python
class FallbackChain:
    """Chain of fallback strategies for robust operation."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.strategies = [
            self._primary_strategy,
            self._cached_response_strategy,
            self._simplified_prompt_strategy,
            self._template_response_strategy,
            self._emergency_response_strategy
        ]

    def send_message(self, message: str) -> Dict:
        """Send message using fallback chain."""

        for i, strategy in enumerate(self.strategies):
            try:
                logger.info(f"Attempting strategy {i+1}: {strategy.__name__}")
                result = strategy(message)

                return {
                    "status": "success",
                    "strategy": strategy.__name__,
                    "response": result
                }

            except Exception as e:
                logger.warning(
                    f"Strategy {i+1} failed: {str(e)}"
                )
                if i == len(self.strategies) - 1:
                    # All strategies failed
                    return {
                        "status": "all_fallbacks_failed",
                        "error": str(e)
                    }
                # Try next strategy
                continue

    def _primary_strategy(self, message: str) -> str:
        """Primary: Normal agent interaction."""
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=message,
            role="user"
        )
        return self._extract_response(response)

    def _cached_response_strategy(self, message: str) -> str:
        """Fallback 1: Use cached response for similar query."""
        # Simple cache lookup - extend with actual cache implementation
        cache = {
            "hello": "Hello! How can I help you today?",
            "help": "I can assist you with various tasks. What do you need?"
        }

        # Simple matching
        for key, response in cache.items():
            if key in message.lower():
                return f"[CACHED] {response}"

        raise Exception("No cached response found")

    def _simplified_prompt_strategy(self, message: str) -> str:
        """Fallback 2: Try with simplified prompt."""
        simplified = f"Answer briefly: {message[:100]}"

        response = self.client.send_message(
            agent_id=self.agent_id,
            message=simplified,
            role="user"
        )
        return self._extract_response(response)

    def _template_response_strategy(self, message: str) -> str:
        """Fallback 3: Use template-based response."""
        templates = {
            "question": "That's an interesting question. Let me think about that...",
            "greeting": "Hello! I'm here to help.",
            "request": "I understand you'd like help with that."
        }

        # Classify message type (simplified)
        if "?" in message:
            return templates["question"]
        elif any(word in message.lower() for word in ["hello", "hi", "hey"]):
            return templates["greeting"]
        else:
            return templates["request"]

    def _emergency_response_strategy(self, message: str) -> str:
        """Fallback 4: Emergency static response."""
        return (
            "I'm experiencing technical difficulties at the moment. "
            "Your message has been noted and I'll respond as soon as possible."
        )

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
fallback_agent = FallbackChain("agent-123")
result = fallback_agent.send_message("Hello, how are you?")
print(f"Strategy used: {result['strategy']}")
print(f"Response: {result['response']}")
```

## Complete Resilient Agent System

Putting it all together:

```python
class ProductionAgent:
    """
    Production-ready agent with all resilience patterns.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()

        # Initialize resilience components
        self.circuit_breaker = CircuitBreaker()
        self.health_monitor = AgentHealthMonitor(agent_id)
        self.fallback_chain = FallbackChain(agent_id)

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_uses": 0
        }

    def send_message(self, message: str) -> Dict:
        """Send message with full resilience stack."""
        self.metrics["total_requests"] += 1

        # Check health first
        health = self.health_monitor.check_health()
        if health.status == "unhealthy":
            logger.warning("Agent unhealthy, using fallback")
            return self._use_fallback(message)

        # Try with circuit breaker protection
        try:
            response = self.circuit_breaker.call(
                self._send_with_retry,
                message
            )

            self.metrics["successful_requests"] += 1
            return response

        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Request failed: {e}")
            return self._use_fallback(message)

    def _send_with_retry(self, message: str, max_retries: int = 3) -> Dict:
        """Send with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.client.send_message(
                    agent_id=self.agent_id,
                    message=message,
                    role="user"
                )

                return {
                    "status": "success",
                    "response": self._extract_response(response),
                    "source": "primary"
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise e

    def _use_fallback(self, message: str) -> Dict:
        """Use fallback chain."""
        self.metrics["fallback_uses"] += 1
        return self.fallback_chain.send_message(message)

    def get_metrics(self) -> Dict:
        """Get agent metrics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
            "health": self.health_monitor.check_health()
        }

    def _extract_response(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
agent = create_client().create_agent(name="ProductionBot")
prod_agent = ProductionAgent(agent.id)

# Use the agent
for i in range(10):
    result = prod_agent.send_message(f"Message {i}")
    print(f"{i}: {result['status']}")

# Check metrics
metrics = prod_agent.get_metrics()
print(f"\nSuccess rate: {metrics['success_rate']:.1%}")
print(f"Fallback uses: {metrics['fallback_uses']}")
```

## Next Steps

You've mastered resilience patterns! Now you're ready for the advanced section. Next: **Building RAG Systems with Letta** - integrating external knowledge bases.

## Resources

- [Resilience4j Patterns](https://resilience4j.readme.io/docs/getting-started)
- [AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/)
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)

Build agents that never go down!

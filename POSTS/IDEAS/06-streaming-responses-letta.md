---
title: "Streaming Responses in Letta: Real-Time Agent Communication"
date: "2025-02-01"
description: "Learn how to implement streaming responses for real-time, ChatGPT-style interactions with Letta agents."
tags: ["Letta", "Streaming", "Real-time", "Tutorial", "Advanced"]
---

# Streaming Responses in Letta: Real-Time Agent Communication

When users interact with AI agents, waiting for complete responses can feel sluggish. In this tutorial, we'll implement **streaming responses** - delivering agent output token-by-token in real-time, just like ChatGPT.

## Why Streaming?

Traditional request-response:
```
User: "Explain quantum computing"
[Wait 5 seconds...]
Agent: [Complete response appears]
```

With streaming:
```
User: "Explain quantum computing"
Agent: "Quantum computing is..." [tokens appear in real-time]
```

Benefits:
- **Better UX**: Users see progress immediately
- **Perceived speed**: Feels faster even if total time is similar
- **Interruptibility**: Users can stop generation early
- **Engagement**: More conversational feel

## Setting Up Streaming

### Install Dependencies

```bash
pip install letta fastapi uvicorn sse-starlette
```

### Basic Streaming with Letta

```python
from letta import create_client
from typing import Generator

client = create_client()

def stream_agent_response(agent_id: str, message: str) -> Generator[str, None, None]:
    """
    Stream agent responses token by token.

    Args:
        agent_id: The agent's ID
        message: User message

    Yields:
        Response tokens as they're generated
    """
    # Enable streaming in the API call
    response = client.send_message(
        agent_id=agent_id,
        message=message,
        role="user",
        stream=True  # Enable streaming
    )

    # Iterate over streamed chunks
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage
agent = client.create_agent(name="StreamingAgent")

for token in stream_agent_response(agent.id, "Tell me about AI"):
    print(token, end="", flush=True)
```

## Building a Streaming API with FastAPI

Let's create a production-ready streaming API:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from letta import create_client
import json
import asyncio

app = FastAPI()
letta_client = create_client()

class StreamRequest(BaseModel):
    agent_id: str
    message: str

async def generate_stream(agent_id: str, message: str):
    """
    Async generator for streaming responses.
    """
    try:
        # Send message with streaming enabled
        response = letta_client.send_message(
            agent_id=agent_id,
            message=message,
            role="user",
            stream=True
        )

        # Stream each chunk
        for chunk in response:
            if chunk.choices[0].delta.content:
                data = {
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }
                yield f"data: {json.dumps(data)}\n\n"

                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/stream")
async def stream_message(request: StreamRequest):
    """
    Stream agent responses using Server-Sent Events.
    """
    return EventSourceResponse(
        generate_stream(request.agent_id, request.message)
    )

@app.get("/")
def root():
    return {"status": "Streaming API ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Frontend Client for Streaming

### JavaScript/TypeScript Client

```typescript
// streaming-client.ts
interface StreamChunk {
  type: 'content' | 'done' | 'error';
  content?: string;
  message?: string;
}

async function streamAgentMessage(
  agentId: string,
  message: string,
  onChunk: (content: string) => void,
  onComplete: () => void,
  onError: (error: string) => void
) {
  try {
    const response = await fetch('http://localhost:8000/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ agent_id: agentId, message }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No reader available');
    }

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data: StreamChunk = JSON.parse(line.slice(6));

          if (data.type === 'content') {
            onChunk(data.content!);
          } else if (data.type === 'done') {
            onComplete();
          } else if (data.type === 'error') {
            onError(data.message!);
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

// Usage example
streamAgentMessage(
  'agent-123',
  'Tell me about quantum computing',
  (content) => {
    // Append each chunk to UI
    document.getElementById('response')!.textContent += content;
  },
  () => {
    console.log('Stream complete');
  },
  (error) => {
    console.error('Stream error:', error);
  }
);
```

### React Hook for Streaming

```typescript
// useStreamingAgent.ts
import { useState, useCallback } from 'react';

interface UseStreamingAgentResult {
  response: string;
  isStreaming: boolean;
  error: string | null;
  sendMessage: (agentId: string, message: string) => Promise<void>;
  stop: () => void;
}

export function useStreamingAgent(): UseStreamingAgentResult {
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const sendMessage = useCallback(async (agentId: string, message: string) => {
    setResponse('');
    setError(null);
    setIsStreaming(true);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const res = await fetch('http://localhost:8000/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId, message }),
        signal: controller.signal,
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'content') {
              setResponse(prev => prev + data.content);
            } else if (data.type === 'error') {
              setError(data.message);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      setIsStreaming(false);
      setAbortController(null);
    }
  }, []);

  const stop = useCallback(() => {
    abortController?.abort();
    setIsStreaming(false);
  }, [abortController]);

  return { response, isStreaming, error, sendMessage, stop };
}

// Usage in component
function ChatComponent() {
  const { response, isStreaming, sendMessage, stop } = useStreamingAgent();

  return (
    <div>
      <div>{response}</div>
      {isStreaming && <button onClick={stop}>Stop</button>}
      <button onClick={() => sendMessage('agent-123', 'Hello!')}>
        Send
      </button>
    </div>
  );
}
```

## Advanced Streaming Patterns

### 1. Streaming with Tool Calls

```python
async def generate_stream_with_tools(agent_id: str, message: str):
    """Stream responses including tool call notifications."""

    response = letta_client.send_message(
        agent_id=agent_id,
        message=message,
        role="user",
        stream=True
    )

    for chunk in response:
        # Regular content
        if chunk.choices[0].delta.content:
            yield {
                "type": "content",
                "content": chunk.choices[0].delta.content
            }

        # Tool call started
        if chunk.choices[0].delta.function_call:
            yield {
                "type": "tool_call",
                "tool": chunk.choices[0].delta.function_call.name,
                "status": "started"
            }

        # Tool call result
        if hasattr(chunk, 'function_return'):
            yield {
                "type": "tool_result",
                "result": chunk.function_return
            }
```

### 2. Streaming with Progress Indicators

```python
async def generate_stream_with_progress(agent_id: str, message: str):
    """Stream with progress indicators for long operations."""

    # Send thinking indicator
    yield {"type": "status", "status": "thinking"}

    response = letta_client.send_message(
        agent_id=agent_id,
        message=message,
        stream=True
    )

    token_count = 0

    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            token_count += len(content.split())

            yield {
                "type": "content",
                "content": content,
                "tokens": token_count
            }

    yield {"type": "status", "status": "complete", "total_tokens": token_count}
```

### 3. Multi-Agent Streaming

```python
async def stream_multi_agent(agent_ids: list[str], message: str):
    """Stream responses from multiple agents concurrently."""

    import asyncio

    async def stream_single_agent(agent_id: str):
        for chunk in letta_client.send_message(
            agent_id=agent_id,
            message=message,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                yield {
                    "agent_id": agent_id,
                    "content": chunk.choices[0].delta.content
                }

    # Stream from all agents concurrently
    tasks = [stream_single_agent(aid) for aid in agent_ids]

    for task in asyncio.as_completed(tasks):
        async for chunk in task:
            yield chunk
```

## WebSocket Streaming Alternative

For bidirectional communication:

```python
from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for bidirectional streaming."""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Stream response back
            response = letta_client.send_message(
                agent_id=agent_id,
                message=message_data['message'],
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk.choices[0].delta.content
                    })

            # Send completion
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print(f"Client disconnected from agent {agent_id}")
```

### WebSocket Client

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/agent-123');

ws.onopen = () => {
  ws.send(JSON.stringify({ message: 'Hello!' }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'chunk') {
    appendToChat(data.content);
  } else if (data.type === 'done') {
    console.log('Response complete');
  }
};
```

## Handling Streaming Edge Cases

### 1. Connection Interruption

```python
async def resilient_stream(agent_id: str, message: str):
    """Streaming with connection retry logic."""

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = letta_client.send_message(
                agent_id=agent_id,
                message=message,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            break  # Success

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                yield f"\n\n[Error: Connection failed after {max_retries} retries]"
            else:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
```

### 2. Rate Limiting During Streams

```python
from collections import deque
import time

class StreamRateLimiter:
    def __init__(self, max_tokens_per_second: int = 50):
        self.max_tokens = max_tokens_per_second
        self.tokens = deque()

    async def check(self, token_count: int):
        """Check if we should throttle."""
        now = time.time()

        # Remove tokens older than 1 second
        while self.tokens and self.tokens[0] < now - 1:
            self.tokens.popleft()

        # Check rate
        if len(self.tokens) >= self.max_tokens:
            wait_time = 1 - (now - self.tokens[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.tokens.append(now)

# Usage
limiter = StreamRateLimiter(max_tokens_per_second=50)

async def rate_limited_stream(agent_id: str, message: str):
    response = letta_client.send_message(agent_id=agent_id, message=message, stream=True)

    for chunk in response:
        if chunk.choices[0].delta.content:
            await limiter.check(len(chunk.choices[0].delta.content))
            yield chunk.choices[0].delta.content
```

## Performance Optimization

### 1. Chunking for Better UX

```python
async def chunked_stream(agent_id: str, message: str, chunk_size: int = 5):
    """Accumulate tokens before sending to reduce overhead."""

    buffer = ""
    response = letta_client.send_message(agent_id=agent_id, message=message, stream=True)

    for chunk in response:
        if chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content

            if len(buffer) >= chunk_size:
                yield buffer
                buffer = ""

    # Send remaining buffer
    if buffer:
        yield buffer
```

### 2. Compression for Large Streams

```python
import gzip

async def compressed_stream(agent_id: str, message: str):
    """Compress streamed content for bandwidth efficiency."""

    response = letta_client.send_message(agent_id=agent_id, message=message, stream=True)

    for chunk in response:
        if chunk.choices[0].delta.content:
            # Compress each chunk
            compressed = gzip.compress(chunk.choices[0].delta.content.encode())
            yield compressed
```

## Testing Streaming Endpoints

```python
import pytest
from fastapi.testclient import TestClient

def test_streaming_endpoint():
    """Test streaming response."""
    client = TestClient(app)

    with client.stream(
        "POST",
        "/stream",
        json={"agent_id": "test-agent", "message": "Hello"}
    ) as response:
        chunks = []
        for line in response.iter_lines():
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if data['type'] == 'content':
                    chunks.append(data['content'])

        full_response = ''.join(chunks)
        assert len(full_response) > 0
        assert 'Hello' in full_response or len(chunks) > 0
```

## Complete Example: Chat Application

```python
# streaming_chat.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from letta import create_client
import json

app = FastAPI()
letta_client = create_client()

# Create a default agent
agent = letta_client.create_agent(
    name="ChatAgent",
    persona="You are a helpful, friendly assistant."
)

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive user message
            message = await websocket.receive_text()

            # Stream response
            response = letta_client.send_message(
                agent_id=agent.id,
                message=message,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk.choices[0].delta.content
                    })

            await websocket.send_json({"type": "done"})

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

# Mount static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

Frontend HTML:

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Streaming Chat</title>
    <style>
        #messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        #input { width: 80%; }
    </style>
</head>
<body>
    <div id="messages"></div>
    <input id="input" type="text" placeholder="Type a message...">
    <button onclick="send()">Send</button>

    <script>
        const ws = new WebSocket('ws://localhost:8000/chat');
        const messages = document.getElementById('messages');
        const input = document.getElementById('input');
        let currentMessage = null;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'chunk') {
                if (!currentMessage) {
                    currentMessage = document.createElement('div');
                    messages.appendChild(currentMessage);
                }
                currentMessage.textContent += data.content;
                messages.scrollTop = messages.scrollHeight;
            } else if (data.type === 'done') {
                currentMessage = null;
            }
        };

        function send() {
            const message = input.value;
            if (!message) return;

            const userDiv = document.createElement('div');
            userDiv.textContent = 'You: ' + message;
            userDiv.style.fontWeight = 'bold';
            messages.appendChild(userDiv);

            ws.send(message);
            input.value = '';
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });
    </script>
</body>
</html>
```

## Monitoring Streaming Performance

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class StreamMetrics:
    total_tokens: int = 0
    start_time: float = 0
    first_token_time: float = 0
    end_time: float = 0

    @property
    def time_to_first_token(self) -> float:
        return self.first_token_time - self.start_time if self.first_token_time else 0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.total_time if self.total_time else 0

async def measured_stream(agent_id: str, message: str):
    """Stream with performance metrics."""
    metrics = StreamMetrics(start_time=time.time())

    response = letta_client.send_message(agent_id=agent_id, message=message, stream=True)

    for chunk in response:
        if chunk.choices[0].delta.content:
            if metrics.total_tokens == 0:
                metrics.first_token_time = time.time()

            content = chunk.choices[0].delta.content
            metrics.total_tokens += len(content.split())

            yield content

    metrics.end_time = time.time()

    # Log metrics
    print(f"Time to first token: {metrics.time_to_first_token:.2f}s")
    print(f"Total time: {metrics.total_time:.2f}s")
    print(f"Tokens/second: {metrics.tokens_per_second:.2f}")
```

## Next Steps

You now know how to implement streaming responses in Letta! Next tutorial, we'll explore **multi-agent systems** - orchestrating multiple agents to work together.

## Resources

- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Letta Streaming API Reference](https://docs.letta.ai/streaming)

Happy streaming!

---
title: "Deploying Letta Agents to Production: From Local Development to Live Services"
date: "2025-01-30"
description: "A comprehensive guide to deploying Letta agents as production services with REST APIs, authentication, monitoring, and scalability."
tags: ["Letta", "Deployment", "Production", "REST API", "DevOps"]
---

# Deploying Letta Agents to Production: From Local Development to Live Services

You've built an amazing Letta agent, tested it locally, and it's working perfectly. Now what? In this tutorial, we'll take your agent from development to production, building a REST API, adding authentication, implementing monitoring, and deploying to the cloud.

## Deployment Architecture Overview

Here's what we'll build:

```
┌─────────────┐
│   Client    │
│ (Web/Mobile)│
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────────┐
│      Load Balancer/CDN          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   FastAPI Server                │
│   - Authentication              │
│   - Rate Limiting               │
│   - Request Validation          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Letta Agent                   │
│   - Memory Management           │
│   - Tool Execution              │
│   - Response Generation         │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   PostgreSQL Database           │
│   - Agent Memory                │
│   - User Data                   │
│   - Conversation History        │
└─────────────────────────────────┘
```

## Step 1: Build a REST API with FastAPI

Let's create a production-ready REST API for your Letta agent.

### Install Dependencies

```bash
pip install fastapi uvicorn letta python-dotenv pydantic
```

### Create the API

Create `main.py`:

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from letta import create_client
from letta.schemas.llm_config import LLMConfig
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Letta Agent API",
    description="Production API for Letta AI agents",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Letta client
letta_client = create_client()

# Request/Response Models
class MessageRequest(BaseModel):
    message: str
    agent_id: Optional[str] = None
    user_id: str

class MessageResponse(BaseModel):
    response: str
    agent_id: str
    usage: dict

class CreateAgentRequest(BaseModel):
    name: str
    persona: str
    user_id: str

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    created_at: str

# API Routes
@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Letta Agent API"}

@app.post("/agent/create", response_model=AgentResponse)
def create_agent(request: CreateAgentRequest):
    """Create a new Letta agent for a user."""
    try:
        agent = letta_client.create_agent(
            name=request.name,
            persona=request.persona,
            human=f"User ID: {request.user_id}"
        )

        return AgentResponse(
            agent_id=agent.id,
            name=agent.name,
            created_at=agent.created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/message", response_model=MessageResponse)
def send_message(request: MessageRequest):
    """Send a message to an agent and get a response."""
    try:
        # Get or create agent
        if not request.agent_id:
            # Create a default agent for this user
            agent = letta_client.create_agent(
                name=f"Agent_{request.user_id}",
                persona="You are a helpful AI assistant."
            )
            agent_id = agent.id
        else:
            agent_id = request.agent_id

        # Send message
        response = letta_client.send_message(
            agent_id=agent_id,
            message=request.message,
            role="user"
        )

        # Extract assistant's response
        assistant_messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        response_text = " ".join(assistant_messages)

        return MessageResponse(
            response=response_text,
            agent_id=agent_id,
            usage=response.usage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    """Get agent details."""
    try:
        agent = letta_client.get_agent(agent_id)
        return {
            "agent_id": agent.id,
            "name": agent.name,
            "created_at": agent.created_at,
            "core_memory": {
                "human": agent.memory.human,
                "persona": agent.memory.persona
            }
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/agent/{agent_id}")
def delete_agent(agent_id: str):
    """Delete an agent."""
    try:
        letta_client.delete_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the API

```bash
python main.py
```

Visit `http://localhost:8000/docs` to see the interactive API documentation!

### Test the API

```bash
# Create an agent
curl -X POST "http://localhost:8000/agent/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MyAgent",
    "persona": "You are a helpful assistant",
    "user_id": "user123"
  }'

# Send a message
curl -X POST "http://localhost:8000/agent/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "user_id": "user123",
    "agent_id": "agent-id-here"
  }'
```

## Step 2: Add Authentication

Security is crucial in production. Let's add API key authentication.

### Install JWT Dependencies

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

### Add Authentication

Create `auth.py`:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
import os

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In production, store these in a database
VALID_API_KEYS = {
    os.getenv("API_KEY_1"): "user_1",
    os.getenv("API_KEY_2"): "user_2",
}

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key and return user ID."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )

    user_id = VALID_API_KEYS.get(api_key)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    return user_id
```

### Update main.py to Use Authentication

```python
from auth import get_api_key

@app.post("/agent/message", response_model=MessageResponse)
def send_message(
    request: MessageRequest,
    user_id: str = Depends(get_api_key)  # Requires valid API key
):
    """Send a message to an agent (authenticated)."""
    # Verify user owns this agent
    if request.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # ... rest of the code
```

### Test with Authentication

```bash
export API_KEY_1="your-secret-key-here"

curl -X POST "http://localhost:8000/agent/message" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{
    "message": "Hello!",
    "user_id": "user_1"
  }'
```

## Step 3: Add Rate Limiting

Prevent abuse with rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/agent/message")
@limiter.limit("20/minute")  # 20 requests per minute
def send_message(request: Request, message_data: MessageRequest):
    # ... your code
    pass
```

## Step 4: Environment Configuration

Create `.env` file for configuration:

```bash
# .env
OPENAI_API_KEY=your-openai-key
DATABASE_URL=postgresql://user:pass@localhost/letta
API_KEY_1=secret-key-1
API_KEY_2=secret-key-2
ENVIRONMENT=production
LOG_LEVEL=INFO
```

Load in `main.py`:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()
```

## Step 5: Add Logging and Monitoring

Create `logging_config.py`:

```python
import logging
import sys
from datetime import datetime

def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
```

Add logging to your endpoints:

```python
from logging_config import logger

@app.post("/agent/message")
def send_message(request: MessageRequest):
    logger.info(f"Message received from user {request.user_id}")

    try:
        # ... process message
        logger.info(f"Message processed successfully for agent {agent_id}")
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## Step 6: Database Configuration

Configure Letta to use PostgreSQL for production:

```bash
pip install psycopg2-binary
```

Set up Letta with PostgreSQL:

```python
from letta import create_client
from letta.config import LettaConfig

config = LettaConfig(
    database_url="postgresql://user:password@localhost:5432/letta",
    storage_type="postgres"
)

letta_client = create_client(config=config)
```

## Step 7: Containerize with Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://letta:password@db:5432/letta
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=letta
      - POSTGRES_USER=letta
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

Run with Docker:

```bash
docker-compose up -d
```

## Step 8: Deploy to Cloud

### Option 1: Deploy to Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Create app
fly launch

# Set secrets
fly secrets set OPENAI_API_KEY=your-key

# Deploy
fly deploy
```

### Option 2: Deploy to Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add PostgreSQL
railway add

# Deploy
railway up
```

### Option 3: Deploy to AWS (ECS)

Create `task-definition.json`:

```json
{
  "family": "letta-agent-api",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-ecr-repo/letta-api:latest",
      "memory": 512,
      "cpu": 256,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://..."
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:..."
        }
      ]
    }
  ]
}
```

## Step 9: Monitoring and Observability

### Add Prometheus Metrics

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Add metrics
Instrumentator().instrument(app).expose(app)
```

### Add Error Tracking with Sentry

```bash
pip install sentry-sdk
```

```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    environment=config.ENVIRONMENT,
    traces_sample_rate=1.0,
)
```

## Step 10: Production Checklist

Before going live:

- [ ] Environment variables secured
- [ ] Database backups configured
- [ ] Rate limiting enabled
- [ ] Authentication implemented
- [ ] Logging configured
- [ ] Error tracking set up
- [ ] Health checks working
- [ ] HTTPS/SSL configured
- [ ] CORS configured properly
- [ ] API documentation generated
- [ ] Load testing completed
- [ ] Monitoring dashboards created

## Performance Optimization

### 1. Add Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_agent_cached(agent_id: str):
    return letta_client.get_agent(agent_id)
```

### 2. Use Async Endpoints

```python
@app.post("/agent/message")
async def send_message(request: MessageRequest):
    # Use async operations where possible
    return await process_message_async(request)
```

### 3. Connection Pooling

Configure database connection pooling for better performance.

## Scaling Strategies

### Horizontal Scaling
- Deploy multiple API instances behind a load balancer
- Use session affinity if needed
- Share database across instances

### Vertical Scaling
- Increase server resources (CPU, RAM)
- Optimize database queries
- Use caching layers (Redis)

## Monitoring Production Agents

Create a monitoring dashboard:

```python
@app.get("/metrics/agents")
def get_agent_metrics():
    """Get aggregate agent metrics."""
    agents = letta_client.list_agents()

    return {
        "total_agents": len(agents),
        "active_today": count_active_agents(),
        "total_messages": count_total_messages(),
        "average_response_time": get_avg_response_time()
    }
```

## Troubleshooting Production Issues

### Common Issues

1. **High latency**: Check database queries, add caching
2. **Memory leaks**: Monitor memory usage, restart containers
3. **Rate limit exceeded**: Adjust limits or upgrade plan
4. **Authentication failures**: Verify API keys and JWT tokens

### Debugging Tools

```python
@app.get("/debug/agent/{agent_id}")
async def debug_agent(agent_id: str):
    """Debug endpoint for agent inspection."""
    agent = letta_client.get_agent(agent_id)

    return {
        "agent": agent.to_dict(),
        "memory": {
            "human": agent.memory.human,
            "persona": agent.memory.persona
        },
        "recent_messages": letta_client.get_messages(agent_id, limit=10)
    }
```

## Cost Optimization

- Use cheaper models (gpt-3.5-turbo) for development
- Cache frequently requested responses
- Implement message queuing for batch processing
- Monitor and set OpenAI API budgets

## Security Best Practices

1. **Never commit secrets** - Use environment variables
2. **Validate all inputs** - Prevent injection attacks
3. **Use HTTPS only** - Encrypt data in transit
4. **Implement rate limiting** - Prevent abuse
5. **Log security events** - Track unauthorized access
6. **Regular updates** - Keep dependencies current

## Conclusion

You've now learned how to:
- Build a production REST API for Letta agents
- Add authentication and security
- Deploy to cloud platforms
- Monitor and scale your service
- Handle production issues

Your Letta agents are now ready for the real world!

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Letta Production Guide](https://docs.letta.ai/production)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS ECS Guide](https://aws.amazon.com/ecs/)

## Next Steps

- Implement WebSocket support for real-time chat
- Add multi-tenancy for SaaS applications
- Build a frontend client
- Integrate analytics and A/B testing

Congratulations on completing this Letta tutorial series! 🎉

---
title: "Building SaaS Applications with Letta: From Idea to Production"
date: "2025-03-02"
description: "Build complete SaaS applications powered by Letta agents - from architecture to billing, auth to deployment."
tags: ["Letta", "SaaS", "Production", "Expert", "Full Stack"]
---

# Building SaaS Applications with Letta

Ready to build a complete SaaS product with Letta? This expert tutorial covers the full stack - from architecture to monetization.

## SaaS Architecture

```
┌──────────────────────────────────────┐
│        Frontend (React/Next.js)      │
└────────────┬─────────────────────────┘
             │
┌────────────▼─────────────────────────┐
│      API Gateway (FastAPI)           │
│   - Authentication                   │
│   - Rate Limiting                    │
│   - Request Routing                  │
└────────────┬─────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────────┐
│  Letta   │  │  PostgreSQL  │
│  Agents  │  │  - Users     │
│          │  │  - Billing   │
└──────────┘  │  - Logs      │
              └──────────────┘
```

## Multi-Tenant Agent System

```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
import stripe

class SaaSAgentPlatform:
    """Multi-tenant Letta agent platform."""
    
    def __init__(self):
        self.app = FastAPI()
        self.letta_client = create_client()
        stripe.api_key = "your_stripe_key"
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/api/agents/create")
        async def create_agent(
            user: User = Depends(get_current_user)
        ):
            """Create agent for user (tenant)."""
            
            # Check subscription limits
            if not self.check_agent_limit(user.id):
                return {"error": "Agent limit reached for your plan"}
            
            # Create agent with user isolation
            agent = self.letta_client.create_agent(
                name=f"{user.id}_agent_{uuid.uuid4()}",
                persona="Custom persona"
            )
            
            # Store agent-user mapping in database
            db.agents.insert({
                "agent_id": agent.id,
                "user_id": user.id,
                "created_at": datetime.now()
            })
            
            return {"agent_id": agent.id}
        
        @self.app.post("/api/agents/{agent_id}/message")
        async def send_message(
            agent_id: str,
            message: str,
            user: User = Depends(get_current_user)
        ):
            """Send message to agent with tenant isolation."""
            
            # Verify user owns this agent
            agent_record = db.agents.find_one({
                "agent_id": agent_id,
                "user_id": user.id
            })
            
            if not agent_record:
                return {"error": "Agent not found or access denied"}
            
            # Track usage for billing
            self.track_usage(user.id, "message")
            
            # Send message
            response = self.letta_client.send_message(
                agent_id=agent_id,
                message=message,
                role="user"
            )
            
            return {"response": response}
    
    def check_agent_limit(self, user_id: str) -> bool:
        """Check if user can create more agents."""
        user = db.users.find_one({"id": user_id})
        plan_limits = {
            "free": 1,
            "pro": 10,
            "enterprise": -1  # unlimited
        }
        
        limit = plan_limits.get(user["plan"], 0)
        if limit == -1:
            return True
        
        current_count = db.agents.count({"user_id": user_id})
        return current_count < limit
```

## Billing Integration

```python
class BillingManager:
    """Handle subscription billing."""
    
    def __init__(self):
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    
    def create_subscription(self, user_id: str, plan: str) -> Dict:
        """Create Stripe subscription."""
        
        user = db.users.find_one({"id": user_id})
        
        # Create Stripe customer if doesn't exist
        if not user.get("stripe_customer_id"):
            customer = stripe.Customer.create(
                email=user["email"],
                metadata={"user_id": user_id}
            )
            
            db.users.update(
                {"id": user_id},
                {"$set": {"stripe_customer_id": customer.id}}
            )
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=user["stripe_customer_id"],
            items=[{"price": self.get_price_id(plan)}],
            metadata={"user_id": user_id}
        )
        
        # Update user plan
        db.users.update(
            {"id": user_id},
            {"$set": {"plan": plan, "subscription_id": subscription.id}}
        )
        
        return {"subscription_id": subscription.id}
    
    def track_usage(self, user_id: str, event_type: str):
        """Track usage for usage-based billing."""
        
        # Record usage event
        db.usage.insert({
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": datetime.now()
        })
        
        # For usage-based billing with Stripe
        user = db.users.find_one({"id": user_id})
        if user.get("usage_subscription_item_id"):
            stripe.SubscriptionItem.create_usage_record(
                user["usage_subscription_item_id"],
                quantity=1,
                timestamp=int(time.time())
            )
```

## User Management

```python
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    """Authentication and authorization."""
    
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    ALGORITHM = "HS256"
    
    def create_user(self, email: str, password: str) -> Dict:
        """Register new user."""
        
        # Check if user exists
        if db.users.find_one({"email": email}):
            raise ValueError("Email already registered")
        
        # Hash password
        hashed_password = pwd_context.hash(password)
        
        # Create user
        user_id = str(uuid.uuid4())
        db.users.insert({
            "id": user_id,
            "email": email,
            "password": hashed_password,
            "plan": "free",
            "created_at": datetime.now()
        })
        
        return {"user_id": user_id}
    
    def login(self, email: str, password: str) -> str:
        """Authenticate user and return JWT token."""
        
        user = db.users.find_one({"email": email})
        
        if not user or not pwd_context.verify(password, user["password"]):
            raise ValueError("Invalid credentials")
        
        # Create JWT token
        access_token = self.create_access_token({"sub": user["id"]})
        
        return access_token
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt
```

## Analytics Dashboard

```python
class Analytics:
    """Usage analytics for SaaS platform."""
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user usage statistics."""
        
        # Agent count
        agent_count = db.agents.count({"user_id": user_id})
        
        # Message count (this month)
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        message_count = db.usage.count({
            "user_id": user_id,
            "event_type": "message",
            "timestamp": {"$gte": month_start}
        })
        
        # Cost estimation
        cost = self.calculate_cost(user_id)
        
        return {
            "agents": agent_count,
            "messages_this_month": message_count,
            "estimated_cost": cost
        }
    
    def get_platform_stats(self) -> Dict:
        """Get platform-wide statistics."""
        
        return {
            "total_users": db.users.count(),
            "total_agents": db.agents.count(),
            "active_users_today": self._count_active_users(days=1),
            "revenue_this_month": self._calculate_revenue()
        }
```

## Deployment Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/saas
      - REDIS_URL=redis://redis:6379
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: always
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=saas
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Next Steps

Learn **Advanced Agent Architectures** - cognitive architectures and reasoning systems.

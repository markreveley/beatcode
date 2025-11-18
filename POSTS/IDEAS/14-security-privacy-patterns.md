---
title: "Security and Privacy in Letta: Building Secure AI Agents"
date: "2025-02-21"
description: "Implement security best practices, data privacy, access control, and compliance for production Letta agents."
tags: ["Letta", "Security", "Privacy", "Compliance", "Production"]
---

# Security and Privacy in Letta

Production AI agents handle sensitive data. This tutorial covers security hardening, privacy protection, and compliance for Letta systems.

## Data Privacy Fundamentals

```python
from letta import create_client
import hashlib
from cryptography.fernet import Fernet

class PrivateAgent:
    """Agent with data privacy protections."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def send_message(self, message: str, contains_pii: bool = False) -> Dict:
        """Send message with PII protection."""
        
        if contains_pii:
            # Detect and redact PII
            message = self.redact_pii(message)
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=message,
            role="user"
        )
        
        return response
    
    def redact_pii(self, text: str) -> str:
        """Redact personally identifiable information."""
        import re
        
        # Email
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card (simplified)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text
    
    def encrypt_sensitive_memory(self, data: str) -> str:
        """Encrypt sensitive data before storage."""
        encrypted = self.cipher.encrypt(data.encode())
        return encrypted.decode()
    
    def decrypt_memory(self, encrypted_data: str) -> str:
        """Decrypt retrieved data."""
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return decrypted.decode()
```

## Access Control

```python
from enum import Enum
from typing import Set

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class RBACAgent:
    """Agent with role-based access control."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.roles = {
            "viewer": {Permission.READ},
            "editor": {Permission.READ, Permission.WRITE},
            "admin": {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        }
        self.user_roles = {}
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        if role in self.roles:
            self.user_roles[user_id] = role
    
    def check_permission(self, user_id: str, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        role = self.user_roles.get(user_id)
        if not role:
            return False
        
        permissions = self.roles.get(role, set())
        return required_permission in permissions
    
    def send_message(self, user_id: str, message: str) -> Dict:
        """Send message with permission check."""
        if not self.check_permission(user_id, Permission.READ):
            raise PermissionError("User lacks READ permission")
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=message,
            role="user"
        )
        
        return response
```

## Audit Logging

```python
import logging
from datetime import datetime

class AuditedAgent:
    """Agent with comprehensive audit logging."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        
        # Setup audit logger
        self.audit_logger = logging.getLogger('audit')
        handler = logging.FileHandler('audit.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def send_message(self, user_id: str, message: str) -> Dict:
        """Send message with audit trail."""
        
        # Log request
        self.audit_logger.info(f"USER:{user_id} REQUEST:{message[:100]}")
        
        try:
            response = self.client.send_message(
                agent_id=self.agent_id,
                message=message,
                role="user"
            )
            
            # Log success
            self.audit_logger.info(f"USER:{user_id} SUCCESS")
            
            return response
            
        except Exception as e:
            # Log failure
            self.audit_logger.error(f"USER:{user_id} ERROR:{str(e)}")
            raise
```

## GDPR Compliance

```python
class GDPRCompliantAgent:
    """Agent with GDPR compliance features."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.user_consents = {}
    
    def request_consent(self, user_id: str, purpose: str) -> bool:
        """Request user consent for data processing."""
        # In production, this would be an actual UI interaction
        self.user_consents[user_id] = {
            "purpose": purpose,
            "granted": True,
            "timestamp": datetime.now()
        }
        return True
    
    def right_to_be_forgotten(self, user_id: str):
        """Delete all user data (GDPR Right to Erasure)."""
        # Delete from agent memory
        # Delete from logs
        # Delete from backups
        pass
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (GDPR Right to Data Portability)."""
        # Collect all data related to user
        user_data = {
            "consents": self.user_consents.get(user_id),
            "conversations": [],  # Retrieve from database
            "memory": {}  # Retrieve from agent memory
        }
        return user_data
```

## Rate Limiting and DDoS Protection

```python
from collections import defaultdict
import time

class RateLimitedAgent:
    """Agent with rate limiting."""
    
    def __init__(self, agent_id: str, max_requests_per_minute: int = 60):
        self.agent_id = agent_id
        self.client = create_client()
        self.max_rpm = max_requests_per_minute
        self.request_log = defaultdict(list)
    
    def send_message(self, user_id: str, message: str) -> Dict:
        """Send message with rate limiting."""
        
        # Check rate limit
        if not self._check_rate_limit(user_id):
            raise Exception("Rate limit exceeded")
        
        # Log request
        self.request_log[user_id].append(time.time())
        
        # Process request
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=message,
            role="user"
        )
        
        return response
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.request_log[user_id] = [
            ts for ts in self.request_log[user_id]
            if ts > minute_ago
        ]
        
        # Check limit
        return len(self.request_log[user_id]) < self.max_rpm
```

## Next Steps

Explore **Performance Tuning and Optimization** for high-scale deployments.

---
title: "Custom LLM Providers in Letta: Beyond OpenAI"
date: "2025-02-18"
description: "Integrate custom LLM providers, local models, and alternative APIs with Letta agents for flexibility and cost optimization."
tags: ["Letta", "LLM", "Custom Models", "Integration", "Advanced"]
---

# Custom LLM Providers in Letta

While Letta works great with OpenAI, you can integrate any LLM provider. This tutorial covers using Anthropic, local models, Azure, and custom endpoints.

## Multi-Provider Setup

```python
from letta import create_client
from letta.schemas.llm_config import LLMConfig

# OpenAI
openai_config = LLMConfig(
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="https://api.openai.com/v1"
)

# Anthropic Claude
claude_config = LLMConfig(
    model="claude-3-opus-20240229",
    model_endpoint_type="anthropic",
    model_endpoint="https://api.anthropic.com/v1"
)

# Azure OpenAI
azure_config = LLMConfig(
    model="gpt-4",
    model_endpoint_type="azure",
    model_endpoint="https://your-resource.openai.azure.com/",
    model_wrapper="azure"
)

# Local Ollama
local_config = LLMConfig(
    model="llama2",
    model_endpoint_type="ollama",
    model_endpoint="http://localhost:11434"
)

# Create agents with different providers
client = create_client()

claude_agent = client.create_agent(
    name="ClaudeAgent",
    llm_config=claude_config
)

local_agent = client.create_agent(
    name="LocalAgent",
    llm_config=local_config
)
```

## Provider Comparison and Selection

```python
class ProviderManager:
    """Intelligently select LLM provider based on task."""
    
    def __init__(self):
        self.providers = {
            "openai_gpt4": {"cost_per_1k": 0.03, "speed": "medium", "quality": "high"},
            "openai_gpt3.5": {"cost_per_1k": 0.002, "speed": "fast", "quality": "medium"},
            "claude_opus": {"cost_per_1k": 0.015, "speed": "slow", "quality": "highest"},
            "local_llama": {"cost_per_1k": 0.0, "speed": "medium", "quality": "medium"}
        }
    
    def select_provider(self, task_type: str, budget: str) -> str:
        """Select best provider for task and budget."""
        if budget == "high" and task_type == "complex":
            return "claude_opus"
        elif budget == "low":
            return "local_llama"
        else:
            return "openai_gpt3.5"
```

## Fallback Provider Chain

```python
class ProviderFallback:
    """Try multiple providers with fallback."""
    
    def __init__(self, provider_configs: List[LLMConfig]):
        self.configs = provider_configs
        self.client = create_client()
    
    def send_with_fallback(self, message: str) -> Dict:
        """Try providers in order until success."""
        for i, config in enumerate(self.configs):
            try:
                agent = self.client.create_agent(
                    name=f"FallbackAgent_{i}",
                    llm_config=config
                )
                
                response = self.client.send_message(
                    agent_id=agent.id,
                    message=message,
                    role="user"
                )
                
                return {"status": "success", "provider": i, "response": response}
                
            except Exception as e:
                if i == len(self.configs) - 1:
                    raise Exception("All providers failed")
                continue
```

## Cost Optimization

```python
class CostOptimizedAgent:
    """Route requests to cost-effective providers."""
    
    def __init__(self):
        self.cheap_config = LLMConfig(model="gpt-3.5-turbo")
        self.expensive_config = LLMConfig(model="gpt-4")
        self.client = create_client()
    
    def send_message(self, message: str) -> Dict:
        """Use cheap model first, upgrade if needed."""
        # Try cheap model
        try:
            response = self._send_with_config(message, self.cheap_config)
            
            # Check if response is adequate
            if self._is_adequate(response):
                return {"provider": "cheap", "response": response}
            else:
                # Upgrade to expensive model
                response = self._send_with_config(message, self.expensive_config)
                return {"provider": "expensive", "response": response}
                
        except Exception as e:
            # Fallback to expensive model
            response = self._send_with_config(message, self.expensive_config)
            return {"provider": "fallback", "response": response}
```

## Next Steps

Learn **Security and Privacy** patterns for production Letta deployments.

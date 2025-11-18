---
title: "Understanding Letta Memory Systems: How AI Agents Remember"
date: "2025-01-25"
description: "A deep dive into Letta's three-tier memory architecture - core memory, archival memory, and recall memory - with practical examples."
tags: ["Letta", "Memory Systems", "Tutorial", "AI Architecture"]
---

# Understanding Letta Memory Systems: How AI Agents Remember

One of Letta's most powerful features is its sophisticated memory management system. Unlike traditional chatbots that lose context or hit token limits, Letta agents intelligently manage three distinct types of memory. In this tutorial, we'll explore how these memory systems work and how to use them effectively.

## The Three-Tier Memory Architecture

Letta agents maintain three types of memory, each serving a different purpose:

```
┌─────────────────────────────────────────┐
│         CORE MEMORY (Context)           │
│  Always in LLM context, limited size    │
│  • human: User information              │
│  • persona: Agent identity              │
└─────────────────────────────────────────┘
           ▼              ▲
┌─────────────────────────────────────────┐
│     RECALL MEMORY (Conversation)        │
│  Recent messages, auto-summarized       │
│  • Last N messages                      │
│  • Searchable history                   │
└─────────────────────────────────────────┘
           ▼              ▲
┌─────────────────────────────────────────┐
│    ARCHIVAL MEMORY (Knowledge Base)     │
│  Long-term storage, unlimited size      │
│  • Facts, documents, data               │
│  • Vector search enabled                │
└─────────────────────────────────────────┘
```

Let's explore each type in detail.

## 1. Core Memory: The Agent's Working Context

Core memory is what the agent "sees" in every interaction. It's limited in size but always present in the LLM's context window.

### Structure

Core memory has two sections:

**human**: Information about the user
**persona**: The agent's identity and instructions

### Example: Viewing Core Memory

```python
from letta import create_client

client = create_client()
agent = client.get_agent(agent_id="your-agent-id")

# View core memory
print("Human:", agent.memory.human)
print("Persona:", agent.memory.persona)
```

### Example: Programmatically Updating Core Memory

```python
# Update the human section
client.update_agent_core_memory(
    agent_id=agent.id,
    human="Name: Alice\nOccupation: Data Scientist\nCurrent project: Building a chatbot\nPreferred language: Python"
)

# Update the persona section
client.update_agent_core_memory(
    agent_id=agent.id,
    persona="You are a helpful coding assistant specializing in Python and machine learning."
)
```

### Best Practices for Core Memory

- **Keep it concise**: Core memory counts against your token limit
- **Update regularly**: As you learn important facts, update core memory
- **Be specific**: "Current project: Chatbot with RAG" is better than "Working on a project"
- **Structure it**: Use clear labels and formatting

### Practical Example: A Learning Agent

Let's create an agent that actively manages its core memory:

```python
from letta import create_client

client = create_client()

# Create an agent with detailed persona instructions
persona = """
You are a learning assistant that actively updates your memory.

When you learn something important about the user:
1. Identify if it's a key fact (name, occupation, preferences)
2. Update your core memory immediately
3. Confirm the update to the user

Core memory format:
- Name: [user's name]
- Role: [their job/role]
- Interests: [comma-separated list]
- Current goals: [what they're working on]
"""

agent = client.create_agent(
    name="LearningAssistant",
    persona=persona,
    human="Name: Unknown\nRole: Unknown\nInterests: Unknown\nCurrent goals: Unknown"
)

# Conversation
response = client.send_message(
    agent_id=agent.id,
    message="Hi! I'm Jordan, a software engineer interested in AI and rock climbing. I'm learning about Letta to build better agents.",
    role="user"
)
```

The agent will automatically update its core memory with this information!

## 2. Recall Memory: Conversation History

Recall memory stores the conversation history. Unlike core memory, it's not always fully in context - older messages get summarized or moved out.

### How Recall Memory Works

1. Recent messages are kept in full
2. Older messages are summarized to save tokens
3. The agent can search recall memory when needed
4. Messages are stored in a database for retrieval

### Example: Searching Recall Memory

```python
# Search conversation history
messages = client.get_messages(
    agent_id=agent.id,
    limit=50  # Get last 50 messages
)

for msg in messages:
    print(f"{msg.role}: {msg.text}")
```

### Example: Accessing Specific Messages

```python
# Get messages from a specific time period
from datetime import datetime, timedelta

one_week_ago = datetime.now() - timedelta(days=7)

messages = client.get_messages(
    agent_id=agent.id,
    start_date=one_week_ago
)
```

### When Agents Use Recall Memory

Agents automatically search recall memory when:
- They need to recall a past conversation
- The user asks "What did I say about X?"
- Context from core memory is insufficient

You can also explicitly prompt recall:

```
You: What did we discuss last Tuesday about the project timeline?
Agent: [Searches recall memory] We discussed that the project deadline is March 15th...
```

## 3. Archival Memory: Long-Term Knowledge Storage

Archival memory is for storing large amounts of information that don't fit in core memory. It's essentially a vector database integrated into the agent.

### Use Cases for Archival Memory

- Storing documentation
- Maintaining a knowledge base
- Keeping project notes
- Storing reference materials
- Building a personal wiki

### Example: Adding to Archival Memory

```python
# Insert information into archival memory
client.insert_archival_memory(
    agent_id=agent.id,
    memory="Python best practices: Always use virtual environments for projects. Popular tools include venv, virtualenv, and conda."
)

client.insert_archival_memory(
    agent_id=agent.id,
    memory="Letta architecture: Uses a three-tier memory system with core, recall, and archival memory. Core memory is always in context."
)
```

### Example: Searching Archival Memory

```python
# Search archival memory
results = client.get_archival_memory(
    agent_id=agent.id,
    query="Python virtual environments",
    limit=5
)

for result in results:
    print(f"Memory: {result.text}")
    print(f"Score: {result.score}\n")
```

### Practical Example: Building a Documentation Assistant

```python
from letta import create_client

client = create_client()

# Create an agent that uses archival memory
agent = client.create_agent(
    name="DocsAssistant",
    persona="You are a documentation assistant. When users ask questions, search your archival memory for relevant information."
)

# Populate archival memory with documentation
docs = [
    "Letta installation: pip install letta",
    "Creating an agent: Use client.create_agent() with name, persona, and human parameters",
    "Core memory has two sections: human (user info) and persona (agent identity)",
    "Archival memory is searched using semantic similarity",
    "To run CLI: Use 'letta run' command"
]

for doc in docs:
    client.insert_archival_memory(agent_id=agent.id, memory=doc)

# Now ask questions
response = client.send_message(
    agent_id=agent.id,
    message="How do I create a new agent?",
    role="user"
)

# The agent will search archival memory and find the answer!
```

## Memory Management Strategies

### Strategy 1: Progressive Information Storage

Start with core memory, overflow to archival:

```python
# Important, frequently needed → Core Memory
core_info = "User prefers concise explanations. Current project: Music AI"

# Detailed, occasionally needed → Archival Memory
archival_info = "Project details: Building a music recommendation system using collaborative filtering. Dataset: 1M songs. Tech stack: Python, PyTorch, FastAPI"
```

### Strategy 2: Context-Aware Memory Updates

Update memory based on conversation context:

```python
def should_update_core_memory(message):
    """Determine if message contains core-memory-worthy information."""
    keywords = ["my name is", "i work as", "i'm interested in", "my goal is"]
    return any(keyword in message.lower() for keyword in keywords)

user_message = "My name is Sam and I work as a product manager"
if should_update_core_memory(user_message):
    # Update core memory
    client.update_agent_core_memory(
        agent_id=agent.id,
        human="Name: Sam\nOccupation: Product Manager"
    )
```

### Strategy 3: Archival Memory as a Knowledge Graph

Structure archival entries for better retrieval:

```python
# Good: Structured and specific
client.insert_archival_memory(
    agent_id=agent.id,
    memory="Project: MusicAI | Component: Recommendation Engine | Tech: Collaborative Filtering | Status: In Development"
)

# Less effective: Vague and unstructured
client.insert_archival_memory(
    agent_id=agent.id,
    memory="Working on some music stuff with AI"
)
```

## Hands-On Exercise: Build a Personal Knowledge Base

Let's build a practical example that uses all three memory types:

```python
from letta import create_client
import json

client = create_client()

# Create a personal knowledge agent
persona = """
You are a Personal Knowledge Base assistant. You help users:
1. Store and organize information
2. Recall past conversations and notes
3. Find relevant information quickly

Memory usage:
- Core Memory: User's name, current focus, preferences
- Recall Memory: Recent conversations
- Archival Memory: All stored knowledge and notes

When users share information, categorize it and store it appropriately.
"""

agent = client.create_agent(
    name="KnowledgeBase",
    persona=persona,
    human="Name: [Unknown]\nCurrent focus: [Unknown]\nPreferences: [Unknown]"
)

def add_knowledge(agent_id, topic, content):
    """Add knowledge to archival memory."""
    memory_entry = f"Topic: {topic} | Content: {content}"
    client.insert_archival_memory(agent_id=agent_id, memory=memory_entry)
    print(f"✓ Added to knowledge base: {topic}")

def search_knowledge(agent_id, query):
    """Search the knowledge base."""
    results = client.get_archival_memory(
        agent_id=agent_id,
        query=query,
        limit=3
    )
    return results

# Example usage
add_knowledge(agent.id, "Python", "Use list comprehensions for concise iteration")
add_knowledge(agent.id, "Letta", "Agents maintain persistent memory across sessions")
add_knowledge(agent.id, "Productivity", "Pomodoro technique: 25 min work, 5 min break")

# Search
results = search_knowledge(agent.id, "Python tips")
for r in results:
    print(f"Found: {r.text}")
```

## Memory Limits and Best Practices

### Core Memory Limits
- Typically 2000-4000 tokens depending on model
- Monitor size to avoid context overflow
- Regularly review and prune unnecessary information

### Archival Memory Limits
- Virtually unlimited (database-backed)
- Search performance may degrade with millions of entries
- Use descriptive text for better vector search results

### Recall Memory Limits
- Automatically managed by Letta
- Older messages are summarized
- Complete history is always available via API

## Common Patterns and Anti-Patterns

### ✅ Good Practices

```python
# Clear, structured core memory
human = """
Name: Alex Rivera
Role: ML Engineer
Current Project: Building Letta agents for task automation
Preferences: Concise explanations, Python code examples
"""

# Descriptive archival entries
archival = "Git workflow: Always create feature branches from main. Use conventional commits. Squash merge to main."
```

### ❌ Anti-Patterns

```python
# Vague core memory
human = "Some person working on stuff"

# Uninformative archival entries
archival = "That thing we talked about"
```

## Debugging Memory Issues

### Check what's in memory:

```python
# Inspect core memory
agent = client.get_agent(agent_id)
print("Core Memory:", agent.memory.human, agent.memory.persona)

# Check archival memory count
archival_memories = client.get_archival_memory(agent_id=agent.id, limit=1000)
print(f"Archival entries: {len(archival_memories)}")

# Review recent messages
messages = client.get_messages(agent_id=agent.id, limit=20)
print(f"Recent messages: {len(messages)}")
```

## Next Steps

Now that you understand Letta's memory systems, you can:
- Build agents with sophisticated knowledge management
- Create personal wikis and knowledge bases
- Design agents that learn and adapt over time

In the next tutorial, we'll explore **adding custom tools to Letta agents** - teaching your agents to perform actions in the real world!

## Resources

- [Letta Memory Documentation](https://docs.letta.ai/memory)
- [Vector Search in Letta](https://docs.letta.ai/archival-memory)
- [Memory Management Examples](https://github.com/letta-ai/letta/tree/main/examples/memory)

Have questions about memory management? Drop them in the comments!

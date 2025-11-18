---
title: "Building Your First Custom Letta Agent: A Step-by-Step Tutorial"
date: "2025-01-22"
description: "Learn how to create a custom Letta agent from scratch using the Python SDK, complete with personality, memory, and custom behaviors."
tags: ["Letta", "Tutorial", "Python", "AI Agents", "Custom Agents"]
---

# Building Your First Custom Letta Agent: A Step-by-Step Tutorial

In our previous post, we explored how to get started with Letta using the CLI. Now, let's level up and build a **custom agent from scratch** using the Python SDK. By the end of this tutorial, you'll have created a specialized agent with its own personality and capabilities.

## What We'll Build

We're going to create a **Personal Knowledge Assistant** - an agent that:
- Helps you organize and recall information
- Learns about your projects and interests
- Provides contextual suggestions based on what it knows about you
- Maintains a persistent memory of your interactions

## Prerequisites

Make sure you've completed the [Getting Started with Letta](./01-getting-started-with-letta.md) tutorial and have:
- Letta installed (`pip install letta`)
- An OpenAI API key configured
- Python 3.10 or higher

## Step 1: Create Your Project

First, let's set up a new Python project:

```bash
mkdir letta-knowledge-assistant
cd letta-knowledge-assistant
```

Create a new Python file:

```bash
touch knowledge_agent.py
```

## Step 2: Import and Initialize Letta

Open `knowledge_agent.py` in your favorite editor and add:

```python
from letta import create_client
from letta.schemas.llm_config import LLMConfig
from letta.schemas.embedding_config import EmbeddingConfig

# Create a Letta client
client = create_client()

print("Letta client initialized!")
```

This creates a client that connects to your Letta server (running locally by default).

## Step 3: Define Your Agent's Persona

The persona defines your agent's personality and behavior. Let's create a helpful knowledge assistant:

```python
# Define the agent's persona
persona = """
You are a Personal Knowledge Assistant designed to help users organize,
recall, and connect information. You are:

- Proactive in organizing information into categories
- Excellent at making connections between different pieces of information
- Patient and thorough when helping users recall details
- Curious and ask clarifying questions to better understand context

When a user shares information, you:
1. Store it in memory with appropriate context
2. Ask if there are related topics or projects to link it to
3. Suggest how this information might be useful later

You use a friendly, professional tone and always confirm when you've
stored something important.
"""
```

## Step 4: Define the Human Profile

This stores what the agent knows about the user:

```python
# Define initial human profile
human = """
First name: [User will provide]
Interests: [To be learned over time]
Current projects: [To be discovered]
Preferred communication style: [To be observed]
"""
```

The agent will update this as it learns about the user.

## Step 5: Create the Agent

Now let's create our custom agent:

```python
# Create the agent
agent_state = client.create_agent(
    name="KnowledgeAssistant",
    persona=persona,
    human=human,
    llm_config=LLMConfig.default_config("gpt-4"),
    embedding_config=EmbeddingConfig.default_config(provider="openai")
)

print(f"Agent created with ID: {agent_state.id}")
print(f"Agent name: {agent_state.name}")
```

## Step 6: Interact with Your Agent

Let's add a conversation loop:

```python
def chat_with_agent(agent_id):
    """Interactive chat loop with the agent."""
    print("\n=== Knowledge Assistant ===")
    print("Type 'exit' to quit, 'memory' to view core memory\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if user_input.lower() == 'memory':
            # View agent's core memory
            agent = client.get_agent(agent_id)
            print(f"\n--- Core Memory ---")
            print(f"Human: {agent.memory.human}")
            print(f"Persona: {agent.memory.persona}")
            print("-------------------\n")
            continue

        if not user_input:
            continue

        # Send message to agent
        response = client.send_message(
            agent_id=agent_id,
            message=user_input,
            role="user"
        )

        # Print agent's response
        for message in response.messages:
            if message.message_type == "assistant_message":
                print(f"\nAssistant: {message.text}\n")

# Start chatting
chat_with_agent(agent_state.id)
```

## Step 7: Complete Code

Here's the complete `knowledge_agent.py`:

```python
from letta import create_client
from letta.schemas.llm_config import LLMConfig
from letta.schemas.embedding_config import EmbeddingConfig

def create_knowledge_assistant():
    """Create a custom knowledge assistant agent."""

    # Initialize client
    client = create_client()

    # Define persona
    persona = """
You are a Personal Knowledge Assistant designed to help users organize,
recall, and connect information. You are:

- Proactive in organizing information into categories
- Excellent at making connections between different pieces of information
- Patient and thorough when helping users recall details
- Curious and ask clarifying questions to better understand context

When a user shares information, you:
1. Store it in memory with appropriate context
2. Ask if there are related topics or projects to link it to
3. Suggest how this information might be useful later

You use a friendly, professional tone and always confirm when you've
stored something important.
    """

    # Define initial human profile
    human = """
First name: [User will provide]
Interests: [To be learned over time]
Current projects: [To be discovered]
Preferred communication style: [To be observed]
    """

    # Create agent
    agent_state = client.create_agent(
        name="KnowledgeAssistant",
        persona=persona,
        human=human,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config(provider="openai")
    )

    print(f"✓ Agent created: {agent_state.name}")
    print(f"✓ Agent ID: {agent_state.id}")

    return client, agent_state.id

def chat_with_agent(client, agent_id):
    """Interactive chat loop with the agent."""
    print("\n=== Knowledge Assistant ===")
    print("Commands: 'exit' to quit | 'memory' to view core memory\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if user_input.lower() == 'memory':
            agent = client.get_agent(agent_id)
            print(f"\n--- Core Memory ---")
            print(f"Human:\n{agent.memory.human}\n")
            print(f"Persona:\n{agent.memory.persona}")
            print("-------------------\n")
            continue

        if not user_input:
            continue

        # Send message to agent
        response = client.send_message(
            agent_id=agent_id,
            message=user_input,
            role="user"
        )

        # Print agent's response
        for message in response.messages:
            if message.message_type == "assistant_message":
                print(f"\nAssistant: {message.text}\n")

if __name__ == "__main__":
    client, agent_id = create_knowledge_assistant()
    chat_with_agent(client, agent_id)
```

## Step 8: Run Your Agent

Run your custom agent:

```bash
python knowledge_agent.py
```

Try these interactions:

```
You: Hi! My name is Alex and I'm working on a machine learning project.
Assistant: Hello Alex! Nice to meet you. I've noted that you're working on
a machine learning project. Could you tell me more about it? What specific
area of ML are you focusing on?

You: I'm building a recommendation system for music.
Assistant: Excellent! I've stored that information. A music recommendation
system is a fascinating project. Are you using collaborative filtering,
content-based filtering, or perhaps a hybrid approach?
```

Now type `memory` to see how the agent has updated its knowledge about you!

## Step 9: Understanding What Happened

Let's break down what makes this work:

1. **Client Creation**: `create_client()` connects to the Letta server
2. **Agent Creation**: `create_agent()` instantiates a new agent with:
   - Custom personality (persona)
   - User profile template (human)
   - LLM configuration
   - Embedding configuration for memory
3. **Message Sending**: `send_message()` sends user input and retrieves responses
4. **Memory Management**: The agent automatically updates its memory based on conversations

## Step 10: Experiment and Extend

Now that you have a working agent, try:

- **Modify the persona** to create different agent types (tutor, creative assistant, etc.)
- **Add more initial context** to the human profile
- **Track multiple conversations** by saving the agent_id
- **Load existing agents** using `client.get_agent(agent_id)`

## Common Gotchas

### Issue: "Agent not found" after restart
**Solution**: Agent IDs persist in the database. Save your agent_id to a file to reuse it:

```python
# Save agent ID
with open('agent_id.txt', 'w') as f:
    f.write(agent_state.id)

# Load existing agent
with open('agent_id.txt', 'r') as f:
    agent_id = f.read().strip()
    agent = client.get_agent(agent_id)
```

### Issue: Agent responses are generic
**Solution**: Make your persona more specific with concrete examples of how the agent should behave.

### Issue: Memory not updating
**Solution**: The agent needs explicit cues. Try adding instructions in the persona like "When you learn new information, always update your core memory."

## Next Steps

You've now built a custom Letta agent! In the next tutorial, we'll explore:
- **Advanced Memory Management**: How agents decide what to remember
- **Memory editing**: Programmatically updating agent memory
- **Archival memory**: Storing and retrieving large knowledge bases

## Resources

- [Letta Python SDK Documentation](https://docs.letta.ai/python-sdk)
- [Agent Schema Reference](https://docs.letta.ai/api-reference/agent)
- [Example Agents Repository](https://github.com/letta-ai/letta/tree/main/examples)

Happy building! Share your custom agents in the comments below.

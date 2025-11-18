---
title: "Getting Started with Letta: Your First Steps into Agentic AI"
date: "2025-01-20"
description: "A complete beginner's guide to installing Letta and understanding the fundamentals of building stateful AI agents."
tags: ["Letta", "Tutorial", "Getting Started", "AI Agents"]
---

# Getting Started with Letta: Your First Steps into Agentic AI

If you're curious about building AI agents that can remember context, learn from interactions, and execute tasks autonomously, **Letta** is the framework you need to explore. In this tutorial, we'll walk through everything you need to get started with Letta from scratch.

## What is Letta?

Letta (formerly MemGPT) is an open-source framework for building stateful AI agents with advanced memory management. Unlike traditional chatbots that forget context between sessions, Letta agents:

- **Maintain long-term memory** across conversations
- **Learn and adapt** from user interactions
- **Execute tools and functions** to perform real-world tasks
- **Manage their own memory** intelligently, deciding what to remember and forget

Think of Letta agents as AI assistants with a persistent brain that evolves over time.

## Prerequisites

Before we begin, make sure you have:

- Python 3.10 or higher installed
- pip (Python package manager)
- An OpenAI API key (or access to another LLM provider)
- Basic understanding of Python

## Step 1: Installation

Let's install Letta using pip. Open your terminal and run:

```bash
pip install letta
```

This installs the Letta framework along with its dependencies. To verify the installation:

```bash
letta version
```

You should see the installed Letta version printed to your terminal.

## Step 2: Configure Your LLM Provider

Letta needs access to a language model. Let's configure it to use OpenAI (you can also use other providers like Anthropic, Azure, or local models).

First, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

For Windows users:
```bash
set OPENAI_API_KEY=your-api-key-here
```

To make this permanent, add it to your `.bashrc`, `.zshrc`, or environment variables.

## Step 3: Initialize Letta

Run the Letta configuration wizard:

```bash
letta configure
```

This interactive setup will ask you to:
1. Choose your LLM provider (select OpenAI)
2. Select your model (gpt-4 or gpt-3.5-turbo)
3. Configure your embedding model for memory
4. Set up your default agent parameters

Just follow the prompts, and Letta will create a configuration file for you.

## Step 4: Run Your First Agent

Now for the exciting part - let's run an agent! Start the Letta CLI:

```bash
letta run
```

This launches an interactive session with a default agent. You can now chat with your agent:

```
You: Hello! What can you help me with?
Agent: Hello! I'm an AI agent powered by Letta. I can help you with various tasks...
```

Try asking the agent to remember something:

```
You: Please remember that my favorite color is blue.
Agent: Got it! I've stored in my memory that your favorite color is blue.
```

Exit the session (type `/exit`), then start a new one:

```bash
letta run
```

Now ask:
```
You: What's my favorite color?
Agent: Your favorite color is blue!
```

**The agent remembered!** This is the power of Letta's persistent memory system.

## Step 5: Understanding Agent Memory

Letta agents have three types of memory:

1. **Core Memory**: Essential information the agent always keeps in context
   - `human`: Facts about the user
   - `persona`: The agent's identity and instructions

2. **Archival Memory**: Long-term storage for vast amounts of information
   - Searchable knowledge base
   - Can store thousands of facts

3. **Recall Memory**: Conversation history
   - Previous messages and interactions
   - Automatically managed by the agent

During your session, try these commands:

```
/memory - View the agent's current core memory
/archival - Search archival memory
/help - See all available commands
```

## Step 6: Next Steps

Congratulations! You've successfully:
- Installed Letta
- Configured your LLM provider
- Run your first stateful agent
- Experienced persistent memory in action

### Where to Go From Here

Now that you have Letta running, you can:
- Create custom agents with specific personalities
- Build agents with custom tools and functions
- Integrate Letta into your applications
- Explore different memory architectures

In our next tutorial, we'll dive into **building your first custom Letta agent from scratch** using the Python SDK.

## Common Issues and Solutions

### Issue: "OpenAI API key not found"
**Solution**: Make sure you've exported your API key correctly and it's available in your current terminal session.

### Issue: "Module not found" errors
**Solution**: Ensure you're using Python 3.10+ and try reinstalling: `pip install --upgrade letta`

### Issue: Agent responses are slow
**Solution**: Consider using gpt-3.5-turbo instead of gpt-4 for faster responses during development.

## Resources

- [Official Letta Documentation](https://letta.ai/docs)
- [Letta GitHub Repository](https://github.com/letta-ai/letta)
- [Letta Discord Community](https://discord.gg/letta)

Ready to build something amazing? Let's continue this journey in the next post!

---
title: "Building Multi-Agent Systems with Letta: Orchestrating AI Collaboration"
date: "2025-02-03"
description: "Learn how to build systems where multiple Letta agents collaborate, delegate, and solve complex problems together."
tags: ["Letta", "Multi-Agent", "Orchestration", "Advanced", "Collaboration"]
---

# Building Multi-Agent Systems with Letta: Orchestrating AI Collaboration

Single agents are powerful, but what if you could coordinate multiple specialized agents working together? In this tutorial, we'll build **multi-agent systems** where agents collaborate, delegate tasks, and solve complex problems that no single agent could handle alone.

## Why Multi-Agent Systems?

**Single Agent Limitations:**
- One persona can't excel at everything
- Context window limitations
- Sequential processing only

**Multi-Agent Benefits:**
- **Specialization**: Each agent focuses on what it does best
- **Parallel processing**: Multiple agents work simultaneously
- **Modularity**: Easy to add, remove, or update agents
- **Scalability**: Distribute workload across agents

## Architecture Patterns

### 1. Hub-and-Spoke (Orchestrator Pattern)

```
           ┌──────────────┐
           │ Orchestrator │
           │    Agent     │
           └──────┬───────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌─────────┐ ┌────────┐
│ Research │ │ Writing │ │ Review │
│  Agent   │ │  Agent  │ │ Agent  │
└──────────┘ └─────────┘ └────────┘
```

### 2. Pipeline Pattern

```
Input → Agent 1 → Agent 2 → Agent 3 → Output
       (Analyze)  (Process) (Validate)
```

### 3. Peer-to-Peer Pattern

```
    Agent A ←→ Agent B
        ↕          ↕
    Agent C ←→ Agent D
```

## Building a Hub-and-Spoke System

Let's create a content creation system with specialized agents:

```python
from letta import create_client
from typing import Dict, List
import json

client = create_client()

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}
        self.orchestrator = None

    def create_specialized_agent(self, name: str, persona: str, tools: List[str] = None):
        """Create a specialized agent."""
        agent = client.create_agent(
            name=name,
            persona=persona,
            tools=tools or []
        )
        self.agents[name] = agent
        return agent

    def create_orchestrator(self):
        """Create the orchestrator agent that manages other agents."""
        persona = f"""
You are an orchestrator managing a team of specialized agents:
{', '.join(self.agents.keys())}

When given a task:
1. Analyze which agents are needed
2. Delegate subtasks to appropriate agents
3. Coordinate their work
4. Synthesize final results

Available agents and their specialties:
{self._get_agent_descriptions()}

Always explain your delegation strategy before executing.
        """

        self.orchestrator = client.create_agent(
            name="Orchestrator",
            persona=persona
        )
        return self.orchestrator

    def _get_agent_descriptions(self) -> str:
        """Get descriptions of all available agents."""
        descriptions = []
        for name, agent in self.agents.items():
            descriptions.append(f"- {name}: {agent.memory.persona[:100]}...")
        return "\n".join(descriptions)

# Create the system
system = MultiAgentOrchestrator()

# Create specialized agents
system.create_specialized_agent(
    name="Researcher",
    persona="""You are a research specialist. You excel at:
    - Finding and analyzing information
    - Identifying credible sources
    - Summarizing research findings
    - Fact-checking claims
    Be thorough, cite sources, and present balanced perspectives.
    """
)

system.create_specialized_agent(
    name="Writer",
    persona="""You are a creative writer. You excel at:
    - Crafting engaging narratives
    - Maintaining consistent tone and style
    - Creating clear, compelling content
    - Structuring information effectively
    Transform research into readable, engaging content.
    """
)

system.create_specialized_agent(
    name="Editor",
    persona="""You are an editor. You excel at:
    - Reviewing content for clarity and accuracy
    - Checking grammar and style
    - Ensuring logical flow
    - Suggesting improvements
    Be constructive and specific in your feedback.
    """
)

# Create orchestrator
system.create_orchestrator()

print("✓ Multi-agent system initialized")
print(f"  Agents: {', '.join(system.agents.keys())}")
print(f"  Orchestrator: {system.orchestrator.name}")
```

## Implementing Agent Communication

```python
class AgentCommunicationHub:
    """Manages communication between agents."""

    def __init__(self, orchestrator_id: str, agent_ids: Dict[str, str]):
        self.orchestrator_id = orchestrator_id
        self.agent_ids = agent_ids
        self.conversation_history = []

    def delegate_task(self, agent_name: str, task: str) -> str:
        """Delegate a task to a specific agent."""
        if agent_name not in self.agent_ids:
            return f"Error: Agent '{agent_name}' not found"

        agent_id = self.agent_ids[agent_name]

        # Send task to agent
        response = client.send_message(
            agent_id=agent_id,
            message=f"Task from Orchestrator: {task}",
            role="user"
        )

        # Extract response
        result = self._extract_response(response)

        # Log interaction
        self.conversation_history.append({
            "from": "orchestrator",
            "to": agent_name,
            "task": task,
            "response": result
        })

        return result

    def orchestrate(self, user_request: str) -> str:
        """Main orchestration logic."""

        # Step 1: Orchestrator analyzes request
        analysis = client.send_message(
            agent_id=self.orchestrator_id,
            message=f"""User request: {user_request}

Please analyze this request and create a delegation plan.
Specify which agents should work on which parts.""",
            role="user"
        )

        orchestrator_plan = self._extract_response(analysis)
        print(f"\n📋 Orchestrator Plan:\n{orchestrator_plan}\n")

        # Step 2: Execute delegation (simplified - in reality, parse the plan)
        # For this example, we'll execute a predefined workflow

        results = {}

        # Research phase
        print("🔍 Delegating to Researcher...")
        results['research'] = self.delegate_task(
            "Researcher",
            f"Research this topic: {user_request}"
        )

        # Writing phase
        print("✍️  Delegating to Writer...")
        results['draft'] = self.delegate_task(
            "Writer",
            f"Write content based on this research:\n{results['research']}"
        )

        # Editing phase
        print("📝 Delegating to Editor...")
        results['final'] = self.delegate_task(
            "Editor",
            f"Review and improve this draft:\n{results['draft']}"
        )

        # Step 3: Orchestrator synthesizes results
        synthesis = client.send_message(
            agent_id=self.orchestrator_id,
            message=f"""The team has completed their work:

Research: {results['research'][:200]}...
Draft: {results['draft'][:200]}...
Final: {results['final'][:200]}...

Please provide a final summary and any additional insights.""",
            role="user"
        )

        final_result = self._extract_response(synthesis)

        return {
            "plan": orchestrator_plan,
            "research": results['research'],
            "draft": results['draft'],
            "edited": results['final'],
            "synthesis": final_result,
            "conversation_log": self.conversation_history
        }

    def _extract_response(self, response) -> str:
        """Extract text from agent response."""
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Usage
hub = AgentCommunicationHub(
    orchestrator_id=system.orchestrator.id,
    agent_ids={name: agent.id for name, agent in system.agents.items()}
)

# Execute multi-agent task
result = hub.orchestrate("Write a blog post about quantum computing for beginners")

print("\n✅ Final Result:")
print(result['synthesis'])
```

## Pipeline Pattern Implementation

Sequential processing where each agent builds on the previous:

```python
class AgentPipeline:
    """Chain agents in a sequential pipeline."""

    def __init__(self):
        self.stages = []

    def add_stage(self, agent_id: str, stage_name: str):
        """Add a stage to the pipeline."""
        self.stages.append({"agent_id": agent_id, "name": stage_name})

    def execute(self, initial_input: str) -> Dict:
        """Execute the pipeline."""
        results = {"input": initial_input}
        current_output = initial_input

        for i, stage in enumerate(self.stages):
            print(f"\n⚙️  Stage {i+1}/{len(self.stages)}: {stage['name']}")

            # Pass previous output as input to next stage
            response = client.send_message(
                agent_id=stage['agent_id'],
                message=f"Process this input:\n\n{current_output}",
                role="user"
            )

            current_output = self._extract_text(response)
            results[stage['name']] = current_output

        results['final_output'] = current_output
        return results

    def _extract_text(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Create pipeline agents
analyzer = client.create_agent(
    name="Analyzer",
    persona="You analyze input and extract key insights."
)

processor = client.create_agent(
    name="Processor",
    persona="You process and transform data based on analysis."
)

validator = client.create_agent(
    name="Validator",
    persona="You validate and quality-check processed data."
)

# Build pipeline
pipeline = AgentPipeline()
pipeline.add_stage(analyzer.id, "analysis")
pipeline.add_stage(processor.id, "processing")
pipeline.add_stage(validator.id, "validation")

# Execute
result = pipeline.execute("Analyze customer feedback: 'Great product but shipping was slow'")

print("\n📊 Pipeline Results:")
for stage, output in result.items():
    print(f"\n{stage.upper()}:")
    print(output[:200] + "..." if len(output) > 200 else output)
```

## Parallel Agent Execution

Run multiple agents simultaneously:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelAgentExecutor:
    """Execute multiple agents in parallel."""

    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_agent_task(self, agent_id: str, task: str) -> Dict:
        """Execute a single agent task."""
        try:
            response = client.send_message(
                agent_id=agent_id,
                message=task,
                role="user"
            )

            return {
                "agent_id": agent_id,
                "status": "success",
                "response": self._extract_text(response)
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "status": "error",
                "error": str(e)
            }

    def execute_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict]:
        """
        Execute multiple agent tasks in parallel.

        Args:
            tasks: List of {"agent_id": str, "task": str}

        Returns:
            List of results
        """
        futures = []

        for task in tasks:
            future = self.executor.submit(
                self.execute_agent_task,
                task['agent_id'],
                task['task']
            )
            futures.append(future)

        # Wait for all to complete
        results = [future.result() for future in futures]

        return results

    def _extract_text(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Create specialized agents for different aspects
agents = {
    "tech": client.create_agent(name="TechAnalyst", persona="Technical analysis expert"),
    "business": client.create_agent(name="BusinessAnalyst", persona="Business analysis expert"),
    "design": client.create_agent(name="DesignAnalyst", persona="Design and UX expert")
}

# Execute parallel analysis
executor = ParallelAgentExecutor()

tasks = [
    {"agent_id": agents["tech"].id, "task": "Analyze technical feasibility of building a mobile app"},
    {"agent_id": agents["business"].id, "task": "Analyze market opportunity for a mobile app"},
    {"agent_id": agents["design"].id, "task": "Suggest UX considerations for a mobile app"}
]

print("🚀 Executing parallel agent tasks...")
results = executor.execute_parallel(tasks)

for result in results:
    if result['status'] == 'success':
        print(f"\n✅ Agent {result['agent_id']}:")
        print(result['response'][:200] + "...")
    else:
        print(f"\n❌ Agent {result['agent_id']} failed:")
        print(result['error'])
```

## Agent Handoff Pattern

One agent delegates to another dynamically:

```python
class AgentHandoffSystem:
    """Implements dynamic agent handoff based on conversation flow."""

    def __init__(self, agents: Dict[str, str]):
        """
        Args:
            agents: Dict of {agent_name: agent_id}
        """
        self.agents = agents
        self.current_agent = None
        self.conversation_history = []

    def start_conversation(self, initial_agent: str, user_message: str):
        """Start conversation with specified agent."""
        self.current_agent = initial_agent
        return self.send_message(user_message)

    def send_message(self, message: str) -> Dict:
        """Send message to current agent and handle potential handoffs."""

        if not self.current_agent:
            return {"error": "No active agent"}

        agent_id = self.agents[self.current_agent]

        # Send message
        response = client.send_message(
            agent_id=agent_id,
            message=message,
            role="user"
        )

        response_text = self._extract_text(response)

        # Check if agent wants to handoff
        handoff = self._detect_handoff(response_text)

        result = {
            "agent": self.current_agent,
            "response": response_text,
            "handoff": handoff
        }

        # Log conversation
        self.conversation_history.append(result)

        # Execute handoff if needed
        if handoff:
            self.current_agent = handoff['to_agent']
            result['handoff_executed'] = True

        return result

    def _detect_handoff(self, response: str) -> Dict:
        """Detect if agent wants to hand off to another agent."""
        # Simple detection - in production, use structured output
        for agent_name in self.agents.keys():
            if f"transfer to {agent_name}" in response.lower():
                return {"to_agent": agent_name, "reason": "explicit transfer"}

        return None

    def _extract_text(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Create agents with handoff capability
sales_agent = client.create_agent(
    name="SalesAgent",
    persona="""You handle sales inquiries. If customer asks technical questions,
    say 'Let me transfer to Support' and they'll be connected to technical support."""
)

support_agent = client.create_agent(
    name="SupportAgent",
    persona="""You handle technical support. If customer wants to make a purchase,
    say 'Let me transfer to SalesAgent' to connect them with sales."""
)

# Setup handoff system
handoff_system = AgentHandoffSystem({
    "SalesAgent": sales_agent.id,
    "SupportAgent": support_agent.id
})

# Simulate conversation with handoffs
conversation = [
    "I'm interested in buying your product",
    "How do I install it?",
    "OK, I'd like to purchase now"
]

handoff_system.start_conversation("SalesAgent", conversation[0])

for message in conversation[1:]:
    result = handoff_system.send_message(message)
    print(f"\n👤 User: {message}")
    print(f"🤖 {result['agent']}: {result['response']}")
    if result.get('handoff'):
        print(f"🔄 Handoff to {result['handoff']['to_agent']}")
```

## Consensus-Based Multi-Agent Decisions

Multiple agents vote on decisions:

```python
class ConsensusSystem:
    """Multiple agents discuss and reach consensus."""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids

    def get_consensus(self, question: str, options: List[str]) -> Dict:
        """
        Get consensus from multiple agents.

        Args:
            question: Question to ask agents
            options: List of possible options

        Returns:
            Consensus result with votes
        """
        votes = {}
        responses = {}

        # Collect votes from each agent
        for agent_id in self.agent_ids:
            prompt = f"""{question}

Options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

Please choose one option and explain your reasoning.
Respond with: VOTE: [number] - [your explanation]"""

            response = client.send_message(
                agent_id=agent_id,
                message=prompt,
                role="user"
            )

            response_text = self._extract_text(response)
            responses[agent_id] = response_text

            # Parse vote
            vote = self._parse_vote(response_text, len(options))
            if vote:
                votes[agent_id] = vote

        # Calculate consensus
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        winning_vote = max(vote_counts, key=vote_counts.get) if vote_counts else None

        return {
            "question": question,
            "options": options,
            "votes": votes,
            "responses": responses,
            "vote_counts": vote_counts,
            "consensus": options[winning_vote - 1] if winning_vote else None,
            "unanimous": len(set(votes.values())) == 1
        }

    def _parse_vote(self, response: str, max_options: int) -> int:
        """Parse vote from response."""
        import re
        match = re.search(r'VOTE:\s*(\d+)', response)
        if match:
            vote = int(match.group(1))
            if 1 <= vote <= max_options:
                return vote
        return None

    def _extract_text(self, response) -> str:
        messages = [
            msg.text for msg in response.messages
            if msg.message_type == "assistant_message"
        ]
        return " ".join(messages)

# Create diverse agents for decision-making
agents = [
    client.create_agent(name="Conservative", persona="You prefer safe, proven approaches"),
    client.create_agent(name="Innovative", persona="You prefer cutting-edge, novel solutions"),
    client.create_agent(name="Practical", persona="You focus on practicality and cost-effectiveness")
]

# Get consensus
consensus_system = ConsensusSystem([a.id for a in agents])

result = consensus_system.get_consensus(
    question="What technology should we use for our new project?",
    options=["Proven frameworks (React, Django)", "Cutting-edge (Bun, Fresh)", "Simple tools (HTML, Flask)"]
)

print(f"\n🗳️  Consensus Result:")
print(f"Question: {result['question']}")
print(f"Consensus: {result['consensus']}")
print(f"Unanimous: {result['unanimous']}")
print(f"\nVote breakdown: {result['vote_counts']}")
```

## Monitoring Multi-Agent Systems

```python
class MultiAgentMonitor:
    """Monitor health and performance of multi-agent systems."""

    def __init__(self, agent_ids: Dict[str, str]):
        self.agent_ids = agent_ids
        self.metrics = {agent: {"messages": 0, "errors": 0} for agent in agent_ids}

    def track_message(self, agent_name: str, success: bool = True):
        """Track message sent to agent."""
        if agent_name in self.metrics:
            self.metrics[agent_name]["messages"] += 1
            if not success:
                self.metrics[agent_name]["errors"] += 1

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "total_agents": len(self.agent_ids),
            "agent_metrics": self.metrics,
            "total_messages": sum(m["messages"] for m in self.metrics.values()),
            "total_errors": sum(m["errors"] for m in self.metrics.values())
        }

    def health_check(self) -> bool:
        """Check if all agents are healthy."""
        for agent_name, agent_id in self.agent_ids.items():
            try:
                # Try to get agent
                agent = client.get_agent(agent_id)
                if not agent:
                    return False
            except:
                return False
        return True

# Usage
monitor = MultiAgentMonitor({
    "Agent1": "agent-1-id",
    "Agent2": "agent-2-id"
})

monitor.track_message("Agent1", success=True)
monitor.track_message("Agent2", success=False)

print(monitor.get_status())
print(f"System healthy: {monitor.health_check()}")
```

## Best Practices

1. **Clear Specialization**: Each agent should have a distinct, well-defined role
2. **Explicit Communication**: Use structured formats for inter-agent communication
3. **Error Handling**: Implement fallbacks when agents fail
4. **Monitoring**: Track agent performance and health
5. **Resource Management**: Limit concurrent agent executions
6. **Testing**: Test agent interactions thoroughly

## Common Patterns Summary

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| Hub-and-Spoke | Central coordination | Medium |
| Pipeline | Sequential processing | Low |
| Parallel | Independent tasks | Medium |
| Handoff | Dynamic routing | High |
| Consensus | Decision making | High |

## Next Steps

You've learned multi-agent orchestration! Next, we'll explore **custom memory architectures** - building sophisticated memory systems tailored to your needs.

## Resources

- [Multi-Agent Systems Theory](https://en.wikipedia.org/wiki/Multi-agent_system)
- [Agent Communication Languages](https://www.fipa.org/repository/aclspecs.html)
- [Letta Multi-Agent Examples](https://github.com/letta-ai/letta/tree/main/examples/multi-agent)

Build something amazing with your agent teams!

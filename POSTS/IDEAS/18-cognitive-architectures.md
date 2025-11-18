---
title: "Cognitive Architectures for Letta: Building Thinking Agents"
date: "2025-03-05"
description: "Implement advanced cognitive architectures including reasoning loops, planning systems, and self-reflection for sophisticated Letta agents."
tags: ["Letta", "Cognitive Architecture", "Reasoning", "AI", "Expert"]
---

# Cognitive Architectures for Letta

Move beyond reactive agents to create agents that think, plan, and reflect. This expert tutorial covers cognitive architectures for Letta.

## ReAct: Reasoning + Acting

```python
class ReActAgent:
    """Agent using ReAct (Reasoning and Acting) pattern."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
    
    def solve_with_react(self, problem: str, max_iterations: int = 5) -> Dict:
        """
        Solve problem using ReAct loop:
        1. Think (reason about problem)
        2. Act (take action)
        3. Observe (examine result)
        4. Repeat until solved
        """
        
        history = []
        
        for iteration in range(max_iterations):
            # THINK: Reason about current state
            thought = self._think(problem, history)
            history.append({"type": "thought", "content": thought})
            
            # ACT: Decide and execute action
            action = self._act(thought)
            history.append({"type": "action", "content": action})
            
            # OBSERVE: Get result of action
            observation = self._observe(action)
            history.append({"type": "observation", "content": observation})
            
            # Check if problem is solved
            if self._is_solved(observation):
                return {
                    "solved": True,
                    "iterations": iteration + 1,
                    "history": history,
                    "solution": observation
                }
        
        return {
            "solved": False,
            "iterations": max_iterations,
            "history": history
        }
    
    def _think(self, problem: str, history: List[Dict]) -> str:
        """Reasoning step."""
        context = self._format_history(history)
        
        prompt = f"""Problem: {problem}

{context}

Think step by step about how to solve this. What should you do next?

Thought:"""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        return self._extract_response(response)
    
    def _act(self, thought: str) -> str:
        """Action step based on reasoning."""
        prompt = f"""Based on this reasoning:
{thought}

What specific action should you take?

Action:"""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        return self._extract_response(response)
```

## Chain of Thought Prompting

```python
class ChainOfThoughtAgent:
    """Agent with explicit reasoning chains."""
    
    def solve_with_cot(self, problem: str) -> Dict:
        """
        Solve problem by explicitly generating reasoning chain.
        """
        
        prompt = f"""Problem: {problem}

Let's solve this step by step:

Step 1: Understand what's being asked
Step 2: Identify what information we have
Step 3: Determine what we need to find
Step 4: Break down the solution process
Step 5: Execute the solution
Step 6: Verify the answer

Please work through each step explicitly."""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        reasoning = self._extract_response(response)
        
        # Extract final answer
        answer = self._extract_final_answer(reasoning)
        
        return {
            "problem": problem,
            "reasoning_chain": reasoning,
            "answer": answer
        }
```

## Tree of Thoughts

```python
class TreeOfThoughtsAgent:
    """Explore multiple reasoning paths simultaneously."""
    
    def solve_with_tot(self, problem: str, branching_factor: int = 3) -> Dict:
        """
        Generate multiple reasoning paths and select best.
        """
        
        # Generate initial thoughts
        initial_thoughts = self._generate_thoughts(problem, branching_factor)
        
        # Evaluate each thought
        evaluated = []
        for thought in initial_thoughts:
            score = self._evaluate_thought(thought, problem)
            evaluated.append({"thought": thought, "score": score})
        
        # Select best paths
        evaluated.sort(key=lambda x: x["score"], reverse=True)
        best_paths = evaluated[:branching_factor]
        
        # Expand best paths
        final_solutions = []
        for path in best_paths:
            solution = self._expand_path(path["thought"], problem)
            final_solutions.append(solution)
        
        # Select best solution
        best_solution = max(final_solutions, key=lambda x: x["confidence"])
        
        return {
            "problem": problem,
            "explored_paths": len(initial_thoughts),
            "solution": best_solution
        }
```

## Self-Reflection and Critique

```python
class ReflectiveAgent:
    """Agent that reflects on and improves its own outputs."""
    
    def generate_with_reflection(
        self,
        task: str,
        max_reflections: int = 3
    ) -> Dict:
        """
        Generate solution with iterative self-reflection.
        """
        
        # Initial attempt
        attempt = self._generate_initial(task)
        reflections = []
        
        for i in range(max_reflections):
            # Reflect on current attempt
            critique = self._reflect(task, attempt)
            reflections.append(critique)
            
            # If critique is positive, we're done
            if self._is_satisfied(critique):
                break
            
            # Improve based on critique
            attempt = self._improve(task, attempt, critique)
        
        return {
            "task": task,
            "final_output": attempt,
            "reflections": reflections,
            "improvement_iterations": len(reflections)
        }
    
    def _reflect(self, task: str, attempt: str) -> str:
        """Generate critique of current attempt."""
        
        prompt = f"""Task: {task}

Current attempt:
{attempt}

Please critique this attempt:
- What is done well?
- What could be improved?
- Are there any errors or omissions?
- How would you rate it on a scale of 1-10?

Critique:"""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        return self._extract_response(response)
    
    def _improve(self, task: str, attempt: str, critique: str) -> str:
        """Improve attempt based on critique."""
        
        prompt = f"""Task: {task}

Previous attempt:
{attempt}

Critique of previous attempt:
{critique}

Please generate an improved version addressing the critique:

Improved version:"""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        return self._extract_response(response)
```

## Planning and Goal Decomposition

```python
class PlanningAgent:
    """Agent with explicit planning capabilities."""
    
    def plan_and_execute(self, goal: str) -> Dict:
        """
        Create plan then execute it step by step.
        """
        
        # Phase 1: Planning
        plan = self._create_plan(goal)
        
        # Phase 2: Execution
        results = []
        for step in plan["steps"]:
            result = self._execute_step(step)
            results.append(result)
            
            # Check if we need to replan
            if result["status"] == "failed":
                plan = self._replan(goal, plan, results)
        
        return {
            "goal": goal,
            "plan": plan,
            "execution_results": results,
            "success": all(r["status"] == "success" for r in results)
        }
    
    def _create_plan(self, goal: str) -> Dict:
        """Generate step-by-step plan."""
        
        prompt = f"""Goal: {goal}

Create a detailed step-by-step plan to achieve this goal:

1. Break down the goal into concrete sub-goals
2. For each sub-goal, list specific actions
3. Identify dependencies between steps
4. Estimate difficulty and time for each step

Plan:"""
        
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=prompt,
            role="user"
        )
        
        plan_text = self._extract_response(response)
        steps = self._parse_plan(plan_text)
        
        return {"goal": goal, "steps": steps}
```

## Meta-Learning: Learning to Learn

```python
class MetaLearningAgent:
    """Agent that learns from experience."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = create_client()
        self.experience_db = []
    
    def solve_with_experience(self, problem: str) -> Dict:
        """Solve problem using past experience."""
        
        # Retrieve similar past problems
        similar = self._find_similar_problems(problem)
        
        # Learn from similar problems
        insights = self._extract_insights(similar)
        
        # Apply insights to current problem
        solution = self._solve_with_insights(problem, insights)
        
        # Store experience
        self.experience_db.append({
            "problem": problem,
            "solution": solution,
            "timestamp": datetime.now()
        })
        
        return solution
    
    def _find_similar_problems(self, problem: str) -> List[Dict]:
        """Find similar past problems."""
        # Use embedding similarity
        problem_embedding = self.encoder.encode(problem)
        
        similarities = []
        for exp in self.experience_db:
            exp_embedding = self.encoder.encode(exp["problem"])
            similarity = cosine_similarity(problem_embedding, exp_embedding)
            similarities.append((exp, similarity))
        
        # Return top 3 most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:3]]
```

## Next Steps

Explore **Enterprise Deployment Patterns** - scaling to organizational level.

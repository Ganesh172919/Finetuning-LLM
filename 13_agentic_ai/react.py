"""
################################################################################
ReAct AGENT — REASONING + ACTING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is ReAct?
    ReAct (Reasoning + Acting) is an agent framework that combines
    reasoning and action in an interleaved manner.

    The agent:
    1. Thinks about what to do (Reasoning)
    2. Takes an action (Acting)
    3. Observes the result
    4. Repeats until task complete

Why does it matter?
    Pure reasoning: can think but can't act
    Pure acting: can act but can't plan
    ReAct: combines both for better problem solving

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Task: "Book a flight to Paris"                    │
    │                                                   │
    │ Thought 1: I need to search for flights           │
    │ Action 1: search_flights(destination="Paris")     │
    │ Observation 1: Found 5 flights...                 │
    │                                                   │
    │ Thought 2: I should compare prices                │
    │ Action 2: compare_prices(flights)                 │
    │ Observation 2: Cheapest is $500...                │
    │                                                   │
    │ Thought 3: I'll book the cheapest                 │
    │ Action 3: book_flight(flight_id=3)                │
    │ Observation 3: Booking confirmed!                 │
    │                                                   │
    │ Answer: Booked flight to Paris for $500           │
    └─────────────────────────────────────────────────┘

Interview Questions:
    1. "What is ReAct?"
       A framework where agents alternate between reasoning
       and acting. The agent thinks about what to do, does it,
       observes the result, and repeats.

    2. "How is ReAct different from chain of thought?"
       CoT only reasons. ReAct reasons AND acts, using tools
       to gather information and affect the world.

    3. "When should I use ReAct?"
       When the task requires external information or actions.
       E.g., booking flights, searching the web, executing code.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

################################################################################
# SECTION 1: TOOL DEFINITION
################################################################################

@dataclass
class Tool:
    """
    Tool that an agent can use.

    Attributes:
        name: Tool name
        description: What the tool does
        function: The actual function to call
        parameters: Parameter schema
    """
    name: str
    description: str
    function: Callable
    parameters: Dict

    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        return self.function(**kwargs)


################################################################################
# SECTION 2: REACT AGENT
################################################################################

class ReActAgent:
    """
    ReAct Agent
    ===========

    An agent that reasons and acts in an interleaved manner.

    Loop:
    1. Think: What should I do next?
    2. Act: Use a tool
    3. Observe: What happened?
    4. Repeat until done

    Interview Question:
        "How does a ReAct agent work?"
        It alternates between reasoning (thinking about what to do)
        and acting (using tools). After each action, it observes
        the result and reasons about the next step.
    """

    def __init__(self, tools: List[Tool], max_steps: int = 10):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.history: List[Dict] = []

    def think(self, task: str, observation: Optional[str] = None) -> str:
        """
        Generate a thought about what to do next.

        In production, this uses an LLM.
        """
        # Simplified: return a thought based on task
        if observation:
            return f"Based on the observation, I should analyze: {observation[:100]}"
        return f"I need to: {task}"

    def act(self, thought: str) -> str:
        """
        Choose and execute an action.

        In production, this uses an LLM to select tool and parameters.
        """
        # Simplified: use first tool
        tool_name = list(self.tools.keys())[0]
        tool = self.tools[tool_name]

        # Execute tool
        result = tool.execute()
        return result

    def run(self, task: str) -> str:
        """
        Run the agent on a task.

        Args:
            task: The task to accomplish

        Returns:
            Final answer
        """
        observation = None

        for step in range(self.max_steps):
            # Think
            thought = self.think(task, observation)
            self.history.append({'type': 'thought', 'content': thought})

            # Act
            action_result = self.act(thought)
            self.history.append({'type': 'action', 'content': action_result})

            # Observe
            observation = action_result
            self.history.append({'type': 'observation', 'content': observation})

            # Check if done (simplified)
            if 'complete' in observation.lower() or 'done' in observation.lower():
                break

        # Generate final answer
        answer = f"Task completed: {task}"
        return answer

    def get_trajectory(self) -> List[Dict]:
        """Get the full reasoning trajectory."""
        return self.history


################################################################################
# SECTION 3: EXAMPLE TOOLS
################################################################################

def search_tool(query: str = "") -> str:
    """Example search tool."""
    return f"Search results for: {query}"


def calculator_tool(expression: str = "") -> str:
    """Example calculator tool."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_react():
    """Demonstrate ReAct agent."""
    print("=" * 70)
    print("ReAct AGENT DEMONSTRATION")
    print("=" * 70)

    # Create tools
    print("\n--- Creating Tools ---")
    tools = [
        Tool(
            name="search",
            description="Search for information",
            function=search_tool,
            parameters={"query": "string"}
        ),
        Tool(
            name="calculator",
            description="Calculate expressions",
            function=calculator_tool,
            parameters={"expression": "string"}
        )
    ]

    # Create agent
    print("\n--- Creating Agent ---")
    agent = ReActAgent(tools=tools, max_steps=5)

    # Run agent
    print("\n--- Running Agent ---")
    task = "What is the capital of France?"
    answer = agent.run(task)
    print(f"Task: {task}")
    print(f"Answer: {answer}")

    # Show trajectory
    print("\n--- Agent Trajectory ---")
    for step in agent.get_trajectory():
        print(f"[{step['type']}]: {step['content'][:80]}...")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_react()


################################################################################
# REFERENCES
################################################################################

# [1] Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models.

################################################################################

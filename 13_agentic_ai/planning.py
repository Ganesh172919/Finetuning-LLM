"""
################################################################################
PLAN-AND-EXECUTE AGENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Plan-and-Execute?
    An agent framework that first creates a plan, then executes it.

    Unlike ReAct (interleaved thinking and acting):
    - Plan-and-Execute: plan first, then execute
    - ReAct: think and act one step at a time

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Task: "Write a research paper on AI"             │
    │        ↓                                          │
    │ Planner: Create step-by-step plan                │
    │   1. Research topic                               │
    │   2. Write outline                                │
    │   3. Write introduction                           │
    │   4. Write body sections                          │
    │   5. Write conclusion                             │
    │   6. Review and edit                              │
    │        ↓                                          │
    │ Executor: Execute each step                       │
    │        ↓                                          │
    │ Result: Complete paper                            │
    └─────────────────────────────────────────────────┘

Interview Questions:
    1. "What's the difference between ReAct and Plan-and-Execute?"
       ReAct: interleaved thinking and acting
       Plan-and-Execute: plan first, then execute

    2. "When should I use Plan-and-Execute?"
       When the task requires multi-step planning.
       Better for complex, structured tasks.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional

################################################################################
# SECTION 1: PLANNER
################################################################################

class Planner:
    """
    Planner: Creates step-by-step plans.

    Takes a task and breaks it into executable steps.
    """

    def create_plan(self, task: str) -> List[str]:
        """
        Create a plan for the task.

        Args:
            task: The task to accomplish

        Returns:
            List of steps
        """
        # Simplified planning
        steps = [
            f"Step 1: Understand the task: {task[:50]}...",
            "Step 2: Gather necessary information",
            "Step 3: Execute main task",
            "Step 4: Verify and refine",
            "Step 5: Finalize"
        ]
        return steps


################################################################################
# SECTION 2: EXECUTOR
################################################################################

class Executor:
    """
    Executor: Executes plan steps.

    Takes a step and executes it using available tools.
    """

    def execute_step(self, step: str) -> str:
        """
        Execute a single step.

        Args:
            step: The step to execute

        Returns:
            Result of execution
        """
        # Simplified execution
        return f"Completed: {step}"


################################################################################
# SECTION 3: PLAN-AND-EXECUTE AGENT
################################################################################

class PlanAndExecuteAgent:
    """
    Plan-and-Execute Agent
    ======================

    Combines planning and execution for complex tasks.

    Interview Question:
        "How does a Plan-and-Execute agent work?"
        First creates a detailed plan, then executes each step.
        Can replan if execution reveals new information.
    """

    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()

    def run(self, task: str) -> Dict:
        """
        Run the agent on a task.

        Args:
            task: The task to accomplish

        Returns:
            Dictionary with plan and results
        """
        # Create plan
        plan = self.planner.create_plan(task)

        # Execute each step
        results = []
        for step in plan:
            result = self.executor.execute_step(step)
            results.append(result)

        return {
            'task': task,
            'plan': plan,
            'results': results,
            'status': 'completed'
        }


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_planning():
    """Demonstrate planning agent."""
    print("=" * 70)
    print("PLAN-AND-EXECUTE AGENT DEMONSTRATION")
    print("=" * 70)

    # Create agent
    agent = PlanAndExecuteAgent()

    # Run task
    task = "Write a summary of machine learning"
    result = agent.run(task)

    print(f"\nTask: {result['task']}")
    print(f"\nPlan:")
    for step in result['plan']:
        print(f"  {step}")
    print(f"\nResults:")
    for r in result['results']:
        print(f"  {r}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_planning()

"""
################################################################################
AGENT TOOLS — EXTERNAL CAPABILITIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Agent Tools?
    Tools are external capabilities that agents can use:
    - Web search
    - Code execution
    - File operations
    - API calls
    - Database queries

Why do they matter?
    Tools extend agent capabilities beyond text generation:
    - Search: find up-to-date information
    - Code: execute and test code
    - APIs: interact with external services

Interview Questions:
    1. "How do agents use tools?"
        The agent generates a tool call (function name + args).
        The system executes the tool and returns the result.
        The agent uses the result to continue reasoning.

    2. "How do you design good tools?"
        Clear description, well-defined parameters,
        helpful error messages, safe execution.

################################################################################
"""

import numpy as np
from typing import List, Dict, Callable, Any
from dataclasses import dataclass

################################################################################
# SECTION 1: TOOL DEFINITION
################################################################################

@dataclass
class Tool:
    """
    Tool definition for agents.

    Attributes:
        name: Tool name
        description: What the tool does
        parameters: Parameter schema
        function: The actual function
    """
    name: str
    description: str
    parameters: Dict[str, str]
    function: Callable

    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"Error: {str(e)}"


################################################################################
# SECTION 2: TOOL REGISTRY
################################################################################

class ToolRegistry:
    """
    Registry of available tools.

    Manages tool registration and lookup.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self.tools.keys())

    def get_descriptions(self) -> str:
        """Get formatted tool descriptions."""
        desc = "Available tools:\n"
        for name, tool in self.tools.items():
            desc += f"- {name}: {tool.description}\n"
        return desc


################################################################################
# SECTION 3: EXAMPLE TOOLS
################################################################################

def search(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"


def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def read_file(path: str) -> str:
    """Read a file."""
    return f"Contents of {path}"


def write_file(path: str, content: str) -> str:
    """Write to a file."""
    return f"Wrote to {path}"


################################################################################
# SECTION 4: DEFAULT TOOL SET
################################################################################

def get_default_tools() -> ToolRegistry:
    """Get default set of tools."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="search",
        description="Search the web for information",
        parameters={"query": "string"},
        function=search
    ))

    registry.register(Tool(
        name="calculate",
        description="Calculate a mathematical expression",
        parameters={"expression": "string"},
        function=calculate
    ))

    registry.register(Tool(
        name="read_file",
        description="Read a file",
        parameters={"path": "string"},
        function=read_file
    ))

    registry.register(Tool(
        name="write_file",
        description="Write to a file",
        parameters={"path": "string", "content": "string"},
        function=write_file
    ))

    return registry


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_tools():
    """Demonstrate tool concepts."""
    print("=" * 70)
    print("AGENT TOOLS DEMONSTRATION")
    print("=" * 70)

    # Create registry
    registry = get_default_tools()

    # List tools
    print("\n--- Available Tools ---")
    print(registry.get_descriptions())

    # Execute tools
    print("\n--- Executing Tools ---")
    search_tool = registry.get("search")
    result = search_tool.execute(query="What is AI?")
    print(f"Search: {result}")

    calc_tool = registry.get("calculate")
    result = calc_tool.execute(expression="2 + 2")
    print(f"Calculate: {result}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tools()

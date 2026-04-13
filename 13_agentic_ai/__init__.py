"""
################################################################################
AGENTIC AI SYSTEMS — AUTONOMOUS AI AGENTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Agentic AI Systems?
    AI systems that can plan, reason, and take actions autonomously.
    They go beyond simple text generation to:
    - Break down complex tasks
    - Use tools (web search, code execution, APIs)
    - Maintain memory across interactions
    - Learn from feedback

Why does it matter?
    Agentic AI enables:
    - Automated research (reading papers, writing reports)
    - Code development (writing, testing, debugging)
    - Customer support (handling complex queries)
    - Scientific discovery (hypothesis generation, experimentation)

Historical Evolution:
    - 2022: ReAct (Reasoning + Acting)
    - 2023: AutoGPT, BabyAGI
    - 2024: Claude Computer Use, OpenAI Assistants
    - 2025: Multi-agent systems, tool use training
    - 2026: Autonomous agents in production

########################################

AGENT FRAMEWORKS:
1. ReAct: Reason then act
2. Plan-and-Execute: Plan first, then execute
3. Reflexion: Learn from mistakes
4. Tool Use: Use external tools
5. Multi-Agent: Multiple agents collaborating

################################################################################
"""

from .react import ReActAgent
from .planning import PlanAndExecuteAgent
from .tools import Tool, ToolRegistry
from .tool_use_rl import ToolUsePolicy, GRPOToolTrainer, ToolAugmentedReasoning
from .search_agent import MCTS, BeamSearchAgent, TreeOfThoughts, RLGuidedSearch
from .multi_agent_rl import MultiAgentPolicy, MultiAgentTrainer, SelfPlayTrainer
from .agent_reward_model import AgentRewardModel, OutcomeRewardModel, ProcessRewardModel

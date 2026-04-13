"""
################################################################################
CLAUDE 5 — ANTHROPIC'S LATEST MODEL FAMILY (2026)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Claude 5?
    Anthropic's latest model family, released June 2026.
    Introduces a new "Mythos" tier above Opus.

Claude 5 Model Tiers (from smallest to largest):
    1. Haiku 4.5 (Oct 2025) — Fast, cheap, good for simple tasks
    2. Sonnet 5 (June 2026) — Mid-tier, 1M token context window
    3. Opus 4.8 (May 2026) — Previous flagship
    4. Fable 5 (June 2026) — Mythos-class with safety guardrails
    5. Mythos 5 (June 2026) — Most capable, limited access

Key Developments in 2026:
    - Claude 4.6 (Feb 2026): "Agent teams" — multiple agents collaborating
    - Claude 4.7 (Apr 2026): Controversial — excessive refusals
    - Claude 4.8 (May 2026): Final Opus 4.x release
    - Claude Fable 5 (June 2026): Mythos-class with safety
    - Claude Sonnet 5 (June 2026): 1M token context
    - Claude Mythos 5 (June 2026): Most capable, restricted access

Agent Teams (Claude 4.6 Innovation):
    - 16 Opus 4.6 agents collaborated to write a C compiler in Rust
    - Successfully compiled the Linux kernel
    - Demonstrated multi-agent software engineering

Safety Levels:
    - Level 1: Standard safety
    - Level 2: Enhanced safety
    - Level 3: "Significantly higher risk" (Opus 4, Mythos 5)

Interview Questions:
    1. "What are Claude 5's model tiers?"
        Haiku (fast), Sonnet (mid), Opus (capable), Fable (safe Mythos),
        Mythos (most capable, restricted). Each tier balances capability,
        speed, cost, and safety differently.

    2. "What are agent teams?"
        Multiple Claude agents collaborating on complex tasks.
        Claude 4.6 demonstrated this by having 16 agents write a
        C compiler in Rust that compiled the Linux kernel.

    3. "Why was Claude 4.7 controversial?"
        Users reported excessive refusals and token burn.
        35 false-positive refusal reports filed in April 2026.
        Anthropic worked to fix the balance between safety and utility.

################################################################################
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

################################################################################
# SECTION 1: MODEL TIERS
################################################################################

class ModelTier(Enum):
    """
    Claude 5 Model Tiers.

    Each tier represents a different balance of:
    - Capability (reasoning, coding, analysis)
    - Speed (tokens per second)
    - Cost (price per token)
    - Safety (guardrail strictness)
    """
    HAIKU_45 = "claude-haiku-4-5"        # Fastest, cheapest
    SONNET_5 = "claude-sonnet-5"          # Mid-tier, 1M context
    OPUS_48 = "claude-opus-4-8"           # Previous flagship
    FABLE_5 = "claude-fable-5"            # Mythos with safety
    MYTHOS_5 = "claude-mythos-5"          # Most capable, restricted


@dataclass
class ModelConfig:
    """
    Configuration for each Claude model.

    Attributes:
        tier: Model tier
        context_window: Maximum context length
        max_output: Maximum output tokens
        safety_level: Safety guardrail level (1-3)
        cost_per_1k_input: Cost per 1K input tokens
        cost_per_1k_output: Cost per 1K output tokens
        capabilities: List of special capabilities
    """
    tier: ModelTier
    context_window: int
    max_output: int
    safety_level: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: List[str]


# Model configurations (approximate)
CLAUDE_MODELS = {
    ModelTier.HAIKU_45: ModelConfig(
        tier=ModelTier.HAIKU_45,
        context_window=200_000,
        max_output=8_192,
        safety_level=1,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
        capabilities=["text", "vision", "code"]
    ),
    ModelTier.SONNET_5: ModelConfig(
        tier=ModelTier.SONNET_5,
        context_window=1_000_000,  # 1M tokens!
        max_output=16_384,
        safety_level=2,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        capabilities=["text", "vision", "code", "extended_thinking"]
    ),
    ModelTier.OPUS_48: ModelConfig(
        tier=ModelTier.OPUS_48,
        context_window=200_000,
        max_output=16_384,
        safety_level=2,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        capabilities=["text", "vision", "code", "extended_thinking", "fast_mode"]
    ),
    ModelTier.FABLE_5: ModelConfig(
        tier=ModelTier.FABLE_5,
        context_window=500_000,
        max_output=32_768,
        safety_level=2,  # Safety guardrails
        cost_per_1k_input=0.020,
        cost_per_1k_output=0.100,
        capabilities=["text", "vision", "code", "extended_thinking", "advanced_reasoning"]
    ),
    ModelTier.MYTHOS_5: ModelConfig(
        tier=ModelTier.MYTHOS_5,
        context_window=500_000,
        max_output=32_768,
        safety_level=3,  # "Significantly higher risk"
        cost_per_1k_input=0.025,
        cost_per_1k_output=0.125,
        capabilities=["text", "vision", "code", "extended_thinking",
                      "advanced_reasoning", "unrestricted_domains"]
    ),
}


################################################################################
# SECTION 2: SAFETY SYSTEM
################################################################################

class SafetyLevel(Enum):
    """Safety levels for Claude models."""
    STANDARD = 1      # Basic safety
    ENHANCED = 2      # More conservative
    HIGH_RISK = 3     # Requires special access


class SafetyGuardrail:
    """
    Claude 5 Safety System.

    Different models have different safety levels:
    - Fable 5: Has safety guardrails, downgrades to Opus 4.8 for high-risk
    - Mythos 5: Less restricted, but requires trusted access

    High-risk domains:
    - Cybersecurity (exploits, attacks)
    - Biology (dangerous pathogens)
    - Chemical weapons
    - Nuclear weapons

    Interview Question:
        "How does Claude handle safety?"
        Multiple levels: standard, enhanced, high-risk.
        Fable 5 has guardrails that restrict high-risk domains,
        falling back to Opus 4.8 for sensitive requests.
        Mythos 5 is less restricted but requires trusted access.
    """

    def __init__(self, level: SafetyLevel = SafetyLevel.STANDARD):
        self.level = level
        self.high_risk_domains = [
            "cybersecurity_exploits",
            "dangerous_biology",
            "chemical_weapons",
            "nuclear_weapons",
        ]

    def check_request(self, request: str) -> Dict[str, Any]:
        """
        Check if request is safe for current model.

        Args:
            request: User's request text

        Returns:
            Dictionary with safety assessment
        """
        is_high_risk = self._detect_high_risk(request)

        if is_high_risk and self.level == SafetyLevel.ENHANCED:
            return {
                "allowed": False,
                "reason": "High-risk domain detected",
                "fallback": "claude-opus-4-8",
                "message": "This request requires a less restricted model"
            }
        elif is_high_risk and self.level == SafetyLevel.HIGH_RISK:
            return {
                "allowed": True,
                "reason": "High-risk allowed for trusted access",
                "warning": "Response may be restricted"
            }
        else:
            return {
                "allowed": True,
                "reason": "Standard request"
            }

    def _detect_high_risk(self, request: str) -> bool:
        """Detect if request touches high-risk domains."""
        high_risk_keywords = [
            "exploit", "attack", "vulnerability",
            "pathogen", "weapon", "dangerous",
        ]
        request_lower = request.lower()
        return any(kw in request_lower for kw in high_risk_keywords)


################################################################################
# SECTION 3: AGENT TEAMS
################################################################################

class AgentTeam:
    """
    Agent Teams — Claude 4.6 Innovation
    ====================================

    Multiple Claude agents working together on complex tasks.

    How it works:
        1. Lead agent decomposes task into subtasks
        2. Specialist agents work on subtasks in parallel
        3. Agents communicate and share results
        4. Lead agent integrates and verifies

    Achievement (Feb 2026):
        - 16 Opus 4.6 agents wrote a C compiler in Rust
        - Successfully compiled the Linux kernel
        - Demonstrated multi-agent software engineering

    Interview Question:
        "What are agent teams?"
        Multiple Claude agents collaborating on complex tasks.
        A lead agent decomposes work, specialist agents execute
        in parallel, and results are integrated. Claude 4.6
        demonstrated this by having 16 agents write a C compiler.
    """

    def __init__(self, n_agents: int = 4, lead_model: str = "opus"):
        """
        Args:
            n_agents: Number of specialist agents
            lead_model: Model for lead agent
        """
        self.n_agents = n_agents
        self.lead_model = lead_model
        self.agents = []
        self.results = {}

    def decompose_task(self, task: str) -> List[Dict[str, str]]:
        """
        Lead agent decomposes task into subtasks.

        Args:
            task: High-level task description

        Returns:
            List of subtask dictionaries
        """
        # In real implementation, this would call the lead agent
        subtasks = [
            {"id": f"subtask_{i}", "description": f"Part {i} of {task}",
             "agent_id": i}
            for i in range(self.n_agents)
        ]
        return subtasks

    def execute_parallel(self, subtasks: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Execute subtasks in parallel with specialist agents.

        Args:
            subtasks: List of subtasks to execute

        Returns:
            Dictionary of results from each agent
        """
        results = {}
        for subtask in subtasks:
            agent_id = subtask["agent_id"]
            # In real implementation, each agent would run independently
            results[agent_id] = {
                "status": "completed",
                "output": f"Result from agent {agent_id}",
                "subtask": subtask["id"]
            }
        return results

    def integrate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lead agent integrates results from all specialists.

        Args:
            results: Results from parallel execution

        Returns:
            Integrated final result
        """
        # In real implementation, lead agent would verify and integrate
        return {
            "status": "integrated",
            "n_agents": len(results),
            "all_completed": all(r["status"] == "completed"
                                for r in results.values()),
            "final_output": "Integrated result from all agents"
        }

    def run(self, task: str) -> Dict[str, Any]:
        """
        Full agent team workflow.

        Args:
            task: High-level task description

        Returns:
            Final integrated result
        """
        # 1. Decompose
        subtasks = self.decompose_task(task)

        # 2. Execute in parallel
        results = self.execute_parallel(subtasks)

        # 3. Integrate
        final = self.integrate_results(results)

        return final


################################################################################
# SECTION 4: EXTENDED THINKING
################################################################################

class ExtendedThinking:
    """
    Extended Thinking — Claude's Reasoning Capability
    ==================================================

    Introduced in Claude 3.7 Sonnet (Feb 2025).
    Model "thinks" for variable lengths before responding.

    How it works:
        1. Model generates internal reasoning (not shown to user)
        2. Reasoning can be short (seconds) or long (minutes)
        3. Final answer incorporates the reasoning

    Use cases:
        - Complex math problems
        - Multi-step logic
        - Code debugging
        - Analysis tasks

    Interview Question:
        "What is extended thinking?"
        A capability where Claude generates internal reasoning
        before responding. The thinking can be variable length,
        from seconds to minutes. Improves accuracy on complex
        tasks like math, logic, and code debugging.
    """

    def __init__(self, max_thinking_tokens: int = 10000):
        """
        Args:
            max_thinking_tokens: Maximum tokens for thinking
        """
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_budget_used = 0

    def think(self, problem: str) -> Dict[str, Any]:
        """
        Generate extended thinking for a problem.

        Args:
            problem: Problem to think about

        Returns:
            Dictionary with thinking process and answer
        """
        # Simulate thinking process
        thinking_steps = [
            "Let me break down this problem...",
            "First, I need to understand what's being asked...",
            "The key insight is...",
            "Let me verify this approach...",
            "After consideration, I conclude..."
        ]

        return {
            "problem": problem,
            "thinking_steps": thinking_steps,
            "thinking_tokens": len(thinking_steps) * 50,
            "has_thinking": True
        }

    def should_think(self, task_complexity: float) -> bool:
        """
        Decide whether to use extended thinking.

        Args:
            task_complexity: 0.0 (simple) to 1.0 (complex)

        Returns:
            Whether to use extended thinking
        """
        # Use thinking for complex tasks
        return task_complexity > 0.5


################################################################################
# SECTION 5: COMPLETE CLAUDE 5 SYSTEM
################################################################################

class Claude5:
    """
    Claude 5 Complete System
    =========================

    Combines all Claude 5 capabilities:
    1. Multiple model tiers (Haiku, Sonnet, Opus, Fable, Mythos)
    2. Safety system with domain restrictions
    3. Agent teams for complex tasks
    4. Extended thinking for reasoning
    5. 1M token context (Sonnet 5)

    Interview Question:
        "Describe Claude 5's architecture."
        Claude 5 is a family of models at different tiers:
        Haiku (fast), Sonnet (mid, 1M context), Opus (capable),
        Fable (Mythos with safety), Mythos (most capable).
        Key innovations include agent teams (multiple agents
        collaborating), extended thinking for reasoning, and
        a sophisticated safety system with domain restrictions.
    """

    def __init__(self, tier: ModelTier = ModelTier.SONNET_5):
        self.tier = tier
        self.config = CLAUDE_MODELS[tier]
        self.safety = SafetyGuardrail(
            SafetyLevel(self.config.safety_level)
        )
        self.thinking = ExtendedThinking()

    def generate(self, prompt: str, use_thinking: bool = False
                 ) -> Dict[str, Any]:
        """
        Generate a response.

        Args:
            prompt: User's prompt
            use_thinking: Whether to use extended thinking

        Returns:
            Response dictionary
        """
        # Safety check
        safety_check = self.safety.check_request(prompt)
        if not safety_check["allowed"]:
            return {
                "error": safety_check["reason"],
                "fallback": safety_check.get("fallback"),
                "message": safety_check["message"]
            }

        # Extended thinking if needed
        thinking_result = None
        if use_thinking:
            thinking_result = self.thinking.think(prompt)

        return {
            "model": self.tier.value,
            "response": f"Response to: {prompt}",
            "thinking": thinking_result,
            "context_window": self.config.context_window,
            "safety_level": self.config.safety_level
        }

    def create_team(self, n_agents: int = 4) -> AgentTeam:
        """
        Create an agent team for complex tasks.

        Args:
            n_agents: Number of agents in team

        Returns:
            AgentTeam instance
        """
        return AgentTeam(n_agents=n_agents, lead_model=self.tier.value)


################################################################################
# DEMO
################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDE 5 ARCHITECTURE DEMO")
    print("=" * 70)

    # Model tiers
    print("\n1. Model Tiers")
    print("-" * 40)
    for tier, config in CLAUDE_MODELS.items():
        print(f"   {tier.value}:")
        print(f"     Context: {config.context_window:,} tokens")
        print(f"     Safety: Level {config.safety_level}")
        print(f"     Cost: ${config.cost_per_1k_input}/1K in, "
              f"${config.cost_per_1k_output}/1K out")

    # Safety system
    print("\n2. Safety System")
    print("-" * 40)
    safety = SafetyGuardrail(SafetyLevel(2))  # Fable 5 level

    test_requests = [
        "Help me write a Python script",
        "Tell me about cybersecurity exploits",
        "Explain quantum computing",
    ]

    for req in test_requests:
        result = safety.check_request(req)
        status = "✓ Allowed" if result["allowed"] else "✗ Blocked"
        print(f"   '{req[:40]}...' → {status}")

    # Agent teams
    print("\n3. Agent Teams")
    print("-" * 40)
    team = AgentTeam(n_agents=4)
    result = team.run("Build a web application")
    print(f"   Task decomposition: {team.n_agents} agents")
    print(f"   Result: {result['all_completed']} completed")

    # Extended thinking
    print("\n4. Extended Thinking")
    print("-" * 40)
    thinking = ExtendedThinking()
    result = thinking.think("What is 2+2?")
    print(f"   Thinking steps: {len(result['thinking_steps'])}")
    print(f"   Tokens used: {result['thinking_tokens']}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. 5 model tiers: Haiku, Sonnet, Opus, Fable, Mythos")
    print("2. Sonnet 5 has 1M token context window")
    print("3. Agent teams: multiple agents collaborate")
    print("4. Extended thinking for complex reasoning")
    print("5. Safety system with domain restrictions")

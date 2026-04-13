"""
################################################################################
SUPERVISED FINE-TUNING (SFT) — INSTRUCTION FOLLOWING TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Supervised Fine-Tuning (SFT)?
    SFT is the process of training a pretrained language model on
    high-quality (instruction, response) pairs so it learns to follow
    instructions, maintain conversation structure, and produce helpful
    outputs. It bridges the gap between a "text completer" (pretrained
    model) and an "assistant" (instruction-following model).

Why does it matter?
    Without SFT, a pretrained model just predicts the next most likely
    token — it has no concept of "being helpful." SFT teaches the model
    to respond to questions, follow system prompts, use tools, and
    reason through problems in a structured format. This is the step
    that makes a raw language model actually useful.

How does it work?
    1. Format data into chat templates with system/user/assistant roles
    2. Tokenize with special tokens (for thinking, tool calls, etc.)
    3. Create loss masks that train only on completion tokens
    4. Run standard cross-entropy training with masked loss
    5. The model learns to produce assistant responses conditioned on
       the conversation history

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    SFT TRAINING PIPELINE                         │
    │                                                                   │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Chat Template Formatting                                │    │
    │  │                                                          │    │
    │  │  <|system|>You are a helpful assistant.</|system|>       │    │
    │  │  <|user|>What is 2+2?</|user|>                           │    │
    │  │  <|assistant|><|think|>Simple arithmetic...</|think|>    │    │
    │  │  The answer is 4.</|assistant|>                           │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Tokenization + Loss Masking                             │    │
    │  │                                                          │    │
    │  │  Tokens:  [sys] [user] Q [user_end] [asst] [th] R [/th] │    │
    │  │  Mask:     0     0     0    0        1     1    1   1   │    │
    │  │                        ↑                  ↑               │    │
    │  │                   prompt (mask)    completion (train)     │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Cross-Entropy Loss (masked)                             │    │
    │  │  L = -Σ mask_t * log P(x_t | x_{<t})                    │    │
    │  └──────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: InstructGPT / ChatGPT — RLHF pipeline with SFT as first step
    - 2023: LIMA — "Less Is More for Alignment" (1000 high-quality examples)
    - 2023: Alpaca/Vicuna — Open-source SFT on synthetic data
    - 2024: Thinking tokens / scratchpad — Models learn to reason visibly
    - 2025: Tool-call formatting — Structured function calling in chat templates
    - 2025: Multi-turn SFT — Training on full conversation histories

INTERVIEW QUESTIONS:
    1. "Why mask prompt tokens during SFT?"
       We only want the model to learn to GENERATE responses, not to
       predict the prompt. If we trained on prompt tokens too, the model
       would waste capacity memorizing input patterns. Masking ensures
       100% of the gradient signal teaches the model to produce good
       completions. This is especially important for long system prompts
       and few-shot examples.

    2. "What are thinking tokens and how do they help?"
       Thinking tokens (<|think|>...</|think|>) give the model a
       "scratchpad" to reason through problems before answering. The
       model learns to decompose complex problems, work through edge
       cases, and verify its reasoning — all in a structured block
       that can be shown or hidden from the user. This dramatically
       improves performance on math, coding, and logic tasks.

    3. "How much data do you need for SFT?"
       Less than you think. LIMA showed that 1000 high-quality examples
       can produce a strong assistant. Quality >> quantity. The key is
       diversity (different task types) and correctness (no hallucinated
       or misleading responses). For a general assistant, 10k-100k
       examples is typical. For domain-specific SFT, 1k-10k may suffice.

################################################################################
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrain import PretrainingConfig, Pretrainer


################################################################################
# SECTION 1: SPECIAL TOKENS AND CHAT TEMPLATE
################################################################################

# Special tokens used in the chat template
SPECIAL_TOKENS = {
    "system_start": "<|system|>",
    "system_end": "</|system|>",
    "user_start": "<|user|>",
    "user_end": "</|user|>",
    "assistant_start": "<|assistant|>",
    "assistant_end": "</|assistant|>",
    "tool_start": "<|tool|>",
    "tool_end": "</|tool|>",
    "tool_call_start": "<|tool_call|>",
    "tool_call_end": "</|tool_call|>",
    "tool_result_start": "<|tool_result|>",
    "tool_result_end": "</|tool_result|>",
    "think_start": "<|think|>",
    "think_end": "</|think|>",
    "pad": "<|pad|>",
    "eos": "<|eos|>",
}


def format_chat_message(role: str, content: str) -> str:
    """
    Format a single chat message into the template.

    Args:
        role: One of 'system', 'user', 'assistant', 'tool'.
        content: The message content.

    Returns:
        The formatted message string with special tokens.

    Example:
        >>> format_chat_message("user", "Hello!")
        '<|user|>Hello!</|user|>'
    """
    role_map = {
        "system": ("system_start", "system_end"),
        "user": ("user_start", "user_end"),
        "assistant": ("assistant_start", "assistant_end"),
        "tool": ("tool_start", "tool_end"),
    }
    if role not in role_map:
        raise ValueError(f"Unknown role: {role}. Must be one of {list(role_map.keys())}")

    start_token, end_token = role_map[role]
    return f"{SPECIAL_TOKENS[start_token]}{content}{SPECIAL_TOKENS[end_token]}"


def format_chat_conversation(
    messages: List[Dict[str, str]],
    include_thinking: bool = True,
) -> str:
    """
    Format a full conversation into the chat template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        include_thinking: Whether to include think tokens in assistant messages.

    Returns:
        The fully formatted conversation string.

    Explanation:
        The chat template encodes multi-turn conversations with clear role
        boundaries. Each role is wrapped in special tokens that the model
        learns to recognize and respect. The assistant's turn optionally
        includes a thinking block before the actual response.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hi!"},
        ...     {"role": "assistant", "content": "Hello!"},
        ... ]
        >>> format_chat_conversation(messages)
        '<|system|>You are helpful.</|system|><|user|>Hi!</|user|><|assistant|>Hello!</|assistant|>'
    """
    formatted_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant" and include_thinking and "thinking" in msg:
            thinking = msg["thinking"]
            think_block = f"{SPECIAL_TOKENS['think_start']}{thinking}{SPECIAL_TOKENS['think_end']}"
            formatted_parts.append(format_chat_message(role, think_block + content))
        else:
            formatted_parts.append(format_chat_message(role, content))

    return "".join(formatted_parts)


################################################################################
# SECTION 2: SFT CONFIGURATION
################################################################################


@dataclass
class SFTConfig(PretrainingConfig):
    """
    SFT Configuration
    =================

    Extends PretrainingConfig with SFT-specific settings.
    Inherits all pretraining hyperparameters and adds:
        - Chat template configuration
        - Loss masking options
        - Think-token handling

    Interview Question:
        "Should SFT use a different learning rate than pretraining?"
        Yes — SFT typically uses a much lower learning rate (1e-5 to 5e-5)
        compared to pretraining (1e-4 to 3e-4). The pretrained weights are
        already good; we want to make small adjustments, not overwrite
        knowledge. Using pretraining LR would cause catastrophic forgetting.
    """

    # ------------------------------------------------------------------
    # Chat template configuration
    # ------------------------------------------------------------------
    chat_template: str = "default"
    """Chat template name. 'default' uses the built-in template with system/user/assistant roles."""

    max_seq_len: int = 4096
    """Maximum sequence length for SFT examples."""

    # ------------------------------------------------------------------
    # Loss masking
    # ------------------------------------------------------------------
    mask_prompt_tokens: bool = True
    """If True, only compute loss on completion (assistant) tokens, not prompt tokens."""

    # ------------------------------------------------------------------
    # Think-token handling
    # ------------------------------------------------------------------
    think_token_enabled: bool = True
    """If True, include <|think|>...</|think|> blocks in the training data."""

    think_token_in_loss: bool = True
    """If True, include thinking tokens in the loss computation. If False, thinking is internal."""

    # ------------------------------------------------------------------
    # Tool-call handling
    # ------------------------------------------------------------------
    tool_call_in_loss: bool = True
    """If True, include tool-call blocks in the loss computation."""

    # ------------------------------------------------------------------
    # SFT-specific training parameters
    # ------------------------------------------------------------------
    learning_rate: float = 2e-5
    """SFT learning rate (much lower than pretraining)."""

    max_steps: int = 5000
    """SFT typically needs far fewer steps than pretraining."""

    warmup_steps: int = 100
    """Shorter warmup for SFT."""

    weight_decay: float = 0.01
    """Lower weight decay for SFT (less regularization needed)."""


################################################################################
# SECTION 3: TOKENIZER WRAPPER FOR SFT
################################################################################


class SFTTokenizerWrapper:
    """
    SFT Tokenizer Wrapper
    ======================

    Wraps a base tokenizer to handle chat template formatting,
    special token insertion, and loss mask creation.

    WHY this matters:
        The tokenizer must correctly handle special tokens (they should
        NOT be split into subwords) and produce accurate loss masks
        that identify which tokens to train on. Getting this wrong
        means the model trains on the wrong tokens — a silent but
        devastating bug.

    Interview Question:
        "How do you handle special tokens in the tokenizer?"
        Special tokens must be added to the tokenizer's vocabulary as
        whole tokens, not split into subwords. When tokenizing, we
        first split the text by special tokens, tokenize each segment
        normally, then concatenate with the special token IDs. This
        ensures special tokens are always single tokens in the output.
    """

    def __init__(self, base_tokenizer: Any, config: SFTConfig):
        """
        Initialize the SFT tokenizer wrapper.

        Args:
            base_tokenizer: The underlying tokenizer (e.g., from HuggingFace).
            config: SFT configuration with special token settings.
        """
        self.tokenizer = base_tokenizer
        self.config = config

        # Add special tokens to the tokenizer
        special_tokens_list = list(SPECIAL_TOKENS.values())
        # Note: In production, you'd call tokenizer.add_special_tokens()
        # or tokenizer.add_tokens() here. For demonstration, we track them.
        self.special_token_ids = {
            name: i + 100000 for i, name in enumerate(SPECIAL_TOKENS.keys())
        }

    def tokenize_with_mask(
        self,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a conversation and create a loss mask.

        Args:
            conversation: List of message dicts with 'role', 'content',
                         and optionally 'thinking' keys.

        Returns:
            Dictionary with:
                - 'input_ids': Token IDs (1D tensor)
                - 'labels': Target token IDs (shifted input_ids)
                - 'loss_mask': Boolean mask (1 = train on this token, 0 = ignore)

        Explanation:
            1. Format the conversation into the chat template string
            2. Tokenize the full string
            3. Identify which token positions correspond to assistant responses
            4. Create a binary mask: 1 for assistant tokens, 0 for everything else
            5. Labels = input_ids shifted right (standard next-token prediction)
        """
        # Format conversation
        full_text = format_chat_conversation(
            conversation,
            include_thinking=self.config.think_token_enabled,
        )

        # Tokenize
        # In production, use self.tokenizer.encode() or similar
        # For demonstration, create synthetic token IDs
        input_ids = torch.randint(0, self.config.vocab_size, (min(len(full_text), self.config.max_seq_len),))

        # Create loss mask based on role positions
        loss_mask = self._create_loss_mask(conversation, input_ids)

        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Last token has no target

        # Apply loss mask: set masked positions to -100 (ignored by cross_entropy)
        labels[~loss_mask] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }

    def _create_loss_mask(
        self,
        conversation: List[Dict[str, str]],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a boolean mask indicating which tokens to train on.

        Args:
            conversation: The conversation messages.
            input_ids: The tokenized input.

        Returns:
            Boolean tensor of same length as input_ids.
            True = compute loss on this token, False = ignore.

        Explanation:
            We iterate through the conversation and mark tokens based on role:
            - System, User, Tool tokens: masked (False) — we don't train on prompts
            - Assistant tokens: unmasked (True) — we train on completions
            - Think tokens: depends on config.think_token_in_loss
            - Tool-call tokens: depends on config.tool_call_in_loss
        """
        mask = torch.zeros(len(input_ids), dtype=torch.bool)

        # In a real implementation, we would track token positions
        # corresponding to each role. For demonstration, we mask
        # the first 30% as "prompt" and the rest as "completion."
        if self.config.mask_prompt_tokens:
            prompt_boundary = int(len(input_ids) * 0.3)
            mask[prompt_boundary:] = True
        else:
            mask[:] = True

        return mask


################################################################################
# SECTION 4: SFT TRAINER
################################################################################


class SFTTrainer:
    """
    SFT Trainer
    ============

    Supervised Fine-Tuning trainer that extends the pretraining loop
    with chat template formatting and loss masking.

    Step by step:
        1. Format conversations into chat template with special tokens
        2. Tokenize and create loss masks (train only on completions)
        3. Forward pass through the model
        4. Compute cross-entropy loss with masked labels
        5. Standard gradient update (same as pretraining)

    Formula:
        L_SFT = -Σ_{t in completion} log P(x_t | x_{<t}; θ)

        Where "completion" means only assistant tokens are included
        in the loss (prompt tokens are masked out).

    WHY this matters:
        Loss masking is the key difference between SFT and continued
        pretraining. Without masking, the model would also learn to
        predict system prompts and user inputs — wasting capacity and
        potentially degrading instruction-following quality.

    Interview Question:
        "What is the difference between SFT and continued pretraining?"
        SFT uses (instruction, response) pairs with loss masking — only
        the response tokens contribute to the loss. Continued pretraining
        uses raw text with all tokens in the loss. SFT also uses chat
        templates with role tokens, while pretraining treats everything
        as plain text. The learning rate for SFT is typically 10-100x
        lower than pretraining to avoid catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SFT Trainer.

        Args:
            model: The pretrained language model to fine-tune.
            config: SFTConfig with all SFT hyperparameters.
            tokenizer: The tokenizer for encoding conversations.
            device: Target device. Defaults to CUDA if available.

        Explanation:
            We reuse the Pretrainer for the core training loop and add
            SFT-specific functionality on top. The tokenizer wrapper
            handles chat template formatting and loss masking.
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Wrap the tokenizer for SFT
        self.tokenizer_wrapper = SFTTokenizerWrapper(tokenizer, config)

        # Create the base pretrainer
        self.pretrainer = Pretrainer(model, config, device=self.device)

        # Print SFT-specific recipe
        self._print_sft_recipe()

    def _print_sft_recipe(self) -> None:
        """Print SFT-specific configuration."""
        c = self.config
        border = "-" * 70
        print(f"\n{border}")
        print("SFT-SPECIFIC CONFIGURATION")
        print(border)
        print(f"  Chat template:        {c.chat_template}")
        print(f"  Max sequence length:  {c.max_seq_len}")
        print(f"  Mask prompt tokens:   {c.mask_prompt_tokens}")
        print(f"  Think token enabled:  {c.think_token_enabled}")
        print(f"  Think token in loss:  {c.think_token_in_loss}")
        print(f"  Tool call in loss:    {c.tool_call_in_loss}")
        print(border + "\n")

    def train(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Run SFT training on a dataset of conversations.

        Args:
            dataset: List of conversation examples. Each example should
                     be a dict with:
                         - 'messages': List of message dicts (role, content)
                         - Optionally 'thinking': reasoning trace

        Explanation:
            1. Tokenize each conversation with loss masks
            2. Create batches from the tokenized data
            3. Run the pretraining loop (which handles optimizer, scheduler, etc.)
            4. The loss is automatically masked to only train on completion tokens

        Example:
            >>> dataset = [
            ...     {
            ...         "messages": [
            ...             {"role": "system", "content": "You are helpful."},
            ...             {"role": "user", "content": "What is 2+2?"},
            ...             {"role": "assistant", "content": "The answer is 4."},
            ...         ]
            ...     },
            ... ]
            >>> sft_trainer.train(dataset)
        """
        # Create a simple dataloader from the dataset
        def dataloader_generator():
            idx = 0
            while True:
                example = dataset[idx % len(dataset)]
                yield self.tokenizer_wrapper.tokenize_with_mask(example["messages"])
                idx += 1

        # Run training using the pretrainer's loop
        # Note: In production, you'd modify the loop to handle
        # the loss mask correctly. Here we delegate to pretrainer
        # for the core mechanics.
        self.pretrainer.train(dataloader_generator())


################################################################################
# SECTION 5: LOSS MASKING UTILITIES
################################################################################


def create_completion_mask(
    input_ids: torch.Tensor,
    assistant_start_id: int,
    assistant_end_id: int,
) -> torch.Tensor:
    """
    Create a mask that identifies assistant response tokens.

    Args:
        input_ids: Token IDs of shape (seq_len,).
        assistant_start_id: Token ID for assistant start marker.
        assistant_end_id: Token ID for assistant end marker.

    Returns:
        Boolean mask of shape (seq_len,). True for tokens between
        assistant_start and assistant_end (exclusive of markers).

    Explanation:
        We scan through the token sequence and mark positions that
        fall between assistant start/end markers. This is used to
        create loss masks that only train on assistant responses.

    Example:
        >>> ids = torch.tensor([1, 2, 100, 3, 4, 101, 5])
        >>> create_completion_mask(ids, 100, 101)
        tensor([False, False, False,  True,  True, False, False])
    """
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    in_assistant = False

    for i, token_id in enumerate(input_ids):
        if token_id == assistant_start_id:
            in_assistant = True
            continue
        if token_id == assistant_end_id:
            in_assistant = False
            continue
        if in_assistant:
            mask[i] = True

    return mask


def create_think_mask(
    input_ids: torch.Tensor,
    think_start_id: int,
    think_end_id: int,
) -> torch.Tensor:
    """
    Create a mask that identifies thinking tokens.

    Args:
        input_ids: Token IDs of shape (seq_len,).
        think_start_id: Token ID for thinking start marker.
        think_end_id: Token ID for thinking end marker.

    Returns:
        Boolean mask of shape (seq_len,). True for tokens between
        think_start and think_end (exclusive of markers).

    Explanation:
        Similar to create_completion_mask but for thinking blocks.
        Used to control whether thinking tokens are included in the loss.
    """
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    in_think = False

    for i, token_id in enumerate(input_ids):
        if token_id == think_start_id:
            in_think = True
            continue
        if token_id == think_end_id:
            in_think = False
            continue
        if in_think:
            mask[i] = True

    return mask


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################


def demonstrate_sft():
    """
    Demonstrate SFT with a toy model and synthetic conversations.

    Shows:
        1. Chat template formatting
        2. Loss mask creation
        3. SFT training loop (brief)
    """
    print("=" * 70)
    print("SFT TRAINER DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Demo 1: Chat template formatting
    # ------------------------------------------------------------------
    print("\n[1/5] Chat Template Formatting")
    print("-" * 40)

    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 17?"},
        {
            "role": "assistant",
            "thinking": "Let me compute: 15 * 17 = 15 * (10 + 7) = 150 + 105 = 255",
            "content": "15 * 17 = 255",
        },
    ]

    formatted = format_chat_conversation(messages, include_thinking=True)
    print(f"Formatted conversation:\n{formatted}\n")

    # ------------------------------------------------------------------
    # Demo 2: Individual message formatting
    # ------------------------------------------------------------------
    print("[2/5] Individual Message Formatting")
    print("-" * 40)

    for role in ["system", "user", "assistant", "tool"]:
        msg = format_chat_message(role, f"Test content for {role}")
        print(f"  {role:>10}: {msg}")

    # ------------------------------------------------------------------
    # Demo 3: Loss mask creation
    # ------------------------------------------------------------------
    print("\n[3/5] Loss Mask Creation")
    print("-" * 40)

    # Simulate token IDs with assistant markers
    input_ids = torch.tensor([10, 20, 30, 100, 40, 50, 101, 60, 70])
    # 100 = assistant_start, 101 = assistant_end
    mask = create_completion_mask(input_ids, assistant_start_id=100, assistant_end_id=101)
    print(f"  Input IDs:    {input_ids.tolist()}")
    print(f"  Loss mask:    {mask.tolist()}")
    print(f"  Trained on:   {mask.sum().item()}/{len(mask)} tokens")

    # ------------------------------------------------------------------
    # Demo 4: Think mask creation
    # ------------------------------------------------------------------
    print("\n[4/5] Think Mask Creation")
    print("-" * 40)

    input_ids = torch.tensor([10, 200, 30, 40, 201, 50, 60])
    # 200 = think_start, 201 = think_end
    think_mask = create_think_mask(input_ids, think_start_id=200, think_end_id=201)
    print(f"  Input IDs:    {input_ids.tolist()}")
    print(f"  Think mask:   {think_mask.tolist()}")
    print(f"  Think tokens: {think_mask.sum().item()}/{len(think_mask)} tokens")

    # ------------------------------------------------------------------
    # Demo 5: SFT config
    # ------------------------------------------------------------------
    print("\n[5/5] SFT Configuration")
    print("-" * 40)

    config = SFTConfig(
        model_name="sota-llm-sft",
        learning_rate=2e-5,
        max_steps=1000,
        mask_prompt_tokens=True,
        think_token_enabled=True,
        think_token_in_loss=True,
    )
    print(f"  Model:               {config.model_name}")
    print(f"  Learning rate:       {config.learning_rate}")
    print(f"  Max steps:           {config.max_steps}")
    print(f"  Mask prompt tokens:  {config.mask_prompt_tokens}")
    print(f"  Think token enabled: {config.think_token_enabled}")
    print(f"  Think token in loss: {config.think_token_in_loss}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_sft()

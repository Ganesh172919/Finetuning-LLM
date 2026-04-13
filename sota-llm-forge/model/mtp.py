"""
################################################################################
MULTI-TOKEN PREDICTION (MTP) — DENSER TRAINING AND SPECULATIVE DECODING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Multi-Token Prediction (MTP)?
    Standard language models predict one token ahead: at position i, predict
    token i+1. MTP extends this to predict D tokens ahead at each position:
    at position i, predict tokens i+1, i+2, ..., i+D.

    The predictions are sequential: the prediction at depth d conditions on
    the depth d-1 prediction's embedding. This creates a chain of predictions
    where each depth refines the previous one.

Why does it matter?
    Two major benefits:

    1. Training: Denser gradient signal. Each position contributes D loss
       terms instead of 1, improving sample efficiency. The auxiliary MTP
       losses have configurable weights so the main next-token loss dominates.

    2. Inference: The MTP heads serve as a free draft model for speculative
       decoding. Instead of training a separate small model for drafting,
       you reuse the MTP heads that were trained alongside the main model.
       This can provide 1.5-2x speedup in generation latency.

How does it work?
    At each position i:
        1. Main model predicts token i+1 (standard next-token prediction).
        2. MTP depth 1: Take the embedding of the main model's prediction,
           combine with position i's hidden state, predict token i+2.
        3. MTP depth 2: Take the embedding of depth 1's prediction,
           combine with position i's hidden state, predict token i+3.
        4. ... and so on for D depth levels.

    Each depth has its own small transformer block and projection head,
    but shares the main model's embedding table and vocabulary projection.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │                MULTI-TOKEN PREDICTION                        │
    │                                                              │
    │  Main Model hidden states h_i                                │
    │       │                                                      │
    │       ├──→ LM Head ──→ token i+1 (standard next-token)      │
    │       │                                                      │
    │       ├──→ MTP Depth 1:                                      │
    │       │    Input: [h_i ; embed(token_i+1)]                   │
    │       │    Transformer Block ──→ LM Head ──→ token i+2       │
    │       │                                                      │
    │       └──→ MTP Depth 2:                                      │
    │            Input: [h_i ; embed(token_i+2_from_depth1)]       │
    │            Transformer Block ──→ LM Head ──→ token i+3       │
    │                                                              │
    │  Training loss = main_loss                                   │
    │                + mtp_weight * (mtp_loss_1 + mtp_loss_2)      │
    │                                                              │
    │  Inference: MTP heads as draft model for speculative decoding │
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2024: "Better & Faster Large Language Models via Multi-token Prediction"
             (Gloeckle et al., Meta) — Original MTP paper
    - 2024: DeepSeek-V3 — MTP integrated into production LLM
    - 2025: Speculative decoding with MTP heads — Free draft model

INTERVIEW QUESTIONS:
    1. "How does MTP improve training efficiency?"
       Each position contributes D loss terms instead of 1. The MTP losses
       provide gradient signal about future token distributions, which helps
       the model learn better representations even for the main next-token
       task. The auxiliary losses are weighted down so they guide but don't
       dominate training.

    2. "How are MTP heads used for speculative decoding at inference time?"
       Instead of training a separate draft model (which is expensive and
       may not align with the main model), the MTP heads naturally draft
       the next D tokens. At each step:
       (a) Main model generates token i+1.
       (b) MTP heads speculatively generate tokens i+2, ..., i+D+1.
       (c) Main model verifies all D tokens in one forward pass.
       (d) Accept the longest correct prefix, reject the rest.
       This gives up to D-1x speedup with no additional model training.

    3. "Why do MTP heads share the embedding and output projection?"
       Sharing ensures that MTP predictions are in the same vocabulary
       space as the main model. It also saves parameters and keeps the
       draft model aligned with the main model's token distribution.
       Without sharing, the MTP heads might learn a different mapping
       between positions and tokens, leading to poor draft quality.

################################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


################################################################################
# SECTION 1: MTP CONFIGURATION
################################################################################


@dataclass
class MTPConfig:
    """
    Configuration for Multi-Token Prediction.

    Attributes:
        d_model: Model dimension (shared with main model).
        d_head: Attention head dimension for MTP transformer blocks.
        n_heads: Number of attention heads in MTP blocks.
        n_kv_heads: Number of KV heads for GQA in MTP blocks.
        mtp_depth: Number of prediction depths (D). D=1 means standard
                   next-token only. D=2 means predict 2 tokens ahead.
        mtp_loss_weight: Weight for auxiliary MTP losses in total loss.
        intermediate_dim: FFN intermediate dimension in MTP blocks.
        dropout: Dropout rate.
        vocab_size: Vocabulary size (shared with main model).
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency.
    """

    d_model: int = 768
    d_head: int = 64
    n_heads: int = 12
    n_kv_heads: int = 4
    mtp_depth: int = 2
    mtp_loss_weight: float = 0.1
    intermediate_dim: int = 2048
    dropout: float = 0.0
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rope_base: float = 10000.0


################################################################################
# SECTION 2: MTP HEAD (SINGLE DEPTH LEVEL)
################################################################################


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head — Single Depth Level
    =================================================

    Predicts the token at position (pos + depth + 1) given:
    - The main model's hidden state at position pos.
    - The embedding of the previous depth's predicted token at position (pos + depth).

    Architecture:
        Input = concat(h_main[pos], embed(prev_prediction[pos + depth])) along d_model
        → Linear projection back to d_model
        → Small Transformer block (1 layer)
        → LM Head (shared with main model) → next token logits

    Step by step:
        1. Get main model hidden state h at position pos.
        2. Get embedding of previous depth's predicted token at pos + depth.
        3. Concatenate: input = [h; embed] → (batch, seq_len, 2 * d_model).
        4. Project back to d_model: projected = W_project @ input.
        5. Apply RMSNorm.
        6. Pass through a single-layer transformer block.
        7. Project to vocabulary: logits = W_out @ h_mtp.

    WHY this matters:
        Each MTP head adds a small overhead (one transformer layer + projection)
        but provides dense training signal. The key design choice is using a
        SMALL transformer block rather than re-running the full model — this
        keeps the overhead manageable while still capturing the sequential
        dependency between depth levels.

    Interview Question:
        "Why use a small transformer block in each MTP head instead of just
        a linear projection?"
        A linear projection cannot capture the interaction between the main
        model's hidden state and the previous depth's prediction. The
        transformer block allows the head to perform cross-attention-like
        reasoning: "given what the main model knows about context (h_main)
        and what we predicted last depth (embed), what token comes next?"
        This is more expressive than a simple concatenation + linear.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
    ):
        """
        Initialize the MTP head.

        Args:
            d_model: Model dimension.
            d_head: Head dimension for the MTP transformer block.
            n_heads: Number of attention heads.
            n_kv_heads: Number of KV heads (GQA).
            intermediate_dim: FFN intermediate dimension.
            vocab_size: Vocabulary size.
            dropout: Dropout rate.
            rope_base: RoPE base frequency.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model

        # --- Projection: concat(h, embed) → d_model ---
        # W_project: (2 * d_model, d_model)
        self.W_project = nn.Linear(2 * d_model, d_model, bias=False)

        # --- RMSNorm ---
        self.norm = nn.RMSNorm(d_model)

        # --- Single-layer Transformer block ---
        # We implement a minimal transformer block here (attention + FFN)
        # to keep MTP heads lightweight.

        # Attention: GQA
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        # w_q: (d_model, n_heads * d_head)
        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        # w_k: (d_model, n_kv_heads * d_head)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        # w_v: (d_model, n_kv_heads * d_head)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        # w_o: (n_heads * d_head, d_model)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        # FFN (SwiGLU)
        # w_gate: (d_model, intermediate_dim)
        self.w_gate = nn.Linear(d_model, intermediate_dim, bias=False)
        # w_up: (d_model, intermediate_dim)
        self.w_up = nn.Linear(d_model, intermediate_dim, bias=False)
        # w_down: (intermediate_dim, d_model)
        self.w_down = nn.Linear(intermediate_dim, d_model, bias=False)

        # FFN norm
        self.ffn_norm = nn.RMSNorm(d_model)

        # RoPE (imported from attention module)
        from .attention import RotaryPositionEmbedding
        self.rope = RotaryPositionEmbedding(
            d_head=d_head,
            base=rope_base,
            max_seq_len=max_seq_len,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def _attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple GQA attention for the MTP block.

        Args:
            x: Input, shape (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            Attention output, shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        n_rep = self.n_heads // self.n_kv_heads

        # Project to Q, K, V
        q = self.w_q(x)  # (batch, seq_len, n_heads * d_head)
        k = self.w_k(x)  # (batch, seq_len, n_kv_heads * d_head)
        v = self.w_v(x)  # (batch, seq_len, n_kv_heads * d_head)

        # Reshape
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # q: (batch, n_heads, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        # k: (batch, n_kv_heads, seq_len, d_head)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        # v: (batch, n_kv_heads, seq_len, d_head)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Expand KV heads
        if n_rep > 1:
            k = k[:, :, None, :, :].expand(batch_size, self.n_kv_heads, n_rep, seq_len, self.d_head)
            k = k.reshape(batch_size, self.n_heads, seq_len, self.d_head)
            v = v[:, :, None, :, :].expand(batch_size, self.n_kv_heads, n_rep, seq_len, self.d_head)
            v = v.reshape(batch_size, self.n_heads, seq_len, self.d_head)

        # Attention scores
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # attn_output: (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (batch, seq_len, n_heads * d_head)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Output projection
        return self.w_o(attn_output)

    def forward(
        self,
        h_main: torch.Tensor,
        prev_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single MTP depth level.

        Args:
            h_main: Main model hidden states, shape (batch, seq_len, d_model).
            prev_embed: Embedding of previous depth's prediction,
                       shape (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            Logits over vocabulary, shape (batch, seq_len, vocab_size).

        Explanation:
            1. Concatenate main hidden state with previous depth embedding.
            2. Project back to d_model.
            3. Apply transformer block (attention + FFN with residuals).
            4. Project to vocabulary logits.

            The concatenation + projection is the key: it fuses two sources
            of information (context from main model, prediction from prev depth)
            into a single representation for the next prediction.
        """
        # --- Step 1: Concatenate and project ---
        # combined: (batch, seq_len, 2 * d_model)
        combined = torch.cat([h_main, prev_embed], dim=-1)

        # projected: (batch, seq_len, d_model)
        projected = self.W_project(combined)

        # --- Step 2: Transformer block with residual ---
        # Pre-norm attention
        # normed: (batch, seq_len, d_model)
        normed = self.norm(projected)
        # attn_out: (batch, seq_len, d_model)
        attn_out = self._attention(normed, mask=mask)
        # h: (batch, seq_len, d_model) — residual
        h = projected + attn_out

        # Pre-norm FFN
        # normed: (batch, seq_len, d_model)
        normed = self.ffn_norm(h)
        # SwiGLU FFN
        # gate: (batch, seq_len, intermediate_dim)
        gate = self.w_gate(normed)
        # up: (batch, seq_len, intermediate_dim)
        up = self.w_up(normed)
        # ffn_out: (batch, seq_len, d_model)
        ffn_out = self.w_down(F.silu(gate) * up)
        ffn_out = self.ffn_dropout(ffn_out)
        # h: (batch, seq_len, d_model) — residual
        h = h + ffn_out

        # --- Step 3: Project to vocabulary ---
        # logits: (batch, seq_len, vocab_size) — note: W_out is NOT defined here
        # We need a reference to the shared embedding/output weight.
        # This is handled by MultiTokenPredictionHead which passes it in.
        # For now, we return the hidden state and let the parent handle projection.
        return h


################################################################################
# SECTION 3: MULTI-TOKEN PREDICTION HEAD (FULL ASSEMBLY)
################################################################################


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction Module — Full Assembly
    ==============================================

    Manages D-1 auxiliary MTP heads (depth 1 through D-1) that predict
    tokens at increasing distances from the current position.

    Architecture:
        For depth d (d = 1, ..., D-1):
            1. Get main model hidden state h_main at position i.
            2. Get embedding of MTP head (d-1)'s prediction at position i+d.
               For depth 1, this is the main model's top-1 prediction.
            3. Pass through MTPHead[d] to get hidden state.
            4. Project to vocabulary logits using SHARED output weight.

    Sharing:
        - Embedding table: shared with main model (no separate embedding)
        - Output projection (LM head weight): shared with main model
        - Each MTP head has its own transformer block (not shared)

    Training:
        At each position i, for depth d:
            - Predicted token: argmax(logits_d[i])
            - Target token: ground truth at position i + d + 1
            - Loss: cross_entropy(logits_d[i], target_i)
        Total MTP loss = sum over d of (loss_d / (D-1)) * mtp_weight

    Inference (speculative decoding):
        1. Main model generates token t_1.
        2. MTP head 1 predicts t_2, head 2 predicts t_3, etc.
        3. Main model verifies t_1, t_2, ..., t_D in one forward pass.
        4. Accept longest correct prefix.

    WHY this matters:
        MTP provides both training efficiency (denser signal) and inference
        efficiency (free draft model). The depth parameter D controls the
        trade-off: larger D means more auxiliary signal but more overhead.
        D=2 (predict 2 tokens ahead) is the sweet spot in most experiments.

    Interview Question:
        "During training, how do you handle the shifted targets for MTP losses?"
        For depth d, the target at position i is the ground truth token at
        position i + d + 1. This means the last d+1 positions don't have
        valid targets (we'd need future tokens). We mask these positions
        out of the loss computation. The sequence effectively becomes
        (seq_len - d - 1) valid positions for depth d.
    """

    def __init__(self, config: MTPConfig):
        """
        Initialize the multi-token prediction module.

        Args:
            config: MTPConfig with all hyperparameters.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.mtp_depth = config.mtp_depth
        self.mtp_loss_weight = config.mtp_loss_weight
        self.vocab_size = config.vocab_size

        # --- MTP Heads (one per depth level, depth 1 through D-1) ---
        # mtp_heads: ModuleList of (D-1) MTPHead instances
        self.mtp_heads = nn.ModuleList([
            MTPHead(
                d_model=config.d_model,
                d_head=config.d_head,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                intermediate_dim=config.intermediate_dim,
                vocab_size=config.vocab_size,
                dropout=config.dropout,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.mtp_depth - 1)
        ])

        # --- Shared output projection (LM head) ---
        # This should be tied to the main model's embedding weight.
        # We create it here and the main model should tie it.
        # W_out: (d_model, vocab_size) — transposed for linear
        self.W_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- Per-head output norms ---
        self.head_norms = nn.ModuleList([
            nn.RMSNorm(config.d_model)
            for _ in range(config.mtp_depth - 1)
        ])

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """
        Tie the output projection weight to the main model's embedding.

        Args:
            embedding_weight: The embedding table weight from the main model.
                             Shape (vocab_size, d_model).

        Explanation:
            Weight tying means the output projection uses the transpose of
            the embedding matrix: logits = h @ E^T. This is standard in
            LLMs and saves vocab_size * d_model parameters.
        """
        self.W_out.weight = embedding_weight

    def forward(
        self,
        h_main: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        input_embed_fn: Optional[callable] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for multi-token prediction.

        Args:
            h_main: Main model hidden states, shape (batch, seq_len, d_model).
            target_ids: Ground truth token IDs, shape (batch, seq_len).
                       Required for training (to get embeddings of targets).
            input_embed_fn: Function that maps token IDs to embeddings.
                           Signature: fn(ids) → (batch, seq_len, d_model).
                           This is typically the main model's embedding layer.
            mask: Optional causal mask.

        Returns:
            mtp_loss: Combined MTP loss (scalar tensor). None during inference.
            mtp_logits: List of logits tensors, one per depth level.
                       Each shape (batch, seq_len, vocab_size).

        Explanation:
            Training mode:
                For each depth d, we use the GROUND TRUTH token at position
                i+d as the "previous prediction" embedding. This is called
                "teacher forcing" — during training, we don't use the MTP
                head's own predictions as input (that would be unstable).
                The MTP loss trains each head to predict the correct token.

            Inference mode:
                For each depth d, we use the ACTUAL PREDICTION from depth d-1.
                This creates a chain of speculative predictions.
        """
        batch_size, seq_len, _ = h_main.shape
        device = h_main.device

        mtp_logits_list = []
        mtp_losses = []

        # Determine if we're training or inferring
        is_training = target_ids is not None and input_embed_fn is not None

        # Track the "current prediction" for each depth
        # For training: use ground truth (teacher forcing)
        # For inference: use actual predictions

        if is_training:
            # During training: use ground truth embeddings (teacher forcing)
            # prev_embed: (batch, seq_len, d_model) — embedding of target tokens
            # For depth d, prev_embed at position i is embed(target[i + d])
            # We need to handle the shifting carefully.

            for d, (mtp_head, head_norm) in enumerate(zip(self.mtp_heads, self.head_norms)):
                depth = d + 1  # depth 1, 2, ..., D-1

                # Get embedding of ground truth token at position i + depth
                # This is what the previous depth "predicted" (teacher forced)
                if depth == 1:
                    # Depth 1 uses main model's next-token prediction
                    # During teacher forcing, use ground truth at position i+1
                    # target_shifted: (batch, seq_len - 1) — tokens starting from position 1
                    if seq_len > depth:
                        target_for_embed = target_ids[:, depth:]  # (batch, seq_len - depth)
                        # Pad to full seq_len (the last 'depth' positions have no target)
                        padding = torch.zeros(batch_size, depth, device=device, dtype=target_ids.dtype)
                        target_for_embed_padded = torch.cat([target_for_embed, padding], dim=1)
                        # (batch, seq_len)
                    else:
                        target_for_embed_padded = torch.zeros(batch_size, seq_len, device=device, dtype=target_ids.dtype)

                    prev_embed = input_embed_fn(target_for_embed_padded)
                    # prev_embed: (batch, seq_len, d_model)
                else:
                    # Depth d > 1: uses depth d-1's prediction (during training,
                    # we still use ground truth for stability)
                    if seq_len > depth:
                        target_for_embed = target_ids[:, depth:]
                        padding = torch.zeros(batch_size, depth, device=device, dtype=target_ids.dtype)
                        target_for_embed_padded = torch.cat([target_for_embed, padding], dim=1)
                    else:
                        target_for_embed_padded = torch.zeros(batch_size, seq_len, device=device, dtype=target_ids.dtype)

                    prev_embed = input_embed_fn(target_for_embed_padded)
                    # prev_embed: (batch, seq_len, d_model)

                # Run MTP head
                # h_mtp: (batch, seq_len, d_model)
                h_mtp = mtp_head(h_main, prev_embed, mask=mask)

                # Normalize and project to logits
                # h_mtp_normed: (batch, seq_len, d_model)
                h_mtp_normed = head_norm(h_mtp)
                # logits: (batch, seq_len, vocab_size)
                logits = self.W_out(h_mtp_normed)
                mtp_logits_list.append(logits)

                # Compute loss for this depth
                # Target: ground truth at position i + depth + 1
                if seq_len > depth + 1:
                    # target_d: (batch, seq_len - depth - 1)
                    target_d = target_ids[:, depth + 1:]
                    # logits_d: (batch, seq_len - depth - 1, vocab_size)
                    logits_d = logits[:, :seq_len - depth - 1, :]

                    # Flatten for cross entropy
                    # loss_d: scalar
                    loss_d = F.cross_entropy(
                        logits_d.reshape(-1, self.vocab_size),
                        target_d.reshape(-1),
                        ignore_index=-100,  # Standard ignore index
                    )
                    mtp_losses.append(loss_d)
        else:
            # Inference mode: use actual predictions
            # For depth 1, we need the main model's prediction.
            # We'll use a placeholder; in practice, this is called after
            # the main model's forward pass generates its prediction.

            prev_prediction_ids = None  # Will be set from main model output

            for d, (mtp_head, head_norm) in enumerate(zip(self.mtp_heads, self.head_norms)):
                if prev_prediction_ids is not None:
                    prev_embed = input_embed_fn(prev_prediction_ids)
                else:
                    # Fallback: zero embedding (first depth, no prev prediction available)
                    prev_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

                # Run MTP head
                h_mtp = mtp_head(h_main, prev_embed, mask=mask)
                h_mtp_normed = head_norm(h_mtp)
                logits = self.W_out(h_mtp_normed)
                mtp_logits_list.append(logits)

                # Use argmax prediction as input for next depth
                prev_prediction_ids = logits.argmax(dim=-1)

        # --- Combine MTP losses ---
        if mtp_losses:
            # Average MTP loss across depths, weighted by mtp_loss_weight
            # mtp_loss: scalar
            mtp_loss = torch.stack(mtp_losses).mean() * self.mtp_loss_weight
        else:
            mtp_loss = torch.tensor(0.0, device=device)

        return mtp_loss, mtp_logits_list


################################################################################
# SECTION 4: MTP LOSS COMPUTATION
################################################################################


class MTPLoss(nn.Module):
    """
    Combined Loss for Multi-Token Prediction
    =========================================

    Combines the main next-token prediction loss with auxiliary MTP losses.

    Formula:
        total_loss = main_loss + mtp_weight * mean(mtp_losses)

    Where:
        main_loss = cross_entropy(logits[:, :-1], targets[:, 1:])
        mtp_loss_d = cross_entropy(mtp_logits_d[:, :-(d+1)], targets[:, d+1:])

    Step by step:
        1. Compute main next-token loss (standard cross entropy).
        2. MTP losses are already computed by MultiTokenPredictionHead.
        3. Combine: total = main + mtp_weight * mtp_sum.

    WHY this matters:
        The MTP losses should not dominate training. The main next-token
        loss is the primary objective; MTP losses provide auxiliary gradient
        signal that improves representation learning. The weight (typically
        0.01 to 0.1) controls this balance.

    Interview Question:
        "How do you weight the MTP losses relative to the main loss?"
        The MTP paper uses a weight of 0.1-0.3 for the sum of MTP losses.
        This means MTP contributes ~10-30% of the total gradient magnitude.
        Too high and MTP dominates (main task suffers). Too low and MTP
        provides minimal benefit. The weight is a hyperparameter that should
        be tuned on the base model and transferred via muP.
    """

    def __init__(self, mtp_weight: float = 0.1, ignore_index: int = -100):
        """
        Initialize MTP loss.

        Args:
            mtp_weight: Weight for the MTP auxiliary losses.
            ignore_index: Index to ignore in cross entropy (e.g., padding).
        """
        super().__init__()
        self.mtp_weight = mtp_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        main_logits: torch.Tensor,
        target_ids: torch.Tensor,
        mtp_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined MTP + main loss.

        Args:
            main_logits: Main model logits, shape (batch, seq_len, vocab_size).
            target_ids: Ground truth token IDs, shape (batch, seq_len).
            mtp_loss: Pre-computed MTP loss from MultiTokenPredictionHead (scalar).

        Returns:
            total_loss: Combined loss (scalar).
            main_loss: Main next-token loss (scalar).
            mtp_loss: MTP auxiliary loss (scalar, weighted).

        Explanation:
            Main loss: Standard next-token cross entropy.
            For position i, predict token i+1.
            main_logits[:, :-1] are predictions for positions 0..seq_len-2.
            target_ids[:, 1:] are targets for positions 1..seq_len-1.
        """
        # Main next-token loss
        # Shift: predict token at i+1 from position i
        # main_logits_shifted: (batch, seq_len - 1, vocab_size)
        logits_shifted = main_logits[:, :-1, :].contiguous()
        # target_shifted: (batch, seq_len - 1)
        target_shifted = target_ids[:, 1:].contiguous()

        # main_loss: scalar
        main_loss = F.cross_entropy(
            logits_shifted.reshape(-1, logits_shifted.size(-1)),
            target_shifted.reshape(-1),
            ignore_index=self.ignore_index,
        )

        # Combined loss
        # total_loss: scalar
        total_loss = main_loss + mtp_loss

        return total_loss, main_loss, mtp_loss


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################


def demonstrate_mtp():
    """Demonstrate multi-token prediction."""
    print("=" * 70)
    print("MULTI-TOKEN PREDICTION DEMONSTRATION")
    print("=" * 70)

    device = torch.device("cpu")

    # --- Configuration ---
    config = MTPConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_kv_heads=2,
        mtp_depth=3,
        mtp_loss_weight=0.1,
        intermediate_dim=128,
        vocab_size=1000,
        max_seq_len=128,
    )

    print(f"\nConfiguration:")
    print(f"  d_model:              {config.d_model}")
    print(f"  mtp_depth:            {config.mtp_depth}")
    print(f"  mtp_loss_weight:      {config.mtp_loss_weight}")
    print(f"  vocab_size:           {config.vocab_size}")
    print(f"  Number of MTP heads:  {config.mtp_depth - 1}")

    # --- Create MTP Module ---
    mtp = MultiTokenPredictionHead(config)

    # Count parameters
    total_params = sum(p.numel() for p in mtp.parameters())
    per_head_params = total_params // max(len(mtp.mtp_heads), 1)
    print(f"\nParameter counts:")
    print(f"  Total MTP params:     {total_params:,}")
    print(f"  Params per head:      ~{per_head_params:,}")

    # --- Create dummy inputs ---
    batch_size = 2
    seq_len = 16

    # Main model hidden states (simulated)
    # h_main: (batch, seq_len, d_model)
    h_main = torch.randn(batch_size, seq_len, config.d_model, device=device)

    # Ground truth targets
    # target_ids: (batch, seq_len) — token IDs
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Simple embedding function
    embedding = nn.Embedding(config.vocab_size, config.d_model)

    def embed_fn(ids):
        return embedding(ids)

    # --- Training Mode Forward ---
    print(f"\n--- Training Mode ---")
    mtp_loss, mtp_logits = mtp(
        h_main=h_main,
        target_ids=target_ids,
        input_embed_fn=embed_fn,
    )
    print(f"  MTP loss (weighted):  {mtp_loss.item():.4f}")
    print(f"  Number of MTP logits: {len(mtp_logits)}")
    for d, logits in enumerate(mtp_logits):
        print(f"  Depth {d+1} logits shape:  {logits.shape}")

    # --- Combined Loss ---
    print(f"\n--- Combined Loss ---")
    loss_fn = MTPLoss(mtp_weight=config.mtp_loss_weight)
    main_logits = torch.randn(batch_size, seq_len, config.vocab_size, device=device)
    total_loss, main_loss, weighted_mtp_loss = loss_fn(main_logits, target_ids, mtp_loss)
    print(f"  Main loss:            {main_loss.item():.4f}")
    print(f"  MTP loss (weighted):  {weighted_mtp_loss.item():.4f}")
    print(f"  Total loss:           {total_loss.item():.4f}")
    print(f"  MTP contribution:     {weighted_mtp_loss.item() / total_loss.item() * 100:.1f}%")

    # --- Inference Mode (Speculative Decoding) ---
    print(f"\n--- Inference Mode (Speculative Decoding) ---")
    mtp.eval()
    with torch.no_grad():
        _, mtp_logits_inf = mtp(
            h_main=h_main,
            input_embed_fn=embed_fn,
        )
    print(f"  Number of speculative predictions: {len(mtp_logits_inf)}")
    for d, logits in enumerate(mtp_logits_inf):
        # Get top-1 prediction for first token
        pred = logits[0, 0].argmax().item()
        print(f"  Depth {d+1} top-1 prediction (pos 0): token {pred}")

    # --- Show sequential dependency ---
    print(f"\n--- Sequential Dependency ---")
    print(f"  Main model predicts:    token at pos 1")
    print(f"  MTP depth 1 predicts:   token at pos 2 (using main's pred)")
    print(f"  MTP depth 2 predicts:   token at pos 3 (using depth 1's pred)")
    print(f"  MTP depth 3 predicts:   token at pos 4 (using depth 2's pred)")
    print(f"  Total predictions:      4 tokens per forward pass")

    # --- Weight tying ---
    print(f"\n--- Weight Tying ---")
    print(f"  Before tying: W_out weight shape = {mtp.W_out.weight.shape}")
    print(f"  Embedding weight shape = {embedding.weight.shape}")
    print(f"  Tying weights...")
    mtp.tie_weights(embedding.weight)
    print(f"  After tying: W_out weight is embedding.weight (shared)")

    print("\n" + "=" * 70)
    print("All MTP demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mtp()

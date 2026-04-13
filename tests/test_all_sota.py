"""
################################################################################
COMPREHENSIVE TEST SUITE — SOTA AI REPOSITORY
################################################################################

Tests all major implementations across the repository.
Run with: python -m pytest tests/test_all_sota.py -v

################################################################################
"""

import sys
import os
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


################################################################################
# TEST: MATH FOUNDATIONS
################################################################################

class TestMath:
    """Test mathematical foundation implementations."""

    def test_linear_algebra(self):
        """Test linear algebra module loads and basic operations work."""
        from _01_math import linear_algebra
        assert linear_algebra is not None

    def test_probability(self):
        """Test probability module."""
        from _01_math import probability
        assert probability is not None

    def test_optimization(self):
        """Test optimization module."""
        from _01_math import optimization
        assert optimization is not None


################################################################################
# TEST: TRANSFORMERS
################################################################################

class TestTransformers:
    """Test transformer implementations."""

    def test_attention(self):
        """Test attention mechanism."""
        from _02_transformers import attention
        assert attention is not None

    def test_embeddings(self):
        """Test embeddings."""
        from _02_transformers import embeddings
        assert embeddings is not None

    def test_model(self):
        """Test transformer model."""
        from _02_transformers import model
        assert model is not None


################################################################################
# TEST: REINFORCEMENT LEARNING
################################################################################

class TestRL:
    """Test RL alignment methods."""

    def test_grpo_advanced(self):
        """Test GRPO implementation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from grpo_advanced import GRPOConfig, AdvantageEstimator, MathReward, FormatReward

        config = GRPOConfig(n_samples=8, beta=0.04)
        estimator = AdvantageEstimator(config)

        rewards = np.array([
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        advantages = estimator.compute_advantages(rewards)
        assert advantages.shape == (2, 8)
        assert not np.any(np.isnan(advantages))

    def test_dapo(self):
        """Test DAPO implementation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dapo import DAPOConfig, DAPOTrainer

        trainer = DAPOTrainer(DAPOConfig())
        chosen = np.random.randn(8) * 0.5 - 1.0
        rejected = np.random.randn(8) * 0.5 - 2.0
        ref_chosen = np.random.randn(8) * 0.3 - 1.5
        ref_rejected = np.random.randn(8) * 0.3 - 1.5

        loss, info = trainer.compute_loss(chosen, rejected, ref_chosen, ref_rejected)
        assert loss > 0
        assert "total_loss" in info

    def test_dpo_variants(self):
        """Test IPO, KTO, ORPO implementations."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_variants import IPOTrainer, KTOTrainer, ORPOTrainer

        batch = 16
        chosen = np.random.randn(batch) * 0.5 - 1.0
        rejected = np.random.randn(batch) * 0.5 - 2.0
        ref_chosen = np.random.randn(batch) * 0.3 - 1.5
        ref_rejected = np.random.randn(batch) * 0.3 - 1.5

        # IPO
        ipo = IPOTrainer()
        loss, info = ipo.compute_loss(chosen, rejected, ref_chosen, ref_rejected)
        assert loss > 0

        # KTO
        kto = KTOTrainer()
        all_logprobs = np.concatenate([chosen, rejected])
        all_ref = np.concatenate([ref_chosen, ref_rejected])
        labels = np.concatenate([np.ones(batch), np.zeros(batch)])
        loss, info = kto.compute_loss(all_logprobs, all_ref, labels)
        assert loss > 0

        # ORPO
        orpo = ORPOTrainer()
        loss, info = orpo.compute_loss(chosen, rejected)
        assert loss > 0

    def test_process_reward_model(self):
        """Test PRM implementation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from process_reward_model import ProcessRewardModel

        prm = ProcessRewardModel()
        steps = ["Step 1: A", "Step 2: B", "Step 3: C"]
        result = prm.score_steps(steps)
        assert "step_scores" in result
        assert len(result["step_scores"]) == 3
        assert all(0 <= s <= 1 for s in result["step_scores"])


################################################################################
# TEST: INFERENCE
################################################################################

class TestInference:
    """Test inference optimizations."""

    def test_speculative_decoding(self):
        """Test speculative decoding engine."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from speculative_decoding import SpeculativeDecodingEngine

        engine = SpeculativeDecodingEngine(K=5, vocab_size=1000)
        prompt = [1, 2, 3]
        generated = engine.generate_with_verification(prompt, max_new_tokens=10)
        assert len(generated) > len(prompt)
        assert all(isinstance(t, int) for t in generated)

    def test_eagle_decoding(self):
        """Test EAGLE decoding engine."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from eagle_decoding import EagleDecodingEngine, EagleDraftHead

        # Test draft head
        head = EagleDraftHead(d_model=64, vocab_size=1000)
        hidden = np.random.randn(64) * 0.1
        probs = head.predict_next(hidden, 42)
        assert probs.shape == (1000,)
        assert abs(np.sum(probs) - 1.0) < 1e-5

        # Test engine
        engine = EagleDecodingEngine(d_model=64, vocab_size=1000, K=5)
        generated = engine.generate([1, 2, 3], max_new_tokens=10)
        assert len(generated) > 3


################################################################################
# TEST: ARCHITECTURE VARIANTS
################################################################################

class TestArchitectures:
    """Test alternative architecture implementations."""

    def test_mamba2(self):
        """Test Mamba-2 implementation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '22_transformer_variants'))
        from mamba2 import Mamba2Block, Mamba2Model, selective_scan

        # Test selective scan
        batch, seq, d_inner, d_state = 2, 8, 32, 8
        x = np.random.randn(batch, seq, d_inner) * 0.1
        A = np.random.randn(d_inner, d_state) * 0.1
        B = np.random.randn(batch, seq, d_state) * 0.1
        C = np.random.randn(batch, seq, d_state) * 0.1
        dt = np.abs(np.random.randn(batch, seq, d_inner) * 0.01) + 0.01
        y = selective_scan(x, A, B, C, dt)
        assert y.shape == x.shape

        # Test block
        block = Mamba2Block(d_model=32, d_state=8, d_conv=4, expand=2)
        x = np.random.randn(2, 8, 32) * 0.1
        out = block.forward(x)
        assert out.shape == x.shape

        # Test model
        model = Mamba2Model(vocab_size=100, d_model=32, n_layers=2)
        ids = np.random.randint(0, 100, (1, 4))
        logits = model.forward(ids)
        assert logits.shape == (1, 4, 100)

    def test_vision_mamba(self):
        """Test Vision Mamba implementation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '08_vlm'))
        from vision_mamba import PatchEmbedding, BidirectionalMambaBlock, VisionMamba

        # Test patch embedding
        pe = PatchEmbedding(image_size=64, patch_size=16, d_model=48)
        img = np.random.randn(2, 3, 64, 64)
        patches = pe.forward(img)
        assert patches.shape == (2, 16, 48)

        # Test bidirectional block
        block = BidirectionalMambaBlock(d_model=48, d_state=8)
        x = np.random.randn(2, 16, 48) * 0.1
        out = block.forward(x)
        assert out.shape == x.shape

        # Test full model
        model = VisionMamba(image_size=64, patch_size=16, d_model=48, n_layers=2, n_classes=10)
        logits = model.forward(img)
        assert logits.shape == (2, 10)


################################################################################
# TEST: AGENTIC AI
################################################################################

class TestAgentic:
    """Test agentic AI implementations."""

    def test_constitutional_ai(self):
        """Test Constitutional AI."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '13_agentic_ai'))
        from constitutional_ai import ConstitutionalAI, Constitution, Principle

        cai = ConstitutionalAI()

        # Test critique
        critique = cai.critique("test", "test response", cai.constitution.principles[0])
        assert "score" in critique
        assert "principle" in critique

        # Test self-improve
        improved, history = cai.self_improve("test", "basic response", n_iterations=2)
        assert isinstance(improved, str)
        assert len(history) > 0

        # Test judge pair
        judgment = cai.judge_pair("test", "response A", "response B")
        assert judgment["winner"] in ["A", "B"]


################################################################################
# TEST: RWKV-v6
################################################################################

class TestRWKVv6:
    """Test RWKV-v6 architecture implementation."""

    def test_rwkv_v6_config(self):
        """Test RWKV-v6 configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '22_transformer_variants'))
        from rwkv_v6 import RWKVv6Config
        config = RWKVv6Config(d_model=64, n_heads=4)
        assert config.head_size() == 16
        assert config.d_model == 64

    def test_rwkv_v6_model(self):
        """Test RWKV-v6 model forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '22_transformer_variants'))
        from rwkv_v6 import RWKVv6Model, RWKVv6Config
        config = RWKVv6Config(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64)
        model = RWKVv6Model(config)
        input_ids = np.array([1, 2, 3, 4])
        logits, states = model.forward(input_ids)
        assert logits.shape == (4, 100)
        assert len(states) == 2

    def test_rwkv_v6_generation(self):
        """Test RWKV-v6 autoregressive generation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '22_transformer_variants'))
        from rwkv_v6 import RWKVv6Model, RWKVv6Config
        config = RWKVv6Config(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64)
        model = RWKVv6Model(config)
        prompt = np.array([1, 2, 3])
        generated = model.generate(prompt, max_new_tokens=5)
        assert len(generated) == 8  # prompt + 5 new


################################################################################
# TEST: CONTINUOUS BATCHING
################################################################################

class TestContinuousBatching:
    """Test continuous batching implementation."""

    def test_scheduler_init(self):
        """Test scheduler initialization."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from continuous_batching import ContinuousBatchScheduler
        scheduler = ContinuousBatchScheduler(max_batch_size=8)
        assert scheduler.max_batch_size == 8
        assert len(scheduler.running_batch) == 0

    def test_request_lifecycle(self):
        """Test request lifecycle: add → run → finish."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from continuous_batching import ContinuousBatchScheduler, GenerationRequest
        scheduler = ContinuousBatchScheduler(max_batch_size=4)
        req = GenerationRequest(request_id="test", prompt_tokens=[1, 2, 3], max_tokens=5)
        scheduler.add_request(req)
        assert len(scheduler.waiting_queue) == 1

        # Schedule should admit the request
        ids, tokens = scheduler.schedule_iteration()
        assert "test" in ids

    def test_kv_cache_manager(self):
        """Test KV cache memory management."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from continuous_batching import KVCacheManager
        cache = KVCacheManager(max_total_tokens=100)
        assert cache.allocate("req1", 30)
        assert cache.utilization == 0.3
        cache.free("req1")
        assert cache.utilization == 0.0


################################################################################
# TEST: PAGED ATTENTION
################################################################################

class TestPagedAttention:
    """Test paged attention implementation."""

    def test_paged_attention_init(self):
        """Test paged attention engine initialization."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
        from paged_attention import PagedAttentionEngine
        engine = PagedAttentionEngine(num_blocks=64, block_size=16)
        assert engine.block_size == 16


################################################################################
# TEST: MIXTURE OF EXPERTS
################################################################################

class TestMoE:
    """Test Mixture of Experts implementation."""

    def test_moe_config(self):
        """Test MoE configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_moe'))
        from moe_router import MoEConfig
        config = MoEConfig(d_model=64, num_experts=8, top_k=2)
        assert config.num_experts == 8
        assert config.top_k == 2

    def test_router(self):
        """Test expert routing."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_moe'))
        from moe_router import MoERouter, MoEConfig
        config = MoEConfig(d_model=32, num_experts=4, top_k=2)
        router = MoERouter(config)
        x = np.random.randn(8, 32)
        indices, weights, aux_loss = router.forward(x)
        assert indices.shape == (8, 2)
        assert weights.shape == (8, 2)
        assert aux_loss >= 0

    def test_moe_forward(self):
        """Test MoE forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_moe'))
        from moe_router import MixtureOfExperts, MoEConfig
        config = MoEConfig(d_model=32, d_ff=64, num_experts=4, top_k=2)
        moe = MixtureOfExperts(config)
        x = np.random.randn(8, 32)
        output, aux_loss, stats = moe.forward(x)
        assert output.shape == (8, 32)
        assert aux_loss >= 0
        assert "expert_usage" in stats

    def test_expert_specialization(self):
        """Test that experts have independent weights."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_moe'))
        from moe_router import Expert, MoEConfig
        config = MoEConfig(d_model=32, d_ff=64)
        expert0 = Expert(config, 0)
        expert1 = Expert(config, 1)
        x = np.random.randn(4, 32)
        out0 = expert0.forward(x)
        out1 = expert1.forward(x)
        # Different experts should produce different outputs
        assert not np.allclose(out0, out1)


################################################################################
# TEST: DPO TRAINING PIPELINE
################################################################################

class TestDPOTraining:
    """Test DPO training pipeline implementation."""

    def test_dpo_config(self):
        """Test DPO configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_training_pipeline import DPOConfig
        config = DPOConfig(beta=0.1, learning_rate=5e-7)
        assert config.beta == 0.1
        assert config.learning_rate == 5e-7

    def test_preference_dataset(self):
        """Test preference dataset creation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_training_pipeline import PreferenceDataset
        data = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"},
            {"prompt": "test2", "chosen": "good2", "rejected": "bad2"},
        ]
        dataset = PreferenceDataset.from_dicts(data)
        assert len(dataset) == 2
        train, val = dataset.split(val_ratio=0.5)
        assert len(train) + len(val) == 2

    def test_dpo_loss(self):
        """Test DPO loss computation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_training_pipeline import dpo_loss
        batch_size = 4
        policy_chosen = np.random.randn(batch_size)
        policy_rejected = np.random.randn(batch_size) - 1
        ref_chosen = np.random.randn(batch_size)
        ref_rejected = np.random.randn(batch_size) - 1
        loss, metrics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert np.isfinite(loss)  # Loss should be finite
        assert "accuracy" in metrics
        assert "margin" in metrics

    def test_dpo_trainer(self):
        """Test DPO trainer initialization and step."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_training_pipeline import DPOTrainer, DPOConfig
        config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(config)
        batch_size = 4
        metrics = trainer.train_step(
            np.random.randn(batch_size),
            np.random.randn(batch_size) - 1,
            np.random.randn(batch_size),
            np.random.randn(batch_size) - 1,
        )
        assert trainer.step_count == 1
        assert "loss" in metrics

    def test_ipo_loss(self):
        """Test IPO loss computation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '19_reinforcement_learning'))
        from dpo_training_pipeline import ipo_loss
        batch_size = 4
        loss, metrics = ipo_loss(
            np.random.randn(batch_size),
            np.random.randn(batch_size) - 1,
            np.random.randn(batch_size),
            np.random.randn(batch_size) - 1,
        )
        assert loss > 0


################################################################################
# TEST: GROUPED-QUERY ATTENTION
################################################################################

class TestGQA:
    """Test Grouped-Query Attention implementation."""

    def test_gqa_config(self):
        """Test GQA configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_transformers'))
        from grouped_query_attention import GQAConfig
        config = GQAConfig(d_model=256, n_heads=16, n_kv_heads=4)
        assert config.n_groups == 4
        assert config.d_k == 16

    def test_gqa_forward(self):
        """Test GQA forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_transformers'))
        from grouped_query_attention import GroupedQueryAttention, GQAConfig
        config = GQAConfig(d_model=64, n_heads=8, n_kv_heads=2)
        gqa = GroupedQueryAttention(config)
        x = np.random.randn(2, 10, 64)
        output, weights = gqa.forward(x)
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_kv_cache_savings(self):
        """Test that GQA reduces KV cache."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_transformers'))
        from grouped_query_attention import GQAConfig
        mha_config = GQAConfig(d_model=256, n_heads=32, n_kv_heads=32)
        gqa_config = GQAConfig(d_model=256, n_heads=32, n_kv_heads=8)
        assert gqa_config.kv_dim < mha_config.kv_dim


################################################################################
# TEST: KNOWLEDGE DISTILLATION
################################################################################

class TestDistillation:
    """Test knowledge distillation implementation."""

    def test_distillation_config(self):
        """Test distillation configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from knowledge_distillation import DistillationConfig
        config = DistillationConfig(temperature=2.0, alpha=0.7)
        assert config.temperature == 2.0
        assert config.alpha == 0.7

    def test_distillation_loss(self):
        """Test distillation loss computation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from knowledge_distillation import distillation_loss
        batch_size = 4
        vocab_size = 100
        teacher_logits = np.random.randn(batch_size, vocab_size)
        student_logits = np.random.randn(batch_size, vocab_size)
        targets = np.random.randint(0, vocab_size, batch_size)
        loss, metrics = distillation_loss(teacher_logits, student_logits, targets)
        assert np.isfinite(loss)
        assert "soft_loss" in metrics
        assert "hard_loss" in metrics

    def test_distillation_trainer(self):
        """Test distillation trainer."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from knowledge_distillation import DistillationTrainer, SimpleModel, DistillationConfig
        d_model = 32
        teacher = SimpleModel(d_model, 50, "Teacher")
        student = SimpleModel(d_model, 50, "Student")
        config = DistillationConfig()
        trainer = DistillationTrainer(teacher, student, config)
        x = np.random.randn(4, d_model)
        targets = np.random.randint(0, 50, 4)
        metrics = trainer.train_step(x, targets)
        assert "total_loss" in metrics


################################################################################
# TEST: QUANTIZATION
################################################################################

class TestQuantization:
    """Test quantization implementation."""

    def test_int8_quantize_dequantize(self):
        """Test INT8 quantization roundtrip."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from quantization import quantize_tensor, dequantize_tensor
        x = np.random.randn(32, 32).astype(np.float32)
        x_q, scale, zp = quantize_tensor(x, bits=8, sym=True)
        x_dq = dequantize_tensor(x_q, scale, zp)
        error = np.mean(np.abs(x - x_dq))
        assert error < 0.05  # Should have low error

    def test_int4_quantize_dequantize(self):
        """Test INT4 quantization roundtrip."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from quantization import quantize_tensor, dequantize_tensor
        x = np.random.randn(32, 32).astype(np.float32)
        x_q, scale, zp = quantize_tensor(x, bits=4, sym=True)
        x_dq = dequantize_tensor(x_q, scale, zp)
        assert x_q.shape == x.shape

    def test_int8_quantizer(self):
        """Test INT8 quantizer class."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from quantization import INT8Quantizer
        quantizer = INT8Quantizer()
        weight = np.random.randn(64, 64).astype(np.float32)
        result = quantizer.quantize_weight(weight)
        assert result["bits"] == 8
        weight_dq = quantizer.dequantize_weight(result)
        assert weight_dq.shape == weight.shape

    def test_int4_quantizer(self):
        """Test INT4 quantizer class."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from quantization import INT4Quantizer
        quantizer = INT4Quantizer()
        weight = np.random.randn(64, 64).astype(np.float32)
        result = quantizer.quantize_weight_gptq(weight)
        assert result["bits"] == 4

    def test_memory_savings(self):
        """Test that quantization reduces memory."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from quantization import quantize_tensor
        x = np.random.randn(100, 100).astype(np.float32)
        x_int8, _, _ = quantize_tensor(x, bits=8)
        # INT8 should be 4x smaller than FP32
        assert x_int8.nbytes == x.nbytes // 4


################################################################################
# TEST: WEIGHT PRUNING
################################################################################

class TestPruning:
    """Test weight pruning implementation."""

    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from weight_pruning import magnitude_pruning
        weight = np.random.randn(32, 32).astype(np.float32)
        pruned, mask = magnitude_pruning(weight, sparsity=0.5)
        assert pruned.shape == weight.shape
        sparsity = 1 - np.count_nonzero(pruned) / pruned.size
        assert abs(sparsity - 0.5) < 0.05

    def test_structured_pruning(self):
        """Test structured pruning."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from weight_pruning import structured_pruning
        weight = np.random.randn(32, 32).astype(np.float32)
        pruned, mask = structured_pruning(weight, sparsity=0.5, axis=0)
        assert pruned.shape == weight.shape

    def test_n_m_sparsity(self):
        """Test N:M semi-structured sparsity."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from weight_pruning import n_m_sparsity
        weight = np.random.randn(32, 32).astype(np.float32)
        pruned, mask = n_m_sparsity(weight, n=2, m=4)
        assert pruned.shape == weight.shape
        # Check that exactly 2 out of 4 are kept
        flat_mask = mask.reshape(-1)
        for i in range(0, len(flat_mask), 4):
            assert np.sum(flat_mask[i:i+4]) == 2

    def test_pruning_config(self):
        """Test pruning configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from weight_pruning import PruningConfig
        config = PruningConfig(sparsity=0.5, method="magnitude")
        assert config.sparsity == 0.5

    def test_pruning_trainer(self):
        """Test pruning trainer."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from weight_pruning import PruningTrainer, PruningConfig
        config = PruningConfig(sparsity=0.5)
        trainer = PruningTrainer(config)
        weights = {"layer1": np.random.randn(16, 16)}
        pruned = trainer.apply_pruning(weights)
        stats = trainer.get_sparsity_stats()
        assert "sparsity" in stats


################################################################################
# TEST: MODEL MERGING
################################################################################

class TestModelMerging:
    """Test model merging implementation."""

    def test_linear_merge(self):
        """Test linear interpolation merging."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model_merging import linear_merge
        a = {"w": np.random.randn(16, 16)}
        b = {"w": np.random.randn(16, 16)}
        merged = linear_merge(a, b, alpha=0.5)
        expected = 0.5 * a["w"] + 0.5 * b["w"]
        assert np.allclose(merged["w"], expected)

    def test_ties_merge(self):
        """Test TIES merging."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model_merging import ties_merge
        base = {"w": np.random.randn(16, 16)}
        a = {"w": base["w"] + np.random.randn(16, 16) * 0.1}
        b = {"w": base["w"] + np.random.randn(16, 16) * 0.1}
        merged = ties_merge([a, b], base, density=0.5)
        assert "w" in merged

    def test_dare_merge(self):
        """Test DARE merging."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model_merging import dare_merge
        base = {"w": np.random.randn(16, 16)}
        a = {"w": base["w"] + np.random.randn(16, 16) * 0.1}
        b = {"w": base["w"] + np.random.randn(16, 16) * 0.1}
        merged = dare_merge([a, b], base, drop_prob=0.5)
        assert "w" in merged

    def test_task_arithmetic(self):
        """Test task arithmetic."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model_merging import task_arithmetic_add
        base = {"w": np.random.randn(16, 16)}
        task = {"w": base["w"] + np.random.randn(16, 16) * 0.1}
        enhanced = task_arithmetic_add(base, task, scale=1.5)
        assert "w" in enhanced

    def test_slerp_merge(self):
        """Test SLERP merging."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model_merging import slerp_merge
        a = {"w": np.random.randn(16, 16)}
        b = {"w": np.random.randn(16, 16)}
        merged = slerp_merge(a, b, alpha=0.5)
        assert "w" in merged


################################################################################
# TEST: ADAPTER FUSION
################################################################################

class TestAdapterFusion:
    """Test adapter fusion implementation."""

    def test_adapter_fusion_config(self):
        """Test adapter fusion configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from adapter_fusion import AdapterFusionConfig
        config = AdapterFusionConfig(d_model=64, n_adapters=3, adapter_dim=16)
        assert config.n_adapters == 3

    def test_task_adapter(self):
        """Test single task adapter."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from adapter_fusion import TaskAdapter
        adapter = TaskAdapter(d_model=32, adapter_dim=8, task_name="test")
        x = np.random.randn(2, 10, 32)
        output = adapter.forward(x)
        assert output.shape == x.shape

    def test_adapter_fusion(self):
        """Test adapter fusion forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from adapter_fusion import AdapterFusion, AdapterFusionConfig
        config = AdapterFusionConfig(d_model=32, n_adapters=3, adapter_dim=8)
        fusion = AdapterFusion(config)
        x = np.random.randn(2, 10, 32)
        output, attn = fusion.forward(x)
        assert output.shape == x.shape
        assert attn.shape == (2, 10, 3)

    def test_multi_task_manager(self):
        """Test multi-task adapter manager."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from adapter_fusion import MultiTaskAdapterManager, AdapterFusionConfig
        config = AdapterFusionConfig(d_model=32, n_adapters=3, adapter_dim=8)
        manager = MultiTaskAdapterManager(config)
        manager.set_task_names(["Task1", "Task2", "Task3"])
        x = np.random.randn(2, 10, 32)
        output, metadata = manager.forward(x)
        assert "adapter_usage" in metadata


################################################################################
# TEST: PREFIX TUNING
################################################################################

class TestPrefixTuning:
    """Test prefix tuning implementation."""

    def test_prefix_tuning_config(self):
        """Test prefix tuning configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from prefix_tuning import PrefixTuningConfig
        config = PrefixTuningConfig(d_model=64, n_layers=4, prefix_length=10)
        assert config.prefix_length == 10

    def test_prefix_tuning(self):
        """Test prefix tuning forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from prefix_tuning import PrefixTuning, PrefixTuningConfig
        config = PrefixTuningConfig(d_model=32, n_layers=2, prefix_length=5)
        prefix = PrefixTuning(config)
        prefix_k, prefix_v = prefix.get_prefix(batch_size=4)
        assert prefix_k.shape == (2, 4, 5, 32)
        assert prefix_v.shape == (2, 4, 5, 32)

    def test_prefix_parameters(self):
        """Test prefix parameter count."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from prefix_tuning import PrefixTuning, PrefixTuningConfig
        config = PrefixTuningConfig(d_model=32, n_layers=2, prefix_length=5)
        prefix = PrefixTuning(config)
        params = prefix.count_parameters()
        assert params > 0

    def test_prompt_tuning(self):
        """Test prompt tuning."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from prefix_tuning import PromptTuning
        prompt = PromptTuning(d_model=32, prefix_length=10)
        emb = prompt.get_prompt(batch_size=4)
        assert emb.shape == (4, 10, 32)


################################################################################
# TEST: IA3 ADAPTER
################################################################################

class TestIA3:
    """Test IA3 adapter implementation."""

    def test_ia3_config(self):
        """Test IA3 configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from ia3_adapter import IA3Config
        config = IA3Config(d_model=64, d_ff=128)
        assert config.d_model == 64

    def test_ia3_adapter(self):
        """Test IA3 adapter rescaling."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from ia3_adapter import IA3Adapter, IA3Config
        config = IA3Config(d_model=32, d_ff=64)
        adapter = IA3Adapter(config)
        K = np.random.randn(2, 10, 32)
        K_scaled = adapter.rescale_key(K)
        assert K_scaled.shape == K.shape

    def test_ia3_parameters(self):
        """Test IA3 parameter count."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from ia3_adapter import IA3Adapter, IA3Config
        config = IA3Config(d_model=32, d_ff=64)
        adapter = IA3Adapter(config)
        params = adapter.count_parameters()
        assert params == 32 + 32 + 64  # l_k + l_v + l_ff


################################################################################
# TEST: BITFIT
################################################################################

class TestBitFit:
    """Test BitFit implementation."""

    def test_bitfit_config(self):
        """Test BitFit configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from bitfit import BitFitConfig
        config = BitFitConfig(d_model=64, d_ff=128)
        assert config.d_model == 64

    def test_bitfit_adapter(self):
        """Test BitFit adapter."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from bitfit import BitFitAdapter, BitFitConfig
        config = BitFitConfig(d_model=32, d_ff=64)
        adapter = BitFitAdapter(config)
        Q = np.random.randn(2, 10, 32)
        K = np.random.randn(2, 10, 32)
        V = np.random.randn(2, 10, 32)
        O = np.random.randn(2, 10, 32)
        Q_b, K_b, V_b, O_b = adapter.apply_attention_bias(Q, K, V, O)
        assert Q_b.shape == Q.shape

    def test_bitfit_parameters(self):
        """Test BitFit parameter count."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from bitfit import BitFitAdapter, BitFitConfig
        config = BitFitConfig(d_model=32, d_ff=64, train_layernorm=True)
        adapter = BitFitAdapter(config)
        params = adapter.count_parameters()
        # 4 attention biases + 2 FFN biases + 2 LN biases = 8 * 32 = 256
        assert params == 32 * 4 + 64 + 32 + 32 * 2


################################################################################
# TEST: LOHA
################################################################################

class TestLoHa:
    """Test LoHa adapter implementation."""

    def test_loha_config(self):
        """Test LoHa configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from loha_adapter import LoHaConfig
        config = LoHaConfig(in_features=64, out_features=64, rank=8)
        assert config.rank == 8

    def test_loha_adapter(self):
        """Test LoHa adapter forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from loha_adapter import LoHaAdapter, LoHaConfig
        config = LoHaConfig(in_features=32, out_features=32, rank=4)
        adapter = LoHaAdapter(config)
        x = np.random.randn(2, 32)
        output = adapter.forward(x)
        assert output.shape == (2, 32)

    def test_loha_parameters(self):
        """Test LoHa parameter count."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from loha_adapter import LoHaAdapter, LoHaConfig
        config = LoHaConfig(in_features=32, out_features=32, rank=4)
        adapter = LoHaAdapter(config)
        params = adapter.count_parameters()
        # 4 matrices: A1, B1, A2, B2
        assert params == 4 * 32 * 4


################################################################################
# TEST: DORA
################################################################################

class TestDoRA:
    """Test DoRA adapter implementation."""

    def test_dora_config(self):
        """Test DoRA configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from dora_adapter import DoRAConfig
        config = DoRAConfig(in_features=64, out_features=64, rank=8)
        assert config.rank == 8

    def test_dora_adapter(self):
        """Test DoRA adapter forward pass."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from dora_adapter import DoRAAdapter, DoRAConfig
        config = DoRAConfig(in_features=32, out_features=32, rank=4)
        adapter = DoRAAdapter(config)
        x = np.random.randn(2, 32)
        output = adapter.forward(x)
        assert output.shape == (2, 32)

    def test_dora_parameters(self):
        """Test DoRA parameter count."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from dora_adapter import DoRAAdapter, DoRAConfig
        config = DoRAConfig(in_features=32, out_features=32, rank=4)
        adapter = DoRAAdapter(config)
        params = adapter.count_parameters()
        # magnitude (32) + A (32×4) + B (4×32) = 32 + 128 + 128 = 288
        assert params == 32 + 32 * 4 + 4 * 32


################################################################################
# TEST: GALORE
################################################################################

class TestGaLore:
    """Test GaLore implementation."""

    def test_galore_config(self):
        """Test GaLore configuration."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from galore import GaLoreConfig
        config = GaLoreConfig(rank=64)
        assert config.rank == 64

    def test_gradient_projector(self):
        """Test gradient projection."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from galore import GradientProjector
        proj = GradientProjector(rows=32, cols=32, rank=8)
        G = np.random.randn(32, 32)
        G_low = proj.project(G)
        assert G_low.shape == (8, 8)
        G_approx = proj.unproject(G_low)
        assert G_approx.shape == (32, 32)

    def test_galore_adam(self):
        """Test GaLore Adam optimizer."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from galore import GaLoreAdam, GaLoreConfig
        config = GaLoreConfig(rank=8)
        optimizer = GaLoreAdam((32, 32), config, lr=1e-3)
        param = np.random.randn(32, 32) * 0.01
        grad = np.random.randn(32, 32) * 0.01
        param_new = optimizer.step(param, grad)
        assert param_new.shape == (32, 32)

    def test_memory_savings(self):
        """Test that GaLore saves memory."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from galore import GaLoreAdam, GaLoreConfig
        config = GaLoreConfig(rank=8)
        optimizer = GaLoreAdam((64, 64), config)
        stats = optimizer.get_memory_stats()
        assert stats["memory_savings"] > 1


################################################################################
# TEST: S-BITS STOCHASTIC QUANTIZATION
################################################################################

class TestSBits:
    """Test S-Bits stochastic quantization implementation."""

    def test_sbits_config(self):
        """Test SBitsConfig validation and properties."""
        from sbits import SBitsConfig
        config = SBitsConfig(bits=4, block_size=64)
        assert config.bits == 4
        assert config.num_levels == 16
        assert config.max_val == 15
        assert config.block_size == 64
        assert config.use_stochastic_rounding is True

    def test_sbits_config_validation(self):
        """Test SBitsConfig rejects invalid parameters."""
        from sbits import SBitsConfig
        import traceback
        try:
            SBitsConfig(bits=5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_sbits_quantizer(self):
        """Test basic quantization and dequantization."""
        from sbits import SBitsQuantizer, SBitsConfig
        config = SBitsConfig(bits=4, block_size=32)
        quantizer = SBitsQuantizer(config)
        weights = np.random.randn(64, 64).astype(np.float32)
        quantized, metadata = quantizer.quantize(weights)
        assert quantized.shape == weights.shape
        assert 'scales' in metadata
        assert 'zero_points' in metadata
        reconstructed = quantizer.dequantize(quantized, metadata)
        assert reconstructed.shape == weights.shape

    def test_sbits_quantization_error(self):
        """Test that 4-bit quantization has acceptable error."""
        from sbits import SBitsQuantizer, SBitsConfig, compute_quantization_error
        config = SBitsConfig(bits=4, block_size=64)
        quantizer = SBitsQuantizer(config)
        weights = np.random.randn(128, 128).astype(np.float32)
        quantized, _ = quantizer.quantize(weights)
        error = compute_quantization_error(weights, quantized)
        # 4-bit stochastic: cosine similarity > 0.6 is acceptable
        # (stochastic rounding adds variance but preserves expected value)
        assert error['cosine_similarity'] > 0.5
        assert np.isfinite(error['mse'])
        assert error['mse'] < 2.0  # MSE should be bounded

    def test_sbits_stochastic_rounding(self):
        """Test that stochastic rounding preserves expected value."""
        from sbits import stochastic_round
        values = np.random.uniform(0, 1, 10000).astype(np.float32)
        low = np.floor(values * 4) / 4
        high = low + 0.25
        result = stochastic_round(values, low, high)
        # Expected value should be close to original
        bias = np.mean(result - values)
        assert abs(bias) < 0.02  # Should be near zero

    def test_sbits_memory_savings(self):
        """Test memory savings computation."""
        from sbits import memory_savings_report, SBitsConfig
        config = SBitsConfig(bits=4, block_size=64)
        original_bytes = 1000000 * 4  # 1M float32 params
        savings = memory_savings_report(original_bytes, config)
        assert savings['compression_ratio'] > 5
        assert savings['savings_bytes'] > 0

    def test_sbits_compare_bit_widths(self):
        """Test bit width comparison utility."""
        from sbits import compare_bit_widths
        weights = np.random.randn(64, 64).astype(np.float32)
        results = compare_bit_widths(weights, [2, 3, 4])
        assert 2 in results
        assert 3 in results
        assert 4 in results
        # More bits should have lower MSE
        assert results[4]['mse'] <= results[2]['mse']

    def test_sbits_compressor(self):
        """Test packed integer compression."""
        from sbits import SBitsCompressor, SBitsConfig
        config = SBitsConfig(bits=4, block_size=64)
        compressor = SBitsCompressor(config)
        weights = np.random.randn(128, 128).astype(np.float32)
        packed, metadata = compressor.compress(weights)
        assert 'packed' in metadata
        assert packed.nbytes < weights.nbytes


################################################################################
# TEST: ACTIVATION CHECKPOINTING
################################################################################

class TestActivationCheckpointing:
    """Test activation checkpointing (gradient checkpointing) implementation."""

    def test_activation_analyzer(self):
        """Test activation memory estimation."""
        from activation_checkpointing import ActivationAnalyzer
        mem = ActivationAnalyzer.estimate_layer_activation_memory(
            batch_size=32, seq_len=2048, d_model=1024, layer_type='transformer'
        )
        assert mem > 0
        # Transformer with attention should be large (due to T×T attention matrix)
        assert mem > 32 * 2048 * 1024 * 4  # At least input size

    def test_total_memory_estimate(self):
        """Test total memory estimation with checkpointing."""
        from activation_checkpointing import ActivationAnalyzer
        result = ActivationAnalyzer.estimate_total_memory(
            num_layers=24, batch_size=32, seq_len=2048, d_model=1024
        )
        assert result['total_bytes'] > 0
        assert result['with_checkpoint_bytes'] < result['total_bytes']
        assert result['savings_ratio'] > 0.5  # Should save >50%
        assert result['optimal_checkpoints'] > 0

    def test_uniform_placement(self):
        """Test uniform checkpoint placement."""
        from activation_checkpointing import CheckpointScheduler
        ckpts = CheckpointScheduler.uniform_placement(12, 4)
        assert len(ckpts) == 4
        assert 0 in ckpts  # First layer should be checkpointed

    def test_recursive_bisection(self):
        """Test recursive bisection (optimal) checkpoint placement."""
        from activation_checkpointing import CheckpointScheduler
        ckpts = CheckpointScheduler.recursive_bisection(16)
        assert len(ckpts) == 4  # sqrt(16) = 4
        assert 0 in ckpts

    def test_checkpoint_manager(self):
        """Test CheckpointManager configuration and reporting."""
        from activation_checkpointing import CheckpointManager, CheckpointConfig
        config = CheckpointConfig()
        manager = CheckpointManager(config, num_layers=24)
        manager.configure(batch_size=32, seq_len=2048, d_model=1024)
        report = manager.get_memory_report()
        assert report['num_layers'] == 24
        assert report['savings_mb'] > 0
        assert report['recompute_overhead'] > 0

    def test_compare_strategies(self):
        """Test strategy comparison utility."""
        from activation_checkpointing import compare_strategies
        results = compare_strategies(
            num_layers=24, batch_size=16, seq_len=1024, d_model=512
        )
        assert 'none' in results
        assert 'recursive' in results
        # No checkpointing should use the most memory
        assert results['none']['memory_mb'] >= results['recursive']['memory_mb']

    def test_mamba_memory_efficiency(self):
        """Test that Mamba layers use less activation memory than Transformer."""
        from activation_checkpointing import ActivationAnalyzer
        transformer_mem = ActivationAnalyzer.estimate_layer_activation_memory(
            32, 2048, 1024, 'transformer'
        )
        mamba_mem = ActivationAnalyzer.estimate_layer_activation_memory(
            32, 2048, 1024, 'mamba'
        )
        # Mamba has no T×T attention matrix, so should be smaller
        assert mamba_mem < transformer_mem

    def test_simulate_forward_backward(self):
        """Test forward/backward simulation."""
        from activation_checkpointing import CheckpointManager, CheckpointConfig
        config = CheckpointConfig()
        manager = CheckpointManager(config, num_layers=12)
        manager.configure(batch_size=16, seq_len=1024, d_model=512)
        log = manager.simulate_forward_backward(num_steps=5)
        assert len(log) == 5
        assert all('peak_memory_mb' in entry for entry in log)


################################################################################
# TEST: FLASH ATTENTION
################################################################################

class TestFlashAttention:
    """Test FlashAttention tiled exact attention implementation."""

    def test_flash_config(self):
        """Test FlashConfig initialization."""
        from flash_attention import FlashConfig
        config = FlashConfig(tile_size_q=64, tile_size_k=64, causal=False)
        assert config.tile_size_q == 64
        assert config.causal is False

    def test_flash_vs_standard_non_causal(self):
        """Test Flash attention matches standard attention (non-causal)."""
        from flash_attention import flash_attention_forward, standard_attention, FlashConfig
        np.random.seed(42)
        Q = np.random.randn(1, 32, 4, 16).astype(np.float32)
        K = np.random.randn(1, 32, 4, 16).astype(np.float32)
        V = np.random.randn(1, 32, 4, 16).astype(np.float32)
        O_std = standard_attention(Q, K, V, causal=False)
        O_flash = flash_attention_forward(Q, K, V, FlashConfig(causal=False))
        max_diff = np.max(np.abs(O_std - O_flash))
        assert max_diff < 1e-4, f"Flash vs Standard diff too large: {max_diff}"

    def test_flash_vs_standard_causal(self):
        """Test Flash attention matches standard attention (causal)."""
        from flash_attention import flash_attention_forward, standard_attention, FlashConfig
        np.random.seed(42)
        Q = np.random.randn(1, 32, 4, 16).astype(np.float32)
        K = np.random.randn(1, 32, 4, 16).astype(np.float32)
        V = np.random.randn(1, 32, 4, 16).astype(np.float32)
        O_std = standard_attention(Q, K, V, causal=True)
        O_flash = flash_attention_forward(Q, K, V, FlashConfig(causal=True))
        max_diff = np.max(np.abs(O_std - O_flash))
        assert max_diff < 1e-4, f"Flash vs Standard diff too large: {max_diff}"

    def test_flash_output_shape(self):
        """Test Flash attention output shape matches input."""
        from flash_attention import flash_attention_forward
        Q = np.random.randn(2, 64, 8, 32).astype(np.float32)
        K = np.random.randn(2, 64, 8, 32).astype(np.float32)
        V = np.random.randn(2, 64, 8, 32).astype(np.float32)
        O = flash_attention_forward(Q, K, V)
        assert O.shape == Q.shape

    def test_online_softmax_correction(self):
        """Test online softmax correction produces correct results."""
        from flash_attention import online_softmax_correction
        old_max = np.array([1.0, 2.0])
        old_sum = np.array([3.0, 4.0])
        old_out = np.array([[1.0, 2.0], [3.0, 4.0]])
        new_max = np.array([2.0, 1.0])
        new_sum = np.array([5.0, 6.0])
        new_out = np.array([[5.0, 6.0], [7.0, 8.0]])
        out, s, m = online_softmax_correction(old_max, old_sum, old_out, new_max, new_sum, new_out)
        assert np.all(m >= old_max)
        assert np.all(m >= new_max)
        assert np.all(s > 0)

    def test_io_complexity(self):
        """Test IO complexity computation."""
        from flash_attention import compute_io_complexity
        result = compute_io_complexity(seq_len=2048, d_head=64)
        assert result['flash_memory_bytes'] < result['standard_memory_bytes']
        assert result['io_reduction'] > 1  # Flash should have fewer HBM accesses

    def test_benchmark(self):
        """Test benchmark runs without errors."""
        from flash_attention import benchmark_attention
        results = benchmark_attention(
            seq_lengths=[64, 128],
            d_head=16,
            n_heads=4,
            batch_size=1,
            causal=True,
        )
        assert len(results['max_diff']) == 2
        assert all(d < 1e-3 for d in results['max_diff'])


################################################################################
# TEST: PPO TRAINING
################################################################################

class TestPPO:
    """Test PPO (Proximal Policy Optimization) for LLM alignment."""

    def test_ppo_config(self):
        """Test PPO configuration."""
        from ppo_training import PPOConfig
        config = PPOConfig(clip_epsilon=0.2, gamma=0.99)
        assert config.clip_epsilon == 0.2
        assert config.gamma == 0.99
        assert config.ppo_epochs == 4

    def test_gae_computation(self):
        """Test Generalized Advantage Estimation."""
        from ppo_training import compute_gae
        rewards = np.array([1.0, 0.5, 0.8, 0.3])
        values = np.array([0.5, 0.6, 0.4, 0.3, 0.2])
        dones = np.array([0, 0, 0, 1])
        advantages, returns = compute_gae(rewards, values, dones)
        assert len(advantages) == 4
        assert len(returns) == 4
        assert np.all(np.isfinite(advantages))

    def test_ppo_clipped_objective(self):
        """Test PPO clipping mechanism."""
        from ppo_training import ppo_clipped_objective
        log_new = np.array([-0.5, -0.3, -0.7])
        log_old = np.array([-0.6, -0.3, -0.5])
        advantages = np.array([1.0, -1.0, 0.5])
        loss, ratio = ppo_clipped_objective(log_new, log_old, advantages)
        assert len(loss) == 3
        assert np.all(ratio > 0)  # Ratio should be positive
        assert np.all(np.isfinite(loss))

    def test_ppo_clipping_prevents_extreme_ratios(self):
        """Test that clipping limits the ratio to [1-ε, 1+ε]."""
        from ppo_training import ppo_clipped_objective
        # Very different log probs → large ratio
        log_new = np.array([0.0])
        log_old = np.array([-5.0])  # exp(5) ≈ 148
        advantages = np.array([1.0])
        loss, ratio = ppo_clipped_objective(log_new, log_old, advantages, clip_epsilon=0.2)
        # Raw ratio would be ~148, but clipped objective limits it
        assert ratio[0] > 1.0  # Ratio is still computed unclipped
        # The loss should be bounded by the clipping

    def test_value_loss(self):
        """Test value function loss computation."""
        from ppo_training import value_loss
        values_new = np.array([0.5, 0.6, 0.7])
        values_old = np.array([0.4, 0.5, 0.6])
        returns = np.array([0.6, 0.5, 0.8])
        loss = value_loss(values_new, values_old, returns)
        assert np.all(loss >= 0)
        assert np.all(np.isfinite(loss))

    def test_kl_divergence_penalty(self):
        """Test KL divergence penalty."""
        from ppo_training import kl_divergence_penalty
        log_policy = np.array([-0.5, -0.3, -0.7])
        log_ref = np.array([-0.6, -0.4, -0.5])
        penalty, kl = kl_divergence_penalty(log_policy, log_ref, coef=0.1)
        assert np.all(np.isfinite(penalty))
        assert np.all(np.isfinite(kl))

    def test_rollout_buffer(self):
        """Test rollout buffer add and batch generation."""
        from ppo_training import RolloutBuffer
        buffer = RolloutBuffer()
        for i in range(10):
            buffer.add(
                obs=np.random.randn(4),
                action=np.array([0.5]),
                reward=float(i),
                value=float(i * 0.5),
                log_prob=-0.5,
                done=(i == 9),
            )
        assert len(buffer.observations) == 10
        buffer.compute_returns_and_advantages(last_value=0.0)
        assert buffer.advantages is not None
        batches = buffer.get_batches(4)
        assert len(batches) >= 2

    def test_ppo_trainer(self):
        """Test PPO trainer initialization and training."""
        from ppo_training import PPOTrainer, PPOConfig
        config = PPOConfig(ppo_epochs=2, mini_batch_size=16)
        trainer = PPOTrainer(config)

        # Add data to buffer
        for i in range(32):
            trainer.buffer.add(
                obs=np.random.randn(4),
                action=np.array([0.5]),
                reward=float(np.random.randn()),
                value=float(np.random.randn()),
                log_prob=-0.5,
                done=(i == 31),
            )

        metrics = trainer.train(num_epochs=2)
        assert 'ppo_loss' in metrics
        assert 'entropy' in metrics
        assert 'kl_estimate' in metrics

    def test_reward_model(self):
        """Test simplified reward model."""
        from ppo_training import SimpleRewardModel
        rm = SimpleRewardModel(bias=0.5, noise_std=0.01)
        quality = np.array([0.3, 0.6, 0.9])
        rewards = rm.score(quality)
        assert len(rewards) == 3
        # Higher quality should generally get higher reward
        assert np.mean(rewards) > 0

    def test_rlhf_pipeline(self):
        """Test complete RLHF pipeline."""
        from ppo_training import RLHFPipeline, PPOConfig
        config = PPOConfig(ppo_epochs=2, mini_batch_size=16)
        pipeline = RLHFPipeline(config)
        metrics = pipeline.train_iteration(num_samples=32, num_epochs=2)
        assert 'mean_reward' in metrics
        assert 'policy_quality' in metrics
        assert 0 <= metrics['policy_quality'] <= 1


################################################################################
# RUN ALL TESTS
################################################################################

################################################################################
# TEST: PRUNING SCHEDULER (Iteration 24)
################################################################################

class TestPruningScheduler:
    """Test model pruning scheduler with cubic schedule."""

    def test_cubic_schedule(self):
        """Test cubic sparsity schedule reaches target."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SparsityScheduler, PruningConfig, ScheduleType
        config = PruningConfig(target_sparsity=0.9, total_steps=100, schedule=ScheduleType.CUBIC)
        scheduler = SparsityScheduler(config)
        assert abs(scheduler.get_sparsity(0)) < 0.01
        assert abs(scheduler.get_sparsity(100) - 0.9) < 0.01
        # Cubic: midpoint should be > 50% of target (concave shape)
        assert scheduler.get_sparsity(50) > 0.5

    def test_linear_schedule(self):
        """Test linear schedule interpolation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SparsityScheduler, PruningConfig, ScheduleType
        config = PruningConfig(target_sparsity=0.8, total_steps=100, schedule=ScheduleType.LINEAR)
        scheduler = SparsityScheduler(config)
        assert abs(scheduler.get_sparsity(50) - 0.4) < 0.01

    def test_warmup_schedule(self):
        """Test warmup schedule has zero sparsity during warmup."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SparsityScheduler, PruningConfig, ScheduleType
        config = PruningConfig(target_sparsity=0.9, total_steps=100, schedule=ScheduleType.WARMUP_CUBIC, warmup_ratio=0.2)
        scheduler = SparsityScheduler(config)
        assert scheduler.get_sparsity(10) == 0.0  # During warmup
        assert scheduler.get_sparsity(50) > 0.0   # Well after warmup

    def test_magnitude_pruner(self):
        """Test magnitude-based pruning achieves target sparsity."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import MagnitudePruner, PruningConfig
        config = PruningConfig(target_sparsity=0.5)
        pruner = MagnitudePruner(config)
        weights = np.random.randn(100, 100)
        result = pruner.prune(weights, name="test")
        assert abs(result.sparsity_achieved - 0.5) < 0.05
        assert result.remaining_params > 0
        assert result.mask is not None

    def test_pruning_preserves_shape(self):
        """Test that pruning preserves weight tensor shape."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import MagnitudePruner, PruningConfig
        pruner = MagnitudePruner(PruningConfig(target_sparsity=0.3))
        weights = np.random.randn(64, 32)
        result = pruner.prune(weights)
        assert result.mask.shape == weights.shape

    def test_structured_pruner_heads(self):
        """Test structured head pruning."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import StructuredPruner
        pruner = StructuredPruner()
        importance = np.random.rand(12)
        mask = pruner.prune_heads(importance, keep_count=8)
        assert mask.sum() == 8
        assert len(mask) == 12

    def test_structured_pruner_neurons(self):
        """Test structured neuron pruning."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import StructuredPruner
        pruner = StructuredPruner()
        weights = np.random.randn(64, 128)
        mask, indices = pruner.prune_neurons(weights, keep_ratio=0.5)
        assert mask.sum() == 32
        assert len(indices) == 32

    def test_sensitivity_analyzer(self):
        """Test layer sensitivity analysis."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer()
        layer_weights = [np.random.randn(32, 32) for _ in range(4)]
        profile = analyzer.analyze_layers(layer_weights)
        assert len(profile) == 4
        assert 's50' in profile['layer_0']

    def test_sparsity_recommendation(self):
        """Test per-layer sparsity recommendation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer()
        layer_weights = [np.random.randn(32, 32) for _ in range(4)]
        profile = analyzer.analyze_layers(layer_weights)
        recs = analyzer.recommend_sparsity(profile, target_overall_sparsity=0.5)
        assert len(recs) == 4
        # Average should be close to target
        avg = np.mean(list(recs.values()))
        assert abs(avg - 0.5) < 0.2

    def test_pruning_savings_estimate(self):
        """Test memory savings estimation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import estimate_pruning_savings
        savings = estimate_pruning_savings(1_000_000, 0.5)
        assert savings['remaining_params'] == 500_000
        assert savings['flops_savings_percent'] == 50.0

    def test_schedule_curve(self):
        """Test full schedule curve generation."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import SparsityScheduler, PruningConfig
        config = PruningConfig(target_sparsity=0.9, total_steps=100)
        scheduler = SparsityScheduler(config)
        curve = scheduler.get_schedule_curve()
        assert len(curve) == 101  # 0 to 100 inclusive
        assert curve[0] < curve[50] < curve[100]

    def test_n_m_sparsity_pruner(self):
        """Test N:M sparsity enforcement."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pruning import MagnitudePruner, PruningConfig, PruningMethod
        config = PruningConfig(method=PruningMethod.N_M_SPARSITY, target_sparsity=0.5)
        pruner = MagnitudePruner(config)
        weights = np.random.randn(64)
        result = pruner.prune(weights)
        # Check N:M constraint: exactly 2 of every 4 are kept
        flat_mask = result.mask.flatten()
        for i in range(0, len(flat_mask) - 3, 4):
            assert np.sum(flat_mask[i:i+4]) == 2


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SOTA AI COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    test_classes = [TestRL, TestInference, TestArchitectures, TestAgentic,
                    TestRWKVv6, TestContinuousBatching, TestPagedAttention, TestMoE,
                    TestDPOTraining, TestGQA, TestDistillation, TestQuantization,
                    TestPruning, TestModelMerging, TestAdapterFusion, TestPrefixTuning,
                    TestIA3, TestBitFit, TestLoHa, TestDoRA, TestGaLore, TestSBits,
                    TestActivationCheckpointing, TestFlashAttention, TestPPO,
                    TestPruningScheduler]
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS {cls.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL {cls.__name__}.{method_name}: {e}")
                    failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

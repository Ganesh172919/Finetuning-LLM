"""
################################################################################
SOTA LLM FORGE — TEST CONFIGURATION
################################################################################

Pytest configuration and shared fixtures for the SOTA LLM Forge test suite.

################################################################################
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def device():
    """Default device for tests (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_config():
    """Minimal model config for fast unit tests."""
    from model import TransformerConfig

    return TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        d_head=16,
        intermediate_dim=176,
        max_seq_len=128,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Minimal model instance for fast unit tests."""
    from model import TransformerLM

    model = TransformerLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def sample_tokens():
    """Sample token tensor for testing."""
    return torch.randint(0, 1000, (2, 32))


@pytest.fixture
def sample_targets():
    """Sample target tensor for testing."""
    return torch.randint(0, 1000, (2, 32))

"""
################################################################################
SOTA LLM FORGE — TEST SUITE
################################################################################

Unit and integration tests for the SOTA LLM training stack.

Test coverage:
    - tokenizer/test_tokenizer.py — Tokenizer correctness
    - model/ tests — Architecture shape/gradient checks
    - optim/ tests — Optimizer convergence and orthogonality
    - data/ tests — Packing masks, decontamination
    - train/ tests — Training loop smoke tests
    - eval/ tests — Harness correctness
    - serve/ tests — Quantization, speculative decoding

Run all tests:
    python -m pytest tests/ -v

################################################################################
"""

"""
################################################################################
SECURITY — PROTECTING AI SYSTEMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is AI Security?
    Protecting AI systems from attacks and misuse.

Key Concerns:
    - Prompt injection
    - Data poisoning
    - Model theft
    - Adversarial attacks

Interview Questions:
    Q: "How do you secure an AI system?"
    A: Input validation, rate limiting, authentication,
       monitoring for adversarial inputs.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: SECURITY UTILITIES
################################################################################

class InputValidator:
    """
    Input Validator
    ===============

    Validates inputs to prevent attacks.
    """

    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def validate(self, text: str) -> bool:
        """Validate input text."""
        if len(text) > self.max_length:
            return False
        # Check for suspicious patterns
        suspicious = ['<script>', 'rm -rf', 'DROP TABLE']
        for pattern in suspicious:
            if pattern in text:
                return False
        return True


class RateLimiter:
    """
    Rate Limiter
    ============

    Limits request rate per user.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List] = {}

    def check(self, user_id: str) -> bool:
        """Check if request is allowed."""
        import time
        now = time.time()

        if user_id not in self.requests:
            self.requests[user_id] = []

        # Remove old requests
        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if now - t < self.window_seconds
        ]

        if len(self.requests[user_id]) >= self.max_requests:
            return False

        self.requests[user_id].append(now)
        return True


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_security():
    """Demonstrate security utilities."""
    print("=" * 70)
    print("SECURITY DEMONSTRATION")
    print("=" * 70)

    # Input validation
    print("\n--- Input Validation ---")
    validator = InputValidator(max_length=100)
    print(f"Normal input: {validator.validate('Hello world')}")
    print(f"Too long: {validator.validate('x' * 200)}")
    print(f"Suspicious: {validator.validate('<script>alert(1)</script>')}")

    # Rate limiting
    print("\n--- Rate Limiting ---")
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for i in range(5):
        allowed = limiter.check("user1")
        print(f"Request {i+1}: {'allowed' if allowed else 'blocked'}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_security()

"""
################################################################################
LOAD TESTING — VERIFYING SYSTEM CAPACITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Load Testing?
    Testing system behavior under heavy load.

Key Metrics:
    - Requests per second
    - Latency under load
    - Error rate under load

Interview Questions:
    Q: "How do you load test a model server?"
    A: Send concurrent requests, measure latency and throughput.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: LOAD TESTER
################################################################################

class LoadTester:
    """
    Load Tester
    ===========

    Tests system capacity.
    """

    def __init__(self, n_requests: int = 100, concurrency: int = 10):
        self.n_requests = n_requests
        self.concurrency = concurrency

    def run(self, server) -> Dict:
        """
        Run load test.

        Args:
            server: Server to test

        Returns:
            results: Test results
        """
        latencies = []
        errors = 0

        for _ in range(self.n_requests):
            try:
                # Simulate request
                latency = np.random.exponential(0.1)
                latencies.append(latency)
            except Exception:
                errors += 1

        return {
            'n_requests': self.n_requests,
            'errors': errors,
            'latency_mean': np.mean(latencies),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'throughput': self.n_requests / np.sum(latencies),
        }


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_load_testing():
    """Demonstrate load testing."""
    print("=" * 70)
    print("LOAD TESTING DEMONSTRATION")
    print("=" * 70)

    tester = LoadTester(n_requests=1000, concurrency=10)
    results = tester.run(server=None)

    print(f"Requests: {results['n_requests']}")
    print(f"Errors: {results['errors']}")
    print(f"Latency mean: {results['latency_mean']:.4f}s")
    print(f"Latency p95: {results['latency_p95']:.4f}s")
    print(f"Throughput: {results['throughput']:.0f} req/s")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_load_testing()

"""
################################################################################
OBSERVABILITY — UNDERSTANDING SYSTEM BEHAVIOR
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Observability?
    Understanding system behavior through logs, metrics, and traces.

Key Components:
    - Logging
    - Metrics
    - Tracing

Interview Questions:
    Q: "What is observability?"
    A: Understanding system behavior through logs, metrics, and traces.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: OBSERVABILITY
################################################################################

class ObservabilityStack:
    """
    Observability Stack
    ===================

    Comprehensive observability for AI systems.
    """

    def __init__(self):
        self.logs = []
        self.metrics = {}
        self.traces = []

    def log(self, level: str, message: str):
        """Add a log entry."""
        self.logs.append({'level': level, 'message': message})

    def record_metric(self, name: str, value: float):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def start_trace(self, trace_id: str):
        """Start a trace."""
        self.traces.append({'id': trace_id, 'spans': []})

    def get_summary(self) -> Dict:
        """Get observability summary."""
        return {
            'n_logs': len(self.logs),
            'n_metrics': len(self.metrics),
            'n_traces': len(self.traces),
        }


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_observability():
    """Demonstrate observability."""
    print("=" * 70)
    print("OBSERVABILITY DEMONSTRATION")
    print("=" * 70)

    obs = ObservabilityStack()

    obs.log('INFO', 'Model loaded')
    obs.log('INFO', 'Inference complete')
    obs.record_metric('latency', 0.1)
    obs.record_metric('latency', 0.15)

    summary = obs.get_summary()
    print(f"Summary: {summary}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_observability()

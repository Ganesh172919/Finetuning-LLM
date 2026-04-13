"""
################################################################################
DATA COLLECTION — GATHERING TRAINING DATA
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Collection?
    Gathering training data for AI models.

Sources:
    - Web crawls
    - Books
    - Code repositories
    - Scientific papers

Interview Questions:
    Q: "Where do you get training data?"
    A: Web crawls, books, code, papers. Quality matters more than quantity.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: DATA COLLECTOR
################################################################################

class DataCollector:
    """
    Data Collector
    ==============

    Collects training data from various sources.
    """

    def __init__(self):
        self.data = []

    def collect_from_web(self, urls: List[str]) -> List[str]:
        """Collect data from web."""
        # Simplified
        return [f"Content from {url}" for url in urls]

    def collect_from_files(self, paths: List[str]) -> List[str]:
        """Collect data from files."""
        return [f"Content from {path}" for path in paths]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_data_collection():
    """Demonstrate data collection."""
    print("=" * 70)
    print("DATA COLLECTION DEMONSTRATION")
    print("=" * 70)

    collector = DataCollector()
    web_data = collector.collect_from_web(["https://example.com"])
    file_data = collector.collect_from_files(["data.txt"])
    print(f"Web data: {len(web_data)}")
    print(f"File data: {len(file_data)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_collection()

"""Script để chạy tất cả unit tests."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
from tests.test_models import TestSplitNNModels
from tests.test_data import TestDataModule
from tests.test_federated import TestFederatedModule


def run_all_tests():
    """Run tất cả unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNNModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDataModule))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedModule))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)


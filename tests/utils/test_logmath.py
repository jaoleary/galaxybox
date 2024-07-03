"""Unit tests for the log-math utility functions."""

import unittest

import numpy as np

from galaxybox.utils.logmath import logadd, logsum


class TestLogMath(unittest.TestCase):
    """Test suite for the logmath utility functions."""

    def test_logsum_with_different_bases(self):
        """Test logsum function with different bases."""
        test_cases = [
            ([9, 10, 11], 10, 11.04532),
            ([9, 10, 11], np.e, 11.40761),
        ]

        for input, base, expected in test_cases:
            with self.subTest(base=base):
                result = logsum(input, base=base)
                self.assertAlmostEqual(result, expected, places=5)

    def test_logadd_with_different_bases(self):
        """Test logadd function with different bases."""
        test_cases = [
            ([9, 10, 11], 10, np.array([9.30103, 10.30103, 11.30103])),
            ([9, 10, 11], np.e, np.array([9.69315, 10.69315, 11.69315])),
        ]

        for input, base, expected in test_cases:
            with self.subTest(base=base):
                result = logadd(input, input, base=base)
                np.testing.assert_almost_equal(result, expected, decimal=5)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the cosmological time bins class."""

import unittest

import numpy as np

from galaxybox.data.cosmic_time import CosmicTimeBins


class TestCosmicTimeBins(unittest.TestCase):
    """Test cases for the CosmicTimeBins class."""

    def setUp(self):
        """Set up test cases for CosmicTimeBins."""
        cosmo_args = (0.6777 * 100, 0.3070, 0.6930, 0.0485)
        # Initialize CosmicTimeBins with default parameters
        self.default_cosmic_time_bins = CosmicTimeBins(*cosmo_args)
        # Initialize CosmicTimeBins with a custom max_redshift
        self.custom_cosmic_time_bins = CosmicTimeBins(*cosmo_args, max_redshift=10)

    def test_initialization(self):
        """Test the initialization of CosmicTimeBins with default and custom parameters."""
        # Test default initialization
        self.assertEqual(self.default_cosmic_time_bins.max_redshift, 8)
        # Test custom initialization
        self.assertEqual(self.custom_cosmic_time_bins.max_redshift, 10)

    def test_output_option_validation(self):
        """Test the validation of output options."""
        # Test valid output options
        try:
            self.default_cosmic_time_bins._output_option_validation(["scale", "redshift", "time"])
        except ValueError:
            self.fail("_output_option_validation raised ValueError unexpectedly!")

        # Test invalid output option
        with self.assertRaises(ValueError):
            self.default_cosmic_time_bins._output_option_validation(["invalid_option"])

    def test_output_types_for_methods(self):
        """Test the output types of scale_factor_bins, cosmic_time_bins, redshift_bins methods."""
        methods = [
            self.default_cosmic_time_bins.scale_factor_bins,
            self.default_cosmic_time_bins.cosmic_time_bins,
            self.default_cosmic_time_bins.redshift_bins,
        ]
        expected_output_type = list  # Assuming all methods should return a list for this example

        for method in methods:
            with self.subTest(method=method.__name__):
                result = method(output=["scale", "redshift", "time"])
                self.assertIsInstance(result, expected_output_type)
                # Test that each item in the list is an np.ndarray
                for item in result:
                    self.assertIsInstance(item, np.ndarray)

    def test_scale_factor_bins_output_values(self):
        """Test the output values of the scale_factor_bins method for default db."""
        expected = (
            np.array([0.64263, 1.54809, 4.29235, 7.52487, 10.78573, 13.82053]),  # cosmic time
            np.array([8.0, 4.0, 1.5, 0.66667, 0.25, 0.0]),  # redshift
            np.array([0.11111, 0.2, 0.4, 0.6, 0.8, 1.0]),  # scale
        )
        result = self.default_cosmic_time_bins.scale_factor_bins(
            db=0.2, output=["time", "redshift", "scale"]
        )
        self.assertTrue(all(isinstance(item, np.ndarray) for item in result))
        # Check if the values are approximately equal
        for r, e in zip(result, expected):
            self.assertEqual(len(r), len(e))
            np.testing.assert_almost_equal(r, e, decimal=5)

    def test_cosmic_time_bins_output_values(self):
        """Test the output values of the cosmic_time_bins method for default db."""
        expected = (
            np.array([0.64263, 1.82053, 3.82053, 5.82053, 7.82053, 9.82053, 11.82053, 13.82053]),
            np.array([8.0, 3.48439, 1.71036, 1.01487, 0.61844, 0.35172, 0.15458, 0.0]),
            np.array([0.11111, 0.223, 0.36895, 0.49631, 0.61788, 0.7398, 0.86611, 1.0]),
        )
        result = self.default_cosmic_time_bins.cosmic_time_bins(
            db=2, output=["time", "redshift", "scale"]
        )
        self.assertTrue(all(isinstance(item, np.ndarray) for item in result))
        # Check if the values are approximately equal
        for r, e in zip(result, expected):
            self.assertEqual(len(r), len(e))
            np.testing.assert_almost_equal(r, e, decimal=5)

    def test_redshift_bins_output_values(self):
        """Test the output values of the redshift_bins method for default db."""
        expected = (
            np.array([0.64263, 0.93632, 1.54809, 3.29604, 13.82053]),
            np.array([8, 6, 4, 2, 0]),
            np.array([0.11111, 0.14286, 0.2, 0.33333, 1.0]),
        )
        result = self.default_cosmic_time_bins.redshift_bins(
            db=2, output=["time", "redshift", "scale"]
        )
        self.assertTrue(all(isinstance(item, np.ndarray) for item in result))
        # Check if the values are approximately equal
        for r, e in zip(result, expected):
            self.assertEqual(len(r), len(e))
            np.testing.assert_almost_equal(r, e, decimal=5)


if __name__ == "__main__":
    unittest.main()

"""unit tests for data utility functions."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from galaxybox.data.utils import (
    find_keys_in_string,
    hdf5_to_dict,
    key_alias,
    kwargs_to_filters,
    minmax_kwarg_swap_alias,
)


class TestHDF5ToDict(unittest.TestCase):
    """Unit tests for the hdf5_to_dict function with a real HDF5 file using tempfile."""

    def setUp(self):
        """Create a temporary HDF5 file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        with h5py.File(self.temp_file.name, "w") as f:
            grp = f.create_group("Group")
            grp.create_dataset("Dataset1", data=[1, 2, 3])
            subgrp = grp.create_group("Subgroup")
            subgrp.create_dataset("Dataset2", data=[4, 5, 6])

    def tearDown(self):
        """Remove the temporary HDF5 file after tests."""
        os.remove(self.temp_file.name)

    def test_hdf5_to_dict(self):
        """Test hdf5_to_dict function with a real HDF5 file."""
        with h5py.File(self.temp_file.name, "r") as f:
            result = hdf5_to_dict(f["Group"])
            expected = {
                "Dataset1": np.array([1, 2, 3]),
                "Subgroup": {"Dataset2": np.array([4, 5, 6])},
            }
            # Manually iterate over the dictionary to compare NumPy arrays
            for key in expected:
                if isinstance(expected[key], np.ndarray):
                    np.testing.assert_array_equal(result[key], expected[key])
                else:  # Assuming nested dictionaries
                    for nested_key in expected[key]:
                        np.testing.assert_array_equal(
                            result[key][nested_key], expected[key][nested_key]
                        )


class TestFindKeysInString(unittest.TestCase):
    """Test suite for the find_keys_in_string function.

    This class contains tests that verify the functionality of the find_keys_in_string function,
    ensuring it correctly identifies and returns keys from a dictionary that are present in a given
    string.
    """

    def test_find_keys_in_string(self):
        """Test find_keys_in_string with a dictionary and a string."""
        dictionary = {"apple": 1, "banana": 2, "cherry": 3}
        string = "I have an apple and a cherry."
        expected = ["apple", "cherry"]
        result = find_keys_in_string(dictionary, string)
        self.assertEqual(result, expected)


class TestKwargsToFilters(unittest.TestCase):
    """Test suite for the kwargs_to_filters function.

    This class contains tests that verify the functionality of the kwargs_to_filters function,
    ensuring it correctly converts keyword arguments into a list of filter tuples suitable for
    database queries or similar filtering operations.
    """

    def test_kwargs_to_filters(self):
        """Test kwargs_to_filters with various kwargs."""
        kwargs = {
            "min_age": 18,
            "max_age": 30,
            "name": "John",
            "interests": ["reading", "swimming"],
        }
        expected = [
            ("age", ">=", 18),
            ("age", "<", 30),
            ("name", "=", "John"),
            ("interests", "in", ["reading", "swimming"]),
        ]
        result = kwargs_to_filters(kwargs)
        self.assertEqual(result, expected)


class TestKeyAlias(unittest.TestCase):
    """Unit tests for the key_alias function."""

    def setUp(self):
        """Initialize test case with a predefined alias dictionary."""
        self.alias_dict = {
            "Python": ["py", "python3", "Python3"],
            "JavaScript": ["js", "node", "NodeJS"],
            "Java": ["java", "Java8", "JDK"],
        }

    def test_direct_match(self):
        """Test that a direct match returns the correct primary key."""
        self.assertEqual(key_alias("Python", self.alias_dict), "Python")

    def test_case_insensitive_match(self):
        """Test that a case-insensitive match returns the correct primary key."""
        self.assertEqual(key_alias("python", self.alias_dict), "Python")

    def test_alias_match(self):
        """Test that an alias match returns the correct primary key."""
        self.assertEqual(key_alias("py", self.alias_dict), "Python")

    def test_no_match_raises_key_error(self):
        """Test that a non-existent key raises a KeyError."""
        with self.assertRaises(KeyError):
            key_alias("C++", self.alias_dict)


class TestMinmaxKwargSwapAlias(unittest.TestCase):
    """Unit tests for the minmax_kwarg_swap_alias function."""

    def test_alias_replacement(self):
        """Test if the function correctly replaces keys with their aliases."""
        kwargs = {"min_age": 18, "max_height": 200, "name": "John Doe"}
        alias_dict = {
            "age": ["min_age", "max_age"],
            "height": ["min_height", "max_height"],
            "name": ["name"],
        }
        expected = {"min_age": 18, "max_height": 200, "name": "John Doe"}
        result = minmax_kwarg_swap_alias(kwargs, alias_dict)
        self.assertEqual(result, expected)

    def test_preserves_modifiers(self):
        """Test if the function preserves 'min' and 'max' modifiers."""
        kwargs = {"min_salary": 50000, "max_salary": 100000, "location": "New York"}
        alias_dict = {"salary": ["min_salary", "max_salary"], "location": ["location"]}
        expected = {"min_salary": 50000, "max_salary": 100000, "location": "New York"}
        result = minmax_kwarg_swap_alias(kwargs, alias_dict)
        self.assertEqual(result, expected)

    def test_handles_nonexistent_aliases(self):
        """Test how the function handles keys without aliases."""
        kwargs = {"min_experience": 2, "max_experience": 5, "education": "Bachelor"}
        alias_dict = {
            "experience": ["min_experience", "max_experience"],
            "education": ["education"],
        }
        # Expect the function to leave the kwargs unchanged as there's no direct alias mapping
        expected = kwargs
        result = minmax_kwarg_swap_alias(kwargs, alias_dict)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

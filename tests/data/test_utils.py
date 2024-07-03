"""unit tests for data utility functions."""

import unittest

from galaxybox.data.utils import find_keys_in_string, kwargs_to_filters


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


if __name__ == "__main__":
    unittest.main()

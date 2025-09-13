"""unit tests for data utility functions."""

import os
import tempfile

import h5py
import numpy as np
import pytest

from galaxybox.data.utils import (
    find_keys_in_string,
    hdf5_to_dict,
    key_alias,
    kwarg_parser,
    kwargs_to_filters,
    minmax_kwarg_swap_alias,
)


@pytest.fixture
def temp_hdf5_file():
    """Fixture to create and clean up a temporary HDF5 file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    with h5py.File(temp_file.name, "w") as f:
        grp = f.create_group("Group")
        grp.create_dataset("Dataset1", data=[1, 2, 3])
        subgrp = grp.create_group("Subgroup")
        subgrp.create_dataset("Dataset2", data=[4, 5, 6])
    yield temp_file.name
    os.remove(temp_file.name)


def test_hdf5_to_dict(temp_hdf5_file):
    """Test hdf5_to_dict function with a real HDF5 file."""
    with h5py.File(temp_hdf5_file, "r") as f:
        result = hdf5_to_dict(f["Group"])
        expected = {
            "Dataset1": np.array([1, 2, 3]),
            "Subgroup": {"Dataset2": np.array([4, 5, 6])},
        }
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected[key])
            else:
                for nested_key in expected[key]:
                    np.testing.assert_array_equal(
                        result[key][nested_key], expected[key][nested_key]
                    )


def test_find_keys_in_string():
    dictionary = {"apple": 1, "banana": 2, "cherry": 3}
    string = "I have an apple and a cherry."
    expected = ["apple", "cherry"]
    result = find_keys_in_string(dictionary, string)
    assert result == expected


def test_kwargs_to_filters():
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
    assert result == expected


@pytest.fixture
def alias_dict():
    return {
        "Python": ["py", "python3", "Python3"],
        "JavaScript": ["js", "node", "NodeJS"],
        "Java": ["java", "Java8", "JDK"],
    }


def test_key_alias_direct_match(alias_dict):
    assert key_alias("Python", alias_dict) == "Python"


def test_key_alias_case_insensitive(alias_dict):
    assert key_alias("python", alias_dict) == "Python"


def test_key_alias_alias_match(alias_dict):
    assert key_alias("py", alias_dict) == "Python"


def test_key_alias_no_match_raises(alias_dict):
    with pytest.raises(KeyError):
        key_alias("C++", alias_dict)


def test_minmax_kwarg_swap_alias_replacement():
    kwargs = {"min_age": 18, "max_height": 200, "name": "John Doe"}
    alias_dict = {
        "age": ["min_age", "max_age"],
        "height": ["min_height", "max_height"],
        "name": ["name"],
    }
    expected = {"min_age": 18, "max_height": 200, "name": "John Doe"}
    result = minmax_kwarg_swap_alias(kwargs, alias_dict)
    assert result == expected


def test_minmax_kwarg_swap_alias_preserves_modifiers():
    kwargs = {"min_salary": 50000, "max_salary": 100000, "location": "New York"}
    alias_dict = {"salary": ["min_salary", "max_salary"], "location": ["location"]}
    expected = {"min_salary": 50000, "max_salary": 100000, "location": "New York"}
    result = minmax_kwarg_swap_alias(kwargs, alias_dict)
    assert result == expected


def test_minmax_kwarg_swap_alias_handles_nonexistent_aliases():
    kwargs = {"min_experience": 2, "max_experience": 5, "education": "Bachelor"}
    alias_dict = {
        "experience": ["min_experience", "max_experience"],
        "education": ["education"],
    }
    expected = kwargs
    result = minmax_kwarg_swap_alias(kwargs, alias_dict)
    assert result == expected


def _dummy_func(a, b, c=3):
    return a + b + c


def _dummy_func_with_kwargs(a, **kwargs):
    return a + kwargs.get("b", 0)


def test_kwarg_parser_basic():
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    func_kwargs, remaining = kwarg_parser(_dummy_func, drop=False, **kwargs.copy())
    assert func_kwargs == {"a": 1, "b": 2, "c": 3}
    assert remaining == kwargs


def test_kwarg_parser_drop_true():
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    func_kwargs, remaining = kwarg_parser(_dummy_func, drop=True, **kwargs.copy())
    assert func_kwargs == {"a": 1, "b": 2, "c": 3}
    assert remaining == {"d": 4}


def test_kwarg_parser_no_matching_kwargs():
    kwargs = {"x": 10, "y": 20}
    func_kwargs, remaining = kwarg_parser(_dummy_func, drop=True, **kwargs.copy())
    assert func_kwargs == {}
    assert remaining == {"x": 10, "y": 20}


def test_kwarg_parser_partial_match():
    kwargs = {"a": 5, "z": 99}
    func_kwargs, remaining = kwarg_parser(_dummy_func, drop=True, **kwargs.copy())
    assert func_kwargs == {"a": 5}
    assert remaining == {"z": 99}


def test_kwarg_parser_func_with_kwargs():
    kwargs = {"a": 1, "b": 2, "c": 3}
    func_kwargs, remaining = kwarg_parser(_dummy_func_with_kwargs, drop=True, **kwargs.copy())
    # Only 'a' is a named arg but dummy_func_with_kwargs accepts **kwargs, so only 'a' is picked
    assert func_kwargs == {"a": 1}
    assert remaining == {"b": 2, "c": 3}


def test_kwarg_parser_empty_kwargs():
    func_kwargs, remaining = kwarg_parser(_dummy_func, drop=True)
    assert func_kwargs == {}
    assert remaining == {}

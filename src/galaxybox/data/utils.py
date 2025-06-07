"""Module containing general input/output functions."""

import inspect
import re
from typing import Any

import h5py
import numpy as np


def hdf5_to_dict(group: h5py.Group) -> dict:
    """Recursively unpack an HDF5 file into a dictionary.

    This function takes an HDF5 group object and recursively unpacks it into a dictionary.
    It iterates over the keys in the group and checks if each key corresponds to a dataset or a
    subgroup. If it is a dataset, the corresponding value is extracted and stored in the dictionary.
    If it is a subgroup, the function calls itself recursively to unpack the subgroup.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group object to be unpacked.

    Returns
    -------
    dict
        A dictionary containing the data from the HDF5 group, including subgroups.

    """
    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Dataset):
            data[key] = group[key][()]
        else:
            data[key] = hdf5_to_dict(group[key])
    return data


def find_keys_in_string(dictionary: dict[str, Any], string: str) -> list[str]:
    """Find keys from a dictionary that appear in a string.

    This function takes a dictionary and a string as input.
    It searches for keys from the dictionary that appear as whole words in the string.
    The function uses regular expressions to perform the search.

    Parameters
    ----------
    dictionary : dict[str, Any]
        The dictionary containing the keys to search for.
    string : str
        The string in which to search for the keys.

    Returns
    -------
    list[str]
        A list of keys from the dictionary that appear in the string.

    """
    return [key for key in dictionary.keys() if re.search(re.escape(key), string)]


def kwargs_to_filters(kwargs: dict[str, Any]):
    """Convert keyword arguments to a list of filters.

    This method takes a dictionary of keyword arguments and converts it into a list of filters.
    Each filter is a tuple of the form (column, operator, value), where 'column' is the name of
    a column, 'operator' is a comparison operator (">=", "<", or "in"), and 'value' is the value
    to compare against.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments, where each key is a column name with a possible 'min'
        or 'max' prefix or suffix, and each value is the value to compare against.
    columns : list[str]
        A list of column names to consider when creating the filters. Only keys in `kwargs` that
        match these column names will be included in the filters.


    Returns
    -------
    list
        A list of filters, where each filter is a tuple of the form (column, operator, value).

    """
    filters = []
    for key, value in kwargs.items():
        if "min" in key.lower():
            key = re.sub(r"(^min_|^max_|_min$|_max$)", "", key)
            filters.append((key, ">=", value))
        elif "max" in key.lower():
            key = re.sub(r"(^min_|^max_|_min$|_max$)", "", key)
            filters.append((key, "<", value))
        else:
            values = np.atleast_1d(value).tolist()
            if len(values) == 1:
                filters.append((key, "=", values[0]))
            else:
                filters.append((key, "in", values))
    filters = None if len(filters) == 0 else filters
    return filters


def key_alias(key: str, alias_dict: dict[str, list[str]]) -> str:
    """Find the primary key name for a given alias from a dictionary of aliases.

    This function searches through a dictionary where each key is a primary name
    and its value is a list of aliases. It returns the primary key name for the
    given alias. If the alias matches directly or case-insensitively with a primary
    key or any of its aliases, that primary key is returned. If no match is found,
    a KeyError is raised.

    Parameters
    ----------
    key : str
        The alias or primary key name to search for.
    alias_dict : dict[str, list[str]]
        A dictionary where each key is a primary name and its value is a list of aliases.

    Returns
    -------
    str
        The primary key name corresponding to the given alias.

    """
    for k in alias_dict.keys():
        if (key.lower() in alias_dict[k]) or (key.lower() == k.lower()):
            return k
    raise KeyError(f"`{key}` has no known alias.")


def minmax_kwarg_swap_alias(kwargs, alias_dict: dict[str, list[str]]):
    """Alias keyword argument keys based on a provided dictionary.

    This method updates the keys in the `kwargs` dictionary to their aliases as defined in
    `alias_dict`. It handles keys with 'min' or 'max' prefixes or suffixes by preserving these
    modifiers while replacing the base key with its alias.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments. Each key is a column name potentially prefixed or
        suffixed with 'min' or 'max', indicating a range query. Each value is the corresponding
        value for the query.
    alias_dict : dict[str, str]
        A dictionary mapping original column names to their aliases. Only the base column names
        are included, without any 'min' or 'max' modifiers.

    Returns
    -------
    dict
        A new dictionary with the keys replaced by their aliases, preserving any 'min' or 'max'
        modifiers.

    """
    keys = list(kwargs.keys())
    for kw in keys:
        key = re.sub(r"(^min_|^max_|_min$|_max$)", "", kw)
        new_key = kw.replace(key, key_alias(key.lower(), alias_dict))
        kwargs[new_key] = kwargs.pop(kw)
    return kwargs


def kwarg_parser(func, drop=False, **kwargs):
    """From a dict of key word agruments, seperate those relevant to input function.

    Parameters
    ----------
    func : function
        The function for which kwargs should be extracted.
    drop : bool
        if True kwargs for func will be dropped from the orginal list (the default is False).
    **kwargs : dict
        keyword arguments.

    Returns
    -------
    func_kwargs : dict
        kwargs only associated with input function.
    kwargs : dict
        original or modifed dict of kwargs.

    """
    arg_names = inspect.getfullargspec(func)[0]
    func_kwargs = {}
    for i, k in enumerate(arg_names):
        if k in kwargs:
            func_kwargs[k] = kwargs[k]
            if drop:
                kwargs.pop(k)
    return func_kwargs, kwargs

"""Module containing general input/output functions."""

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


def kwargs_to_filters(kwargs: dict[str, Any], columns: list[str]):
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
    for key in columns:
        for kw in kwargs.keys():
            if ("obs" in kw.lower()) & ("obs" not in key.lower()):
                pass
            elif key.lower() in kw.lower():
                if "min" in kw.lower():
                    filters.append((key, ">=", kwargs[kw]))
                elif "max" in kw.lower():
                    filters.append((key, "<", kwargs[kw]))
                else:
                    values = np.atleast_1d(kwargs[kw]).tolist()
                    filters.append((key, "in", values))
    return filters

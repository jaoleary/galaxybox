"""module contains the definition of the ProtoTree classes."""

import re
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from galaxybox.data.utils import kwargs_to_filters


class ProtoTree(ABC):
    """Abstract base class for a tree."""

    @abstractmethod
    def tree(self, *args, **kwargs) -> pd.DataFrame:
        """Abstract method to define a tree."""
        raise NotImplementedError

    @abstractmethod
    def branch(self, *args, **kwargs) -> pd.DataFrame:
        """Abstract method to define a branch."""
        raise NotImplementedError

    @abstractmethod
    def list(self, columns=None, **kwargs) -> pd.DataFrame:
        """Abstract method to list galaxies with specified properties."""
        raise NotImplementedError

    @abstractmethod
    def alias(self, key: str) -> str:
        """Abstract method to return proper column key for input alias key."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query, columns) -> List[pd.DataFrame]:
        """Abstract method to query the tree with a given query and columns."""
        raise NotImplementedError


class ProtoGalaxyTree(ProtoTree):
    """Class representing a prototype galaxy tree.

    Parameters
    ----------
    filepath : str
        The path to the file containing the galaxy tree data.
    alias_path : Optional[str], default=None
        The path to the file containing the alias data. If None, no alias data will be used.
    unittime_in_yr : float, default=1.0e9
        The unit of time, in years, used in the galaxy tree data.

    """

    def __init__(
        self, filepath: str, alias_path: Optional[str] = None, unittime_in_yr: float = 1.0e9
    ) -> None:
        """Initialize the ProtoGalaxyTree."""
        self.filepath = filepath
        self.alias_path = alias_path
        self.unittime_in_yr = unittime_in_yr

        if self.alias_path is not None:
            with open(self.alias_path, "r") as f:
                self.col_alias = yaml.safe_load(f)

    def alias(self, key: str) -> str:
        """Return proper column key for input alias key.

        Parameters
        ----------
        key : str
            A string alias for a galaxy tree column

        Returns
        -------
        str
            The proper column name for the input alias

        """
        for k in self.columns:
            if k in self.col_alias.keys():
                self.col_alias[k].append(k.lower())
            else:
                self.col_alias[k] = [k.lower()]

        for k in self.col_alias.keys():
            if key.lower() in self.col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))

    def _df_query(
        self, query: list[tuple[str, str, str]], columns: list[str]
    ) -> list[pd.DataFrame]:
        """Execute a query on a pandas dataframe.

        Parameters
        ----------
        query : list of tuple
            The query conditions.
        columns : list of str
            The columns to return.

        Returns
        -------
        list of pandas.DataFrame
            The query results.

        """
        query = " & ".join([" ".join(map(str, tup)) for tup in query])
        return [self._loader(fp).query(query)[columns] for fp in self.filepath]

    def _parquet_query(
        self, query: list[tuple[str, str, str]], columns: list[str]
    ) -> list[pd.DataFrame]:
        """Query the Parquet data.

        Parameters
        ----------
        query : list of tuple
            The query conditions.
        columns : list of str
            The columns to return.

        Returns
        -------
        list of pandas.DataFrame
            The query results.

        """
        return [self._loader(fp, filters=query, columns=columns) for fp in self.filepath]

    def kwarg_swap_alias(self, kwargs):
        """Replace the keys in the kwargs dictionary with their aliases.

        This method takes a dictionary of keyword arguments and replaces each key with its alias,
        if it has one. The keys are expected to be column names with a possible 'min' or 'max'
        prefix or suffix. The aliases are determined by the `alias` method of the class.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments, where each key is a column name with a possible 'min'
            or 'max' prefix or suffix, and each value is the value to compare against.

        Returns
        -------
        dict
            The input dictionary with the keys replaced by their aliases.

        """
        keys = list(kwargs.keys())
        for kw in keys:
            key = re.sub(r"(^min_|^max_|_min$|_max$)", "", kw.lower())
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)
        return kwargs

    def list(self, columns=None, **kwargs) -> pd.DataFrame:
        """Return list of galaxies with specified properties.

        This function selects galaxies based on a flexible set of arguments. Any column of the
        galaxy.trees attribute can have a `min` or `max` added as a prefix or suffix to the column
        name or alias of the column name and passed as an argument to this function. This can also
        be used to select galaxies based on derived properties such as color.

        Parameters
        ----------
        columns : list, optional
            List of columns to include in the returned DataFrame, by default None
        kwargs : dict
            Keyword arguments to filter the galaxies based on column values

        Returns
        -------
        pandas.DataFrame
            A masked subset of the galaxy.trees attribute

        Examples
        --------
        # Select galaxies with redshift between 0.5 and 1.0
        galaxies = proto_tree.list(redshift_min=0.5, redshift_max=1.0)

        # Select galaxies with stellar mass greater than 1e10
        galaxies = proto_tree.list(Stellar_mass_min=1e10)

        # Select galaxies with color_obs equal to 'blue' or 'red'
        galaxies = proto_tree.list(color_obs=['blue', 'red'])

        """
        if columns is None:
            columns = self.columns
        # First clean up kwargs, replace aliases
        kwargs = self.kwarg_swap_alias(kwargs)

        filters = kwargs_to_filters(kwargs, self.columns)

        return self.query(query=filters, columns=columns)

    @cached_property
    def scales(self):
        """Find all scalefactors in the galaxy tree."""
        return np.unique(self.query(query=None, columns=[self.time_column]).values)

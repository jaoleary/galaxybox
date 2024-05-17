"""Defines the EmergeGalaxyTrees class, a subclass of ProtoGalaxyTree."""

import os
from functools import partial
from typing import Sequence, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import interp1d

from galaxybox.data.trees.proto_tree import ProtoGalaxyTree

ALIAS_PATH = os.path.join(os.path.dirname(__file__), "../../configs/emerge-galaxy.alias.yaml")


class EmergeGalaxyTrees(ProtoGalaxyTree):
    """A subclass of ProtoGalaxyTree for handling galaxy tree data stored in Parquet files.

    This class provides methods for querying the data and retrieving individual trees. It uses the
    pyarrow library for efficient reading and writing of Parquet files.


    Parameters
    ----------
    tree_format : str, optional
        The format of the tree files, by default "parquet"
    pre_load : bool, optional
        If True, pre-loads the data, by default False

    """

    def __init__(
        self,
        *args,
        tree_format: str = "parquet",
        pre_load: bool = False,
        **kwargs,
    ):
        self.time_column = "Scale"
        super().__init__(*args, alias_path=ALIAS_PATH, **kwargs)
        self.tree_format = tree_format
        self.filepath = np.atleast_1d(self.filepath)

        if self.tree_format not in ["parquet", "hdf5"]:
            raise ValueError(
                f"Invalid tree format: {self.tree_format}. Must be 'parquet' or 'hdf5'."
            )

        if self.tree_format == "hdf5":
            self._init_hdf5()
        elif self.tree_format == "parquet":
            self._init_parquet()

        if pre_load:
            self._loader = [self._loader(fp) for fp in self.filepath]

    def _init_hdf5(self):
        """Initialize the class for handling HDF5 files."""
        dskey = "MergerTree/Galaxy"
        self._loader_kwargs = {"key": dskey, "index_col": "ID"}
        self.columns = pd.read_hdf(self.filepath[0], dskey).columns.tolist()
        self._loader = partial(pd.read_hdf, **self._loader_kwargs)

    def _init_parquet(self):
        """Initialize the class for handling Parquet files."""
        self._loader_kwargs = {"dtype_backend": "pyarrow"}
        parquet_file = pq.ParquetFile("/".join([self.filepath[0], "tree.0.parquet"]))
        schema = parquet_file.schema
        self.columns = schema.names
        self._loader = partial(pd.read_parquet, **self._loader_kwargs)

    def kwarg_swap_alias(self, kwargs):
        """Extend the super function to with a redshift alias.

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
        kwargs = super().kwarg_swap_alias(kwargs)
        if "redshift" in kwargs:
            kwargs[self.time_column] = 1 / (kwargs["redshift"] + 1)
            kwargs.pop("redshift")
        return kwargs

    def query(self, *args, **kwargs) -> pd.DataFrame:
        """Query the data.

        Returns
        -------
        pandas.DataFrame
            The query results.

        """
        if self.tree_format == "hdf5":
            df = pd.concat(self._df_query(*args, **kwargs))
        elif self.tree_format == "parquet":
            df = pd.concat(self._parquet_query(*args, **kwargs))

        if "ID" in df.columns:
            df.set_index("ID", inplace=True, drop=True)
        return df

    def tree(self, index: int) -> pd.DataFrame:
        """Retrieve the tree of a galaxy starting from the given index.

        Parameters
        ----------
        index : int
            The root index of the galaxy tree.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the tree of the galaxy. Each row corresponds to a galaxy,
            and the columns represent various properties of the galaxies.

        Raises
        ------
        NotImplementedError
            If the 'Leaf_ID' column is not present in the DataFrame, indicating that recursive
            loading is not available.

        """
        if "Leaf_ID" in self.columns:
            leaf_idx = self.list(id=index)["Leaf_ID"].squeeze() + 1
            return self.list(min_id=index, max_id=leaf_idx)
        else:
            raise NotImplementedError("recursive loading not yet available.")

    def branch(self, index: int) -> pd.DataFrame:
        """Get the branch of the galaxy tree starting from the given index.

        Parameters
        ----------
        index : int
            The root index of the galaxy tree.

        Returns
        -------
        pd.DataFrame
            The branch of the galaxy tree.

        """
        # TODO: just incrememt index by one until the min scale factor is reached
        tree = self.tree(index)
        # Only include most massive progenitors
        mmp_mask = (tree["MMP"] == 1) | (tree[self.time_column] == tree[self.time_column].max())
        mb = tree.loc[mmp_mask]

        # initialize a mask
        mask = np.full(len(mb), False)
        desc = index

        # iterate over the mmp tree finding the mmp route leading to the root galaxy
        i = 0
        for row in mb.to_records():
            if i == 0:
                mask[i] = True
            else:
                r_id = row["Desc_ID"]
                if r_id == desc:
                    mask[i] = True
                    desc = row["ID"]
            i += 1
        return mb[mask]

    def count(self, target_scales: Union[float, Sequence[float]] = None, dtype=int, **kwargs):
        """Count the number of galaxies at specified scalefactor(s).

        Parameters
        ----------
        target_scales : float, list of floats, optional
            Scale factors at which a galaxy count should be performed, by default None
        dtype : data type, optional
            interpolated values will be cast into this type, by default int
        **kwargs : dict
            Additional keyword arguments for use in the `list` method.

        Returns
        -------
        dtype, numpy array of dtype
            The number of galaxies at each input scale factor

        """
        counts = np.zeros(len(self.scales))
        argsin = []

        n_gal = self.list(columns=[self.time_column], **kwargs).value_counts().sort_index()

        for i, a in enumerate(self.scales):
            if np.isin(a, n_gal.index):
                argsin += [i]
        counts[argsin] = n_gal.values
        if target_scales is None:
            target_scales = self.scales
        target_scales = np.atleast_1d(target_scales)
        func = interp1d(self.scales, counts)
        return func(target_scales).astype(dtype)

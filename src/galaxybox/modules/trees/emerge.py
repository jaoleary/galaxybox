"""Defines the EmergeGalaxyTrees class, a subclass of ProtoGalaxyTree."""

import re
from functools import cached_property, partial
from importlib import resources
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.cosmology import LambdaCDM
from tqdm.auto import tqdm

from galaxybox.data.utils import find_keys_in_string, kwargs_to_filters
from galaxybox.modules.trees.proto_tree import ProtoGalaxyTree
from galaxybox.utils.logmath import logadd, logsum

ALIAS_PATH = resources.files("galaxybox.configs") / "emerge-galaxy.alias.yaml"


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
    output_mstar_threshold : float, optional
        The threshold for outputting star mass, by default 7.0
    fraction_escape_icm : float, optional
        The fraction of something that escapes to the inter-cluster medium (ICM), default 0.562183

    """

    def __init__(
        self,
        *args,
        tree_format: str = "parquet",
        pre_load: bool = False,
        output_mstar_threshold: float = 7.0,
        fraction_escape_icm: float = 0.562183,
        box_size: float = 135.54,
        cosmology: dict = {"H0": 67.77, "Om0": 0.3070, "Ode0": 0.6930, "Ob0": 0.0485},
        **kwargs,
    ):
        self.time_column = "Scale"
        super().__init__(*args, alias_path=ALIAS_PATH, **kwargs)
        self.tree_format = tree_format
        self.filepath = np.atleast_1d(self.filepath)
        self.output_mstar_threshold = output_mstar_threshold
        self.fraction_escape_icm = fraction_escape_icm
        self.box_size = box_size

        self.cosmology = cosmology
        self.columns = []

        if self.tree_format not in ["parquet", "hdf5"]:
            raise ValueError(
                f"Invalid tree format: {self.tree_format}. Must be 'parquet' or 'hdf5'."
            )

        if self.tree_format == "hdf5":
            self._init_hdf5()
        elif self.tree_format == "parquet":
            self._init_parquet()

        for key in self.columns:
            if key not in self.col_alias.keys():
                self.col_alias[key] = [key.lower()]

        if pre_load:
            self._loader = [self._loader(fp) for fp in self.filepath]

    def _init_hdf5(self):
        """Initialize the class for handling HDF5 files."""
        dskey = "MergerTree/Galaxy"
        self._loader_kwargs = {"key": dskey, "index_col": "ID"}
        self.columns = pd.read_hdf(self.filepath[0], dskey).columns.tolist()
        self._loader = partial(pd.read_hdf, **self._loader_kwargs)
        self.scales = (
            self.list(columns=[self.time_column])
            .drop_duplicates()
            .sort_values(self.time_column)
            .values.squeeze()
        )

    def _init_parquet(self):
        """Initialize the class for handling Parquet files."""
        self._loader_kwargs = {"dtype_backend": "pyarrow"}
        parquet_file = pq.ParquetFile("/".join([self.filepath[0], "tree.0.parquet"]))
        schema = parquet_file.schema
        self.columns = schema.names
        self._loader = partial(pd.read_parquet, **self._loader_kwargs)
        self.scales = (
            self.list(columns=[self.time_column])
            .drop_duplicates()
            .sort_values(self.time_column)
            .values.squeeze()
        )

    @property
    def cosmology(self):
        """Astropy cosmology object."""
        return self._cosmology

    @cosmology.setter
    def cosmology(self, value):
        if isinstance(value, dict):
            self._cosmology = LambdaCDM(**value)
        elif isinstance(value, LambdaCDM):
            self._cosmology = value
        else:
            raise TypeError("cosmology must be a dictionary or a LambdaCDM object")

    @cached_property
    def scales(self):
        """Find all scalefactors in the galaxy tree."""
        return np.unique(self.query(query=None, columns=[self.time_column]).values)

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
            leaf_idx = self.list(id=index, columns=["Leaf_ID"]).squeeze() + 1
            return self.list(min_id=index, max_id=leaf_idx)
        else:
            raise NotImplementedError("recursive loading not yet available.")

    def branch(self, index: int) -> pd.DataFrame:
        """Get the main branch of the galaxy tree starting from the given index.

        Parameters
        ----------
        index : int
            The root index of the galaxy tree.

        Returns
        -------
        pd.DataFrame
            The branch of the galaxy tree.

        """
        # the leaf_id is only available when depth first linear indexing is used so we can exploit
        # that along with the simulation time discretness to infer the max id in the main branch
        if "Leaf_ID" in self.columns:
            # first get the scale at the specified index
            scale, leaf_id = self.list(
                id=index, columns=[self.time_column, "Leaf_ID"]
            ).values.squeeze()
            # max_idx is the maximum possible id in the main branch based on the smallest simulation
            # output time.
            max_idx = min(index + np.searchsorted(self.scales, scale), leaf_id + 1)
            # load all possible galaxies in the branch
            branch = self.list(min_id=index, max_id=max_idx)
            # if all galaxies are mmp then then there are no mergers and this is the main branch
            # the first occurance where the most massive progenitor(MMP) is 0 is the end of the
            # main branch
            mmp = branch["MMP"].values == 0
            if sum(mmp) == 0:  # if all are mmp the main `branch` has no mergers
                return branch
            else:
                branch_leaf_idx = np.argwhere(mmp).min()
                return branch.iloc[:branch_leaf_idx]
        else:
            raise NotImplementedError("recursive loading not yet available.")

    def count(self, **kwargs) -> np.ndarray:
        """Count galaxies of some type, at some scale factor."""
        return super().count("Scale", **kwargs)

    @cached_property
    def merger_index(self):
        """Find the progenitor indices and mass ratio of galaxies that have merged."""
        # Since the minor component is destroyed in a merger, only these IDs will be unique
        minor_progs = self.list(
            flag=1,
            min_mstar=self.output_mstar_threshold,
            columns=["Desc_ID", "Stellar_mass", "tdf"],
        ).reset_index()
        minor_progs.rename(columns={"ID": "minor_ID", "Stellar_mass": "minor_mstar"}, inplace=True)

        # Find the galaxy that resulted from the merger
        desc = self.list(
            id=minor_progs["Desc_ID"].values,
            min_mstar=self.output_mstar_threshold,
            columns=["ID", "MMP_ID"],
        ).reset_index()

        # create a table of mergers
        mergers = pd.merge(
            desc, minor_progs, left_on="ID", right_on="Desc_ID", validate="1:m"
        ).drop(columns=["Desc_ID"])
        mergers.rename(columns={"MMP_ID": "major_ID", "ID": "desc_ID"}, inplace=True)

        # set major prog properties
        mergers = pd.merge(
            mergers,
            self.list(
                id=mergers["major_ID"],
                min_mstar=self.output_mstar_threshold,
                columns=["Stellar_mass"],
            ),
            left_on="major_ID",
            right_index=True,
        )
        mergers.rename(columns={"Stellar_mass": "major_mstar"}, inplace=True)

        # Find rows where minor_mstar > major_mstar
        # Then swap major and minor properties for those mergers
        condition = mergers["minor_mstar"] > mergers["major_mstar"]
        mergers.loc[condition, ["major_ID", "minor_ID", "major_mstar", "minor_mstar"]] = (
            mergers.loc[condition, ["minor_ID", "major_ID", "minor_mstar", "major_mstar"]].values
        )

        # our convention is to enforce that MR >= 1
        mergers["stellar_mass_ratio"] = 10 ** (mergers["major_mstar"] - mergers["minor_mstar"])
        mergers.drop(columns=["major_mstar", "minor_mstar"], inplace=True)
        # TODO: correct desc mass and MR for non-binary mergers
        return mergers

    def merger_list(self, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
        """Return a list of mergers based on the provided keyword arguments.

        Parameters
        ----------
        columns : list[str], optional
            A list of columns to return in the DataFrame. If None, all columns are returned.
        **kwargs : dict
            Keyword arguments specifying the criteria for the mergers to be returned.
            These can include properties of the descendant, major progenitor, minor progenitor,
            or other properties related to the mass ratio.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the mergers that meet the specified criteria.

        Raises
        ------
        ValueError
            If multiple prefixes are found in the keyword arguments, a ValueError is raised.

        """
        mergers = self.merger_index.copy()

        # first split kwargs based on which progenior they belong to
        prog_kwargs = {"desc": {}, "major": {}, "minor": {}}
        other_kwargs = {}

        for key in kwargs.keys():
            prefix = find_keys_in_string(prog_kwargs, key)
            # if a prefix is found, remove it from the key and add the key to the prog dictionary
            if len(prefix) == 1:
                clean_key = re.sub(prefix[0], "", key)  # remove the prefix
                clean_key = re.sub(r"^_|_$", "", clean_key)  # remove leading/trailing underscores
                clean_key = re.sub(r"__+", "_", clean_key)  # remove multiple underscores
                prog_kwargs[prefix[0]][clean_key] = kwargs[key]  # add to the correct dictionary
            # other kwargs apply to mass ration
            elif len(prefix) == 0:
                other_kwargs[key] = kwargs[key]
            else:
                raise ValueError(f"Multiple prefixes found: {prefix}")

        # down select based on merger properties first (mass ratio, merger time, etc.)
        other_kwargs = self.kwarg_swap_alias(other_kwargs)
        filters = kwargs_to_filters(other_kwargs)
        if filters is not None:
            query = " & ".join([" ".join(map(str, tup)) for tup in filters])
            mergers = mergers.query(
                query,
                engine="python",  # ? unclear why engine needs to be python
            )

        # secondary selection criteria
        mask = None
        for prefix in prog_kwargs.keys():
            if len(prog_kwargs[prefix]) > 0:
                temp_idx = self.list(columns=["ID"], **prog_kwargs[prefix]).index.values
                if mask is None:
                    mask = mergers[f"{prefix}_ID"].isin(temp_idx)
                else:
                    mask = mask & mergers[f"{prefix}_ID"].isin(temp_idx)

        if mask is not None:
            mergers = mergers[mask]
        # TODO: create the option to load other properties beyond MR, tdf and ID.
        return mergers[columns] if columns is not None else mergers

    def exsitu_mass(self, index: int | Sequence[int], **merger_kwargs) -> pd.DataFrame:
        """Compute the ex-situ stellar mass for galaxies identified by the given index or indices.

        This method calculates the total mass of stars in a galaxy that were not formed in-situ but
        were instead accreted from other galaxies. This is done by aggregating the stellar masses
        of all progenitor galaxies that have merged into the target galaxy up to the current time.

        Parameters
        ----------
        index : int, Sequence[int]
            The identifier(s) of the galaxy or galaxies for which to calculate the ex-situ stellar
            mass. Can be a single ID as a string or a list/array of IDs.
        **merger_kwargs : dict
            Additional keyword arguments to configure the merger process. These can include
            parameters such as the mass ratio threshold for considering a merger significant,
            a time frame for considering historical mergers, and other criteria specific to the
            merger analysis.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the index set to the galaxy IDs and a single column
            'exsitu_Stellar_mass' containing the calculated ex-situ stellar mass for each galaxy.
            The mass values are in log scale.

        Examples
        --------
        >>> galaxy_ids = [123, 456, 789]
        >>> exsitu_mass_df = simulation.exsitu_mass(galaxy_ids)
        >>> print(exsitu_mass_df)
                   exsitu_Stellar_mass
        ID
        123                    10.5
        456                    11.0
        789                    9.8

        """
        index = np.atleast_1d(index)

        # This index list is used to validate which mergers we should consider in the exsitu calc.
        merger_idx = self.merger_list(
            **merger_kwargs, columns=["major_ID", "minor_ID"]
        ).values.flatten()

        # initialize the exsitu_mass DataFrame
        stellar_mass = self.list(id=index, columns=["Stellar_mass"])
        exsitu_mass = pd.DataFrame(
            index=index, columns=["exsitu_Stellar_mass", "next_mmp", "root_ID", "root_mass"]
        )
        exsitu_mass.index.name = "ID"
        exsitu_mass["root_ID"] = index
        exsitu_mass["exsitu_Stellar_mass"] = 0.0
        exsitu_mass["root_mass"] = stellar_mass["Stellar_mass"].values
        exsitu_mass["temp_mass"] = 0.0

        # Starting at the second highest scale factor.
        for a in (pbar := tqdm(self.scales[-2::-1])):
            pbar.set_description(f"scale factor = {a:.3f}")
            # exsitu_mass.index.name = "ID"
            exsitu_mass["temp_mass"] = 0.0

            progs = self.list(
                scale=a,
                idesc=exsitu_mass.index.values,
                max_flag=2,
                columns=["Desc_ID", "MMP", "Stellar_mass_root"],
            )

            # Update the index for next iteration using the ID of the most massive progenitor
            major = progs[progs["MMP"] == 1]
            major.reset_index(inplace=True)
            major.set_index("Desc_ID", inplace=True)
            exsitu_mass["next_mmp"] = exsitu_mass.index.values
            exsitu_mass.update({"next_mmp": major["ID"]})

            # Add the stellar mass of the minor progenitors to the ex-situ mass
            minor = progs[progs["MMP"] == 0]
            # only include mergers selected according to merger_kwargs
            minor = minor[minor.index.isin(merger_idx)]
            mass_frac = minor.groupby("Desc_ID")["Stellar_mass_root"].agg(logsum)
            exsitu_mass.update({"temp_mass": mass_frac})
            exsitu_mass.set_index("next_mmp", inplace=True)

            # zero in the array is zero stellar mass, not log 0 stellar mass
            # so we ignore those rows so that
            non_zero_mass = (exsitu_mass["temp_mass"] > 0) & (
                exsitu_mass["exsitu_Stellar_mass"] > 0
            )
            exsitu_mass.loc[non_zero_mass, "exsitu_Stellar_mass"] = logadd(
                exsitu_mass.loc[non_zero_mass, "exsitu_Stellar_mass"],
                exsitu_mass.loc[non_zero_mass, "temp_mass"],
            )

            # if the exsitu mass is zero but the temp mass is not, then the exsitu mass is updated
            non_zero_temp_mass = (exsitu_mass["exsitu_Stellar_mass"] == 0) & (
                exsitu_mass["temp_mass"] > 0
            )
            exsitu_mass.loc[non_zero_temp_mass, "exsitu_Stellar_mass"] = exsitu_mass.loc[
                non_zero_temp_mass, "temp_mass"
            ]

        exsitu_mass.set_index("root_ID", inplace=True)
        exsitu_mass.index.name = "ID"

        # addjust the accreted mass by the fraction of mass that escapes the galaxy during merging
        zero_mask = exsitu_mass["exsitu_Stellar_mass"] > 0
        exsitu_mass.loc[zero_mask, "exsitu_Stellar_mass"] = np.log10(
            (1 - self.fraction_escape_icm)
            * 10 ** (exsitu_mass.loc[zero_mask, "exsitu_Stellar_mass"])
        )

        return exsitu_mass[["exsitu_Stellar_mass"]]

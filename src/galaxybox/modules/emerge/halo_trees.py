"""Classes for handling Emerge halo merger tree data."""

import glob

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from galaxybox.data.emerge_io import read_halo_forests, read_halo_trees


class HaloTrees:
    """A class for reading in and operating on Emerge formated halo merger trees."""

    def __init__(self, trees_path, add_attrs=None, include="halos"):
        """Initialize a `halo_trees` object.

        Parameters
        ----------
        trees_path : string or list of strings
            A list of tree files to be included into the class.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None
        include : str, optional
            If include='halos' a snapshot dict will be created for the trees, by default 'halos'

        """
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])

        if isinstance(trees_path, str):
            trees_path = [trees_path]

        self.trees_used = sorted(trees_path)

        forest_path = self.trees_used[0].split(".")[0] + ".forests"
        self.forests = read_halo_forests(forest_path)
        self.Ntrees = 0
        self.Nhalos = []
        self.TreeID = []
        frames = [None] * len(self.trees_used)
        for i, file in enumerate(tqdm(self.trees_used, desc="Loading halo trees")):
            ntrees, nhalos, treeid, frames[i] = read_halo_trees(file)
            self.Ntrees += ntrees
            self.Nhalos += list(nhalos)
            self.TreeID += list(treeid)
        forest = pd.concat(frames)
        forest.set_index("haloid", inplace=True)

        #! TreeID numbers are incremented by 1 to match the id system used internally within Emerge
        # TODO: make this optional
        self.TreeID = np.array(self.TreeID) + 1

        self.trees = forest
        # no more little h
        self.trees[["mvir", "rvir", "X_pos", "Y_pos", "Z_pos"]] = self.trees[
            ["mvir", "rvir", "X_pos", "Y_pos", "Z_pos"]
        ] / (self.cosmology.H0.value / 100)
        self.trees["mvir"] = np.log10(self.trees["mvir"])
        self.trees["rvir"] = (
            self.trees["rvir"] * self.trees["scale"]
        )  # we dont want radius in comoving units....

        idx = np.cumsum([0] + self.Nhalos)
        rootid = np.zeros(np.sum(self.Nhalos), dtype=int)
        for i, r in enumerate(idx[:-1]):
            rootid[idx[i] : idx[i + 1]] = self.TreeID[i]
        self.trees["rootid"] = rootid

        self.trees["hostid"] = -1
        max_scale = self.trees["scale"].max()
        root_mask = self.trees["scale"] == max_scale
        forest_mask = self.trees["upid"] == 0
        self.trees.loc[root_mask & forest_mask, "hostid"] = self.trees.loc[
            root_mask & forest_mask
        ].index.values
        self.trees.loc[root_mask & ~forest_mask, "hostid"] = self.trees.loc[
            root_mask & ~forest_mask
        ]["upid"].values
        self.trees["hostid"] = self.trees.loc[self.trees["rootid"].values, "hostid"].values

        self.scales = np.sort(self.trees.scale.unique())
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])

        self.snapshots = []
        if include == "halos":
            self._halos = {}
            for i in range(len(self.scales)):
                self.snapshots.append("S{:d}".format(i))
                self._halos["S{:d}".format(i)] = self.trees["scale"] == self.scales[i]
        else:
            for i in range(len(self.scales)):
                self.snapshots.append("S{:d}".format(i))

    @classmethod
    def from_universe(cls, universe, include):
        """Initiate a `halo_tree` within the `Universe` class.

        Parameters
        ----------
        universe : `galaxybox.Universe`
            The outer universe class used to organize these galaxy trees
        include : str
            If include='halos' a snapshot dict will be created for the trees, by default 'halos'

        Returns
        -------
        galaxybox.halo_trees
            halo trees set from the universe

        """
        add = {
            "out_dir": universe.out_dir,
            "fig_dir": universe.fig_dir,
            "TreefileName": universe.TreefileName,
            "ModelName": universe.ModelName,
            "BoxSize": universe.BoxSize,
            "UnitTime_in_yr": universe.UnitTime_in_yr,
            "UnitMass_in_Msun": universe.UnitMass_in_Msun,
            "UnitLength_in_Mpc": universe.UnitLength_in_Mpc,
            "cosmology": universe.cosmology,
        }
        trees_path = [name for name in glob.glob(universe.TreefileName + ".*")]
        for i, f in enumerate(trees_path):
            if "forests" in f:
                del trees_path[i]
        return cls(trees_path, add, include=include)

    def alias(self, key):
        """Return proper coloumn key for input alias key.

        Parameters
        ----------
        key : str
            A string alias for a halo tree column

        Returns
        -------
        str
            The proper column name for the input alias

        """
        # first lets make all columns case insensitive
        colnames = list(self.trees.keys())
        col_alias = {}
        for k in colnames:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias["scale"] += ["a", "scale_factor"]
        col_alias["redshift"] = ["redshift"]
        col_alias["snapshot"] = ["snapshot", "snapnum", "snap"]
        col_alias["mvir"] += ["halo_mass", "virial_mass", "mass"]
        col_alias["upid"] += ["ihost", "host_id", "id_host", "up_id"]
        col_alias["descid"] += ["idesc", "id_desc", "desc_id"]
        col_alias["np"] += ["num_prog"]
        col_alias["rvir"] += ["halo_radius", "virial_radius", "radius"]
        col_alias["concentration"] += ["concentration", "c"]
        col_alias["lambda"] += ["spin"]
        col_alias["Type"] += ["htype"]

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))

    def list(self, mask_only=False, **kwargs):
        """Return list of halos with specified properties.

        This function selects halos based on a flexible set of arguments. Any column of the
        galaxy.trees attribute can have a `min` are `max` added as a prefix or suffix to the column
        name or alias of the column name and passed as and argument to this function. This can also
        be used to selected halos based on derived properties such as color.

        Parameters
        ----------
        mask_only : bool, optional
            Return the dataframe, or a mask for the dataframe, by default False
        **kwargs : dict
            Additional keyword arguments for specifying halo properties.

        Returns
        -------
        pandas.DataFrame
            A masked subset of the halo.trees attribute

        # TODO: expand this docstring and add examples.

        """
        # First clean up kwargs, replace aliases
        keys = list(kwargs.keys())
        for i, kw in enumerate(keys):
            keyl = kw.lower()
            key = keyl.lower().replace("min", "").replace("max", "").lstrip("_").rstrip("_")
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        # I dont like the implementation of `all` right now.
        # Allowing for specified ranges would be better.
        if "snapshot" in kwargs.keys():
            if kwargs["snapshot"] == "all":
                halo_list = self.trees
            else:
                halo_list = self.trees.loc[self._halos[kwargs["snapshot"]]]
            kwargs.pop("snapshot")
        elif "scale" in kwargs:
            if kwargs["scale"] == "all":
                halo_list = self.trees
            else:
                scale_arg = (np.abs(self.scales - kwargs["scale"])).argmin()
                s_key = self.snapshots[scale_arg]
                halo_list = self.trees.loc[self._halos[s_key]]
            kwargs.pop("scale")
        elif "redshift" in kwargs:
            if kwargs["redshift"] == "all":
                halo_list = self.trees
            else:
                scale_factor = 1 / (kwargs["redshift"] + 1)
                scale_arg = (np.abs(self.scales - scale_factor)).argmin()
                s_key = self.snapshots[scale_arg]
                halo_list = self.trees.loc[self._halos[s_key]]
            kwargs.pop("redshift")
        else:
            s_key = self.snapshots[-1]
            halo_list = self.trees.loc[self._halos[s_key]]

        # Setup a default `True` mask
        mask = halo_list["scale"] > 0
        # Loop of each column in the tree and check if a min/max value mask should be created.
        for i, key in enumerate(self.trees.keys()):
            for j, kw in enumerate(kwargs.keys()):
                if key.lower() in kw.lower():
                    if "min" in kw.lower():
                        mask = mask & (halo_list[key] >= kwargs[kw])
                    elif "max" in kw.lower():
                        mask = mask & (halo_list[key] < kwargs[kw])
                    else:
                        values = np.atleast_1d(kwargs[kw])
                        # Setup a default `False` mask
                        sub_mask = halo_list["scale"] > 1
                        for v in values:
                            sub_mask = sub_mask | (halo_list[key] == v)
                        mask = mask & sub_mask

        if mask_only:
            return mask
        else:
            return halo_list.loc[mask]

    def count(self, target_scales=None, dtype=int, **kwargs):
        """Count the number of halos at specified scalefactor(s).

        Parameters
        ----------
        target_scales : float, list of floats, optional
            Scale factors at which a halo count should be performed, by default None
        dtype : data type, optional
            interpolated values will be cast into this type, by default int
        **kwargs : dict
            Additional keyword arguments for specifying halo properties.

        Returns
        -------
        dtype, numpy array of dtype
            The number of halos at each input scale factor

        """
        counts = np.zeros(len(self.scales))
        argsin = []

        n_halo = self.list(scale="all", **kwargs)["scale"].value_counts().sort_index()

        for i, a in enumerate(self.scales):
            if np.isin(a, n_halo.index):
                argsin += [i]
        counts[argsin] = n_halo.values
        if target_scales is None:
            target_scales = self.scales
        target_scales = np.atleast_1d(target_scales)
        func = interp1d(self.scales, counts)
        return func(target_scales).astype(dtype)

    def find_mergers(self, enforce_positive_mr=True):
        """Create a list of all halo mergers in the tree.

        Mergers are specified in terms of descendant, main progenitor, minor progenitor and compute
        the mass ratio.

        Parameters
        ----------
        enforce_positive_mr : bool, optional
            swap main and minor galaxy progenitors to ensure mass rato > 1, by default True

        Returns
        -------
        pandas.DataFame
            DataFrame of halo mergers

        """
        print("Making merger list from halo trees")
        # Initialise dataframe of merging events
        mergers = pd.DataFrame(
            columns=[
                "Scale",
                "Desc_ID",
                "Desc_mvir",
                "Main_ID",
                "Main_mvir",
                "Minor_ID",
                "Minor_mvir",
                "MR",
            ]
        )

        # Locate merging systems of sufficient mass
        possible_prog_mask = (self.trees["upid"] == 0) & (self.trees["descid"] != 0)
        possible_progs = self.trees.loc[possible_prog_mask]

        true_desc_mask = self.trees.loc[possible_progs["descid"]]["upid"] != 0
        true_desc = self.trees.loc[true_desc_mask.loc[true_desc_mask.values].index]

        mergers["Minor_mvir"] = possible_progs.loc[possible_progs["descid"].isin(true_desc.index)][
            "mvir"
        ].copy()
        mergers["Minor_ID"] = mergers.index.values
        mergers.reset_index(inplace=True, drop=True)
        mergers["Desc_ID"] = true_desc["upid"].values

        # Find properties of descendent galaxy
        mergers[["Scale", "Desc_mvir"]] = self.trees.loc[mergers["Desc_ID"].values][
            ["scale", "mvir"]
        ].values

        # Find coprogenitor properties
        main_progs = possible_progs.loc[
            (possible_progs["mmp"] == 1) & possible_progs.descid.isin(mergers["Desc_ID"])
        ]
        main_progs.reset_index(inplace=True)
        main_progs.set_index("descid", inplace=True)
        desc_mask = mergers["Desc_ID"].isin(main_progs.index)
        mergers.loc[desc_mask, "Main_ID"] = main_progs.loc[
            mergers.loc[desc_mask]["Desc_ID"].values
        ].haloid.values
        mergers.loc[~desc_mask, "Main_ID"] = mergers.loc[~desc_mask]["Desc_ID"].values
        mergers["Main_mvir"] = self.trees.loc[mergers["Main_ID"].values]["mvir"].values

        # Enforce M1 >= M2
        if enforce_positive_mr:
            swap = mergers["Main_mvir"] < mergers["Minor_mvir"]
            mergers.loc[swap, ["Main_ID", "Main_mvir", "Minor_ID", "Minor_mvir"]] = mergers.loc[
                swap, ["Minor_ID", "Minor_mvir", "Main_ID", "Main_mvir"]
            ].values
        # compute mass ratio
        mergers["MR"] = 10 ** (mergers["Main_mvir"] - mergers["Minor_mvir"])
        mergers[["Main_ID", "Minor_ID"]] = mergers[["Main_ID", "Minor_ID"]].astype("int")

        return mergers

import numpy as np
from torch.utils.data import Dataset

class ProtoTree(Dataset):
    def __getitem__(index):
        return
    def tree(self, *args, **kwargs):
        raise NotImplementedError
    
    def branch(self, *args, **kwargs):
        raise NotImplementedError
    
    def list(self, *args, **kwargs):
        raise NotImplementedError
    
    def alias(self, *args, **kwargs):
        raise NotImplementedError
    
    def query(self, query, columns):
        raise NotImplementedError

class ProtoGalaxyTree(ProtoTree):
    def __init__(self, filepath, mode='tree', UnitTime_in_yr=1.0e9):
        self.filepath = filepath
        self.mode = mode
        self.UnitTime_in_yr = UnitTime_in_yr

    def alias(self, key):
        """Return proper coloumn key for input alias key.

        Parameters
        ----------
        key : str
            A string alias for a galaxy tree column

        Returns
        -------
        str
            The proper column name for the input alias
        """
        col_alias : dict[str, list[str]] = {}
        for k in self.columns:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias["Scale"] += ["a", "scale_factor"]
        col_alias["redshift"] = ["redshift", "z"]
        col_alias["snapshot"] = ["snapshot", "snapnum", "snap"]
        col_alias["color"] = ["color"]
        col_alias["color_obs"] = ["color_obs", "col_obs", "color_observed"]
        col_alias["Stellar_mass"] += ["mass", "mstar"]
        col_alias["Halo_mass"] += ["mvir"]
        col_alias["Type"] += ["gtype"]
        col_alias["Up_ID"] += ["ihost", "host_id", "id_host"]
        col_alias["Desc_ID"] += ["idesc", "id_desc"]
        col_alias["Main_ID"] += ["imain", "id_main"]
        col_alias["MMP_ID"] += ["immp", "id_mmp"]
        col_alias["Coprog_ID"] += ["icoprog", "id_coprog"]
        col_alias["Leaf_ID"] += ["ileaf", "id_leaf"]
        col_alias["Original_ID"] += [
            "ogid",
            "rockstar_id",
            "id_rockstar",
            "id_original",
            "rs_id",
            "id_rs",
            "irs",
            "rsid",
        ]
        col_alias["Num_prog"] += ["np"]
        if "Intra_cluster_mass" in self.columns:
            col_alias["Intra_cluster_mass"] += ["icm"]
        col_alias["Stellar_mass_root"] += [
            "mstar_root",
            "root_mstar",
            "rootmass",
            "root_mass",
        ]
        col_alias["Stellar_mass_obs"] += ["mstar_obs"]
        col_alias["Halo_radius"] += ["rvir", "virial_radius", "radius"]

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))
    
    def list(self, columns=None, **kwargs):
        """Return list of galaxy with specified properties.

        This function selects galaxies based on a flexible set of arguments. Any column of the galaxy.trees attribute can
        have a `min` are `max` added as a prefix or suffix to the column name or alias of the column name and passed as
        and argument to this function. This can also be used to selected galaxies based on derived properties such as color.

        Parameters
        ----------
        mask_only : bool, optional
            Return the dataframe, or a mask for the dataframe, by default False

        Returns
        -------
        pandas.DataFrame
            A masked subset of the galaxy.trees attribute

        # TODO: expand this docstring and add examples.
        """
        if columns is None:
            columns = self.columns
        read_columns = columns
        # First clean up kwargs, replace aliases
        keys = list(kwargs.keys())
        for i, kw in enumerate(keys):
            keyl = kw.lower()
            key = (
                keyl.lower()
                .replace("min", "")
                .replace("max", "")
                .lstrip("_")
                .rstrip("_")
            )
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        if "redshift" in kwargs:
            kwargs['Scale'] = 1 / (kwargs["redshift"] + 1)
            kwargs.pop("redshift")

        # Setup a default `True` mask
        # mask = galaxy_list["Scale"] > 0
        filters = []
        # Loop of each column in the tree and check if a min/max value mask should be created.
        for i, key in enumerate(self.columns):
            for j, kw in enumerate(kwargs.keys()):
                if ("obs" in kw.lower()) & ("obs" not in key.lower()):
                    pass
                elif key.lower() in kw.lower():
                    if "min" in kw.lower():
                        filters.append((key, '>=', kwargs[kw]))
                    elif "max" in kw.lower():
                        filters.append((key, '<', kwargs[kw]))
                    else:
                        values = np.atleast_1d(kwargs[kw]).tolist()
                        filters.append((key, 'in', values))

        galaxies = self.query(query=filters, columns=read_columns)

        columns = [item for item in columns if item != galaxies.index.name]
        return galaxies[columns]
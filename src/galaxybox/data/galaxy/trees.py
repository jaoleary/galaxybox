import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq

class GalaxyTreesDataset(Dataset):
    UnitTime_in_yr=1.0e9
    def __init__(self, filepath, preload=True, mode='tree'):
        self.filepath = filepath
        
        # Open the Parquet file and get the schema
        parquet_file = pq.ParquetFile('/'.join([self.filepath, 'tree.0.parquet']))
        schema = parquet_file.schema
        self.columns = schema.names
    def __getitem__(self, index):
        return 
        
    def tree(self, index):
        if 'Leaf_ID' in self.columns:
            leaf_idx = self.list(id=index)['Leaf_ID'].squeeze() + 1
            return self.list(min_id=index, max_id=leaf_idx)
        else:
            raise NotImplementedError('recursive loading not yet available.')
    
    def branch(self, index):
        tree = self.tree(index)
        # Only include most massive progenitors
        mmp_mask = (tree["MMP"] == 1) | (tree["Scale"] == tree["Scale"].max())
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
    
    def list(self, mask_only=False, **kwargs):
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

        galaxies = pd.read_parquet(self.filepath, filters=filters, dtype_backend='pyarrow')
        # Setup a default `True` mask, this is kindof a dumb step
        mask = galaxies["Scale"] > 0
        # Create masks for derived quantities such as `color`.
        if "color" in kwargs.keys():
            if kwargs["color"].lower() == "blue":
                mask = mask & (
                    (np.log10(galaxies["SFR"]) - galaxies["Stellar_mass"])
                    >= np.log10(0.3 / galaxies["Age"] / self.UnitTime_in_yr)
                )
            elif kwargs["color"].lower() == "red":
                mask = mask & (
                    (np.log10(galaxies["SFR"]) - galaxies["Stellar_mass"])
                    < np.log10(0.3 / galaxies["Age"] / self.UnitTime_in_yr)
                )
        if "color_obs" in kwargs.keys():
            if kwargs["color_obs"].lower() == "blue":
                mask = mask & (
                    (np.log10(galaxies["SFR_obs"]) - galaxies["Stellar_mass_obs"])
                    >= np.log10(0.3 / galaxies["Age"] / self.UnitTime_in_yr)
                )
            elif kwargs["color_obs"].lower() == "red":
                mask = mask & (
                    (np.log10(galaxies["SFR_obs"]) - galaxies["Stellar_mass_obs"])
                    < np.log10(0.3 / galaxies["Age"] / self.UnitTime_in_yr)
                )

        return galaxies.loc[mask] 
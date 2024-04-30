import pandas as pd

import numpy as np
import pyarrow.parquet as pq
import h5py

from galaxybox.data.trees.proto_tree import ProtoTree

class EmergeGalaxyTrees(ProtoTree):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dskey = 'data'
        self.filepath = np.atleast_1d(self.filepath)

        # Get the column names from the DataFrame
        self.columns = pd.read_hdf(self.filepath[0], self.dskey).columns.tolist()

    def __getitem__(self, index):
        raise NotImplementedError
    
    def query(self, query, columns):
        query = ' & '.join([' '.join(map(str, tup)) for tup in query])
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

    
    
    
def read_statistics(file_path, universe_num=0):
    """Read an Emerge statistics.h5 file.

    Parameters
    ----------
    file_path : string
        Path of file to be read

    universe_num: int, or boolean, optional
        If int only the specified universe will be loaded, otherwise all are loaded.

    Returns
    -------
    statfile : HDF5 group
        An h5py hdf5 group object containing statistics for simulated universe.

    """
    if file_path.endswith(".h5"):
        statfile = h5py.File(file_path, "r")
        statfilekeys = [key for key in statfile.keys()]
        h5f = statfile[statfilekeys[universe_num]]
        stats = recursive_hdf5(h5f)
        statfile.close()
        return stats
    else:
        stats = {
            "CSFRD": [
                ("Redshift", float),
                ("Csfrd_observed", float),
                ("Sigma_observed", float),
            ],
            "FQ": [
                ("Stellar_mass", float),
                ("Fq_observed", float),
                ("Sigma_observed", float),
                ("Mean_ScaleFactor", float),
                ("Fq_model", float),
                ("Sigma_model", float),
            ],
            "SMF": [
                ("Stellar_mass", float),
                ("Phi_observed", float),
                ("Sigma_observed", float),
                ("Mean_ScaleFactor", float),
                ("Phi_model", float),
                ("Sigma_model", float),
            ],
            "SSFR": [
                ("Redshift", float),
                ("Ssfr_observed", float),
                ("Sigma_observed", float),
                ("Stellar_mass", float),
                ("Ssfr_model", float),
                ("Sigma_model", float),
            ],
        }

        # for now this is hardcoded....in the future will allow for keyword imports
        stat_keys = None
        if stat_keys is not None:
            stat_keys = np.atleast_1d(stat_keys)
            stat_bool = []
            for i, k in enumerate(stat_keys):
                if k in stats.keys():
                    stat_bool.append(True)
                else:
                    stat_bool.append(False)

            if not all(stat_bool):
                raise KeyError(
                    "{} are not valid statistics keys!".format(
                        stat_keys[[not i for i in stat_bool]]
                    )
                )
        else:
            stat_keys = list(stats.keys())

        obs = {}
        for k in stat_keys:
            obs[k] = {}
            obs[k]["Data"] = {}
            if k == "Clustering":
                filepath = file_path + "wpobs.{:d}.out".format(universe_num)
            else:
                filepath = file_path + k.lower() + "obs.{:d}.out".format(universe_num)
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    if line.startswith("#"):
                        line = line.replace("\n", "")
                        key = line.replace("# ", "")
                        data = []
                    elif line.strip():
                        line = np.fromstring(line, sep=" ")
                        data.append(line)
                    else:
                        obs[k]["Data"][key] = np.array(data, dtype=stats[k])
                    line = fp.readline()

            if k == "Clustering":
                filepath = file_path + "wpmod.{:d}.out".format(universe_num)
            else:
                filepath = file_path + k.lower() + "mod.{:d}.out".format(universe_num)
            obs[k]["Model"] = np.loadtxt(filepath)

        return obs
    
class EmergeHaloTrees(ProtoTree):
    def __init__(self, filepath):
        self.filepath = filepath
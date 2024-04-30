import pandas as pd

import numpy as np
import pyarrow.parquet as pq
import h5py

from galaxybox.data.trees.proto_tree import ProtoGalaxyTree

class ParquetGalaxyTrees(ProtoGalaxyTree):
    
    def __init__(self, filepath, mode='tree', UnitTime_in_yr=1.0e9):
        self.filepath = filepath
        self.mode = mode
        self.UnitTime_in_yr = UnitTime_in_yr

        # Open the Parquet file and get the schema
        parquet_file = pq.ParquetFile('/'.join([self.filepath, 'tree.0.parquet']))
        schema = parquet_file.schema
        self.columns = schema.names

    def __getitem__(self, index):
        raise NotImplementedError
    
    def query(self, query, columns):
        columns = columns[:-1]

        return pd.read_parquet(self.filepath, filters=query, dtype_backend='pyarrow', columns=columns)
    
        
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

    

from galaxybox.data.trees.proto_tree import ProtoGalaxyTree
import numpy as np
import pandas as pd

class PYTablesGalaxyTrees(ProtoGalaxyTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dskey = 'data'
        self.filepath = np.atleast_1d(self.filepath)
        self.datastore = [pd.HDFStore(fp) for fp in self.filepath]
        self.columns = list(self.datastore[0][f'/{self.dskey}'].keys())

    def __getitem__(self, index):
        raise NotImplementedError
    
    def query(self, query, columns):
        # make sure all items are strings
        query = ' & '.join([' '.join(map(str, tup)) for tup in query])
        galaxies = pd.concat([ds.select(self.dskey, query) for ds in self.datastore])
        
        return galaxies


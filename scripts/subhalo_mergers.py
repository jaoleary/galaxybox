from galaxybox.io.emerge_io import read_halo_trees
import pandas as pd
import numpy as np
import sys
import glob
from multiprocessing import Pool
import h5py
import os

fbase_in = sys.argv[1] # should be absolute file path
hubble_param = np.float(sys.argv[2]) # as an example: 0.6777
num_procs = np.int(sys.argv[3]) # number of processors


def subhalo_mergers(fname):
    print(fname)
    _, _, _, trees = read_halo_trees(fname)
    trees.set_index('haloid', inplace=True)
    trees[['mvir']] = trees[['mvir']] / (hubble_param)
    trees['mvir'] = np.log10(trees['mvir'])

    # First set Mvir peak for halo
    # Init mvir peak as current mass
    trees['mvir_peak'] = trees['mvir'].values

    # Start at the leaves that are the mmp
    mask = (trees.np == 0) & (trees.mmp == 1) & (trees.descid > 0)
    ids = trees.loc[(trees.np == 0) & (trees.mmp == 1)].index.values

    # loop over consective descendants and update peak mass
    while len(ids) > 0:
        halo_peak_mass = trees.loc[ids]['mvir_peak'].values

        descid = trees.loc[ids]['descid'].values
        desc_mass = trees.loc[descid]['mvir'].values

        update_peak = desc_mass < halo_peak_mass
        update_id = trees.loc[descid].loc[update_peak].index.values

        # If the mass of the desc is less than the peak mass of prog, then udpate desc peak
        trees.loc[update_id, ['mvir_peak']] = halo_peak_mass[update_peak]

        # find the next set of descs and move on
        mask = (trees.loc[descid].mmp == 1) & (trees.loc[descid].descid > 0)
        ids = descid[mask]


    # Find merging systems. i.e mmp !=0
    # initialize mergers array
    mergers = pd.DataFrame(columns=['Scale', 'Desc_ID', 'Desc_mvir',
                                    'Main_ID', 'Main_mvir',
                                    'Minor_ID', 'Minor_mvir', 'Minor_mvir_peak','MR'])


    # first locate the minor progenitor
    prog_mask = (trees['descid'] != 0) & (trees['mmp'] != 1)
    # set values for the minor halo
    mergers['Minor_mvir'] = trees.loc[prog_mask]['mvir'].values
    mergers['Minor_ID'] = trees.loc[prog_mask].index.values
    mergers['Minor_mvir_peak'] = trees.loc[mergers['Minor_ID'].values]['mvir_peak'].values

    # Find properties of descendent galaxy
    mergers['Desc_ID'] = trees.loc[prog_mask]['descid'].values
    mergers[['Scale', 'Desc_mvir']] = trees.loc[mergers['Desc_ID'].values][['scale', 'mvir']].values

    # Find main progenitor properties
    main_progs = trees.loc[(trees['mmp']==1) & trees.descid.isin(mergers['Desc_ID'])]
    main_progs.reset_index(inplace=True)
    main_progs.set_index('descid', inplace=True)
    mergers[['Main_ID', 'Main_mvir']] = main_progs.loc[mergers.Desc_ID.values][['haloid', 'mvir']].values
    mergers['Main_ID'] = mergers['Main_ID'].values.astype(int)
    # compute mass ratio
    mergers['MR'] = 10**(mergers['Main_mvir'] - mergers['Minor_mvir_peak'])
    # enforce MR >= 1
    invert_MR = mergers['MR'] < 1
    mergers.loc[invert_MR, ['MR']] = 1/mergers.loc[invert_MR, ['MR']]

    return mergers


trees_path = [name for name in glob.glob(fbase_in + '*')]
for i, f in enumerate(trees_path):
    if 'forests' in f:
        del trees_path[i]

print('Found {:d} halo merger trees.'.format(len(trees_path)))


with Pool(processes=num_procs) as pool:
    result = list(pool.map(subhalo_mergers, trees_path))

#stitch it all together.
mergers = pd.concat(result)
mergers.reset_index(inplace=True, drop=True)

file_out = './subhalo_mergers.h5'

try:
    os.remove(file_out)
except OSError:
    pass

f = h5py.File(file_out, 'w')
data = mergers.to_records(index=False)
f.create_dataset('Data', data=data, compression='gzip', compression_opts=9)
f.close()


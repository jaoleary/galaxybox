"""Removes the substructure for your emerge formated rockstar/constistent trees halo merger trees. Fun!"""
import galaxybox as gb
import pandas as pd
import numpy as np
import sys
import glob

from tqdm.auto import tqdm

#TODO: This script needs MUCH better documentation

fbase_in = sys.argv[1]
fbase_out = sys.argv[2]

cout = ['descid', 'upid', 'np', 'mmp', 'scale', 'mvir', 'rvir', 'c', 'lambda', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']

trees_path = [name for name in glob.glob(fbase_in + '*')]
for i, f in enumerate(trees_path):
    if 'forests' in f:
        del trees_path[i]

print('Found {:d} halo merger trees.'.format(len(trees_path)))
frames = [None] * len(trees_path)
for i, file in enumerate(tqdm(trees_path, desc='Loading halo trees')):
        _, _, _, frames[i] = gb.read_halo_trees(file)

iso_trees = pd.concat(frames)
iso_trees.set_index('haloid', inplace=True)
forests = gb.read_halo_forests(fbase_in + '.forests')

print('Setting `upid`.')
# first cleanup the upid
satmask = iso_trees['upid'] > 0
broken  = iso_trees.loc[iso_trees.loc[satmask]['upid']].Type == 1
while broken.sum() > 0:
    bnfidx = iso_trees.loc[satmask].loc[broken.values].index # get index with a 'bad' upid
    iso_trees.loc[bnfidx, 'upid'] = iso_trees.loc[iso_trees.loc[iso_trees.loc[bnfidx, 'upid'].values, 'upid'].values].index.values # reset halo.upid to the halo[halo.upid].upid
    broken  = iso_trees.loc[iso_trees.loc[satmask]['upid']].Type == 1 # check if all halos have been fixed this cycle


sats = iso_trees.loc[(iso_trees['Type'] == 1) & (iso_trees['descid'] > 1)]
flythrough = (iso_trees.loc[sats['descid'].values]['Type'] == 0).values
new_leaf = sats.loc[flythrough]['descid'].values
iso_trees.loc[new_leaf, 'np'] = 0

# find infalling halos
# Initialise dataframe of merging events
mergers = pd.DataFrame(columns=['Desc_ID', 'Minor_ID'])

print('Locating merging systems.')
# Locate merging systems
possible_prog_mask = (iso_trees['upid'] == 0) & (iso_trees['descid'] != 0) # locate all main halos that have a desc.
possible_progs = possible_prog_mask[possible_prog_mask].index.values # get their index
descendants = iso_trees.loc[possible_progs]['descid'].values
has_subhalo_desc = (iso_trees.loc[descendants]['upid'] != 0).values
possible_progs = possible_progs[has_subhalo_desc]
descendants = descendants[has_subhalo_desc]
mergers['Minor_ID'] = possible_progs
mergers['Desc_ID'] = iso_trees.loc[descendants]['upid'].values
mergers.set_index('Minor_ID', inplace=True)

print('Setting new descendants.')
infalling = mergers.index.values
iso_trees.loc[infalling, 'descid'] = mergers['Desc_ID'].values # set new desc ID in trees
iso_trees.loc[infalling, 'mmp'] = 0 # Infalling satellites are not mmp.

# kill all subhalos
iso_trees = iso_trees.loc[iso_trees.Type == 0]

num_prog = iso_trees.descid.value_counts()[1:] # start for idx=1 because haloid=0 is reserved
num_prog = num_prog[num_prog > 1]
non_leaf = iso_trees['np'] > 0

#for everything that isnt a leaf, reset the number of pregenitors to 1
iso_trees.loc[non_leaf, 'np'] = 1
#update number of progenitors
iso_trees.loc[num_prog.index.values, 'np'] = num_prog.values

print('Fixing mmp')
#find halos with more than one prog
desc = iso_trees.loc[iso_trees.np>1].index.values
mmp_check = iso_trees.loc[iso_trees.descid.isin(desc)][['descid', 'mvir', 'mmp']].copy()
# set all mmp to zero
iso_trees.loc[mmp_check.index.values, 'mmp'] = 0
# sort by ascending desc_ID descending virial mass
mmp_check.sort_values(['descid', 'mvir'], ascending=[True, False], inplace=True)
# get the index of the actual mmp
true_mmp = mmp_check.iloc[mmp_check.descid.searchsorted(mmp_check.descid.unique())].index.values
# set those in the new trees as mmp=1
iso_trees.loc[true_mmp, 'mmp'] = 1

print('Fixing broken leaves')
# first find all leaves
descid = iso_trees.loc[iso_trees.np==0].index.values
# find all halos that are merger onto that leaf, this is due to a merger immediately post fly through.
progid = iso_trees.loc[iso_trees.descid.isin(descid)].index.values
# make all of these the mmp (this does not gaurantee physically consistent halo growth...)
iso_trees.loc[progid,'mmp'] = 1
descid = iso_trees.loc[progid]['descid'].values
# the descendants are no longer marked as a leaf....
iso_trees.loc[descid, 'np'] = 1

#update the rootid
iso_trees['rootid'] = 0

scale_mask = iso_trees['scale'] == iso_trees['scale'].max()
iso_trees.loc[scale_mask, 'rootid'] = iso_trees.loc[scale_mask].index.values
iso_trees.loc[scale_mask, 'FileNumber'] = iso_trees.loc[scale_mask, 'FileNumber'].values

for i, a in enumerate(tqdm(np.sort(iso_trees.scale.unique())[::-1][1:], desc='Rebuilding trees')):
    scale_mask = iso_trees['scale'] == a
    iso_trees.loc[scale_mask, ['FileNumber', 'rootid']] = iso_trees.loc[iso_trees.loc[scale_mask, 'descid'].values][['FileNumber', 'rootid']].values


desc_mask = iso_trees.descid > 0
iso_trees.loc[desc_mask, 'desc_mass'] = iso_trees.loc[iso_trees.loc[desc_mask, 'descid'].values, 'mvir'].values
iso_trees.loc[~desc_mask, 'desc_mass'] = iso_trees.loc[iso_trees.loc[~desc_mask].index.values, 'mvir'].values
iso_trees['root_mass'] = iso_trees.loc[iso_trees['rootid'].values]['mvir'].values

print('Sorting trees...')
iso_trees.sort_values(['root_mass', 'rootid', 'scale', 'desc_mass', 'mmp', 'mvir'], ascending=[False,False,True, True, True, True], inplace=True)

for fn in tqdm(np.arange(len(trees_path)), desc='Writing iso trees'):
    fn_mask = iso_trees['FileNumber'] == fn
    file_path = fbase_out + '.{:d}'.format(fn)
    hout = iso_trees.loc[fn_mask]
    roots = hout['rootid'].value_counts().loc[hout.loc[hout.scale==1]['rootid'].index.values]
    Ntrees = len(roots)
    Nhalos = roots.values
    TreeID = roots.index.values - 1
    gb.write_halo_trees(file_path=file_path, tree=hout[cout], Ntrees=Ntrees, Nhalos=Nhalos, TreeID=TreeID, file_format='EMERGE')

print('Writing new `.forests` file.')
iso_forest = iso_trees.loc[iso_trees.rootid.unique()][['rootid']].reset_index().to_numpy()
np.savetxt(fbase_out+'.forests', iso_forest, delimiter=' ', fmt = '%.d', header='TreeRootID ForestID')

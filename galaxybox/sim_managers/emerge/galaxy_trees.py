"""Classes for handling Emerge galaxy merger tree data."""
from ...io.emerge_io import read_tree
from ...helper_functions import arg_parser
from ...mock_observables import lightcone

from astropy import cosmology as apcos
from astropy import constants as apconst

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import os
import h5py
import warnings

__author__ = ('Joseph O\'Leary', )


class galaxy_trees:
    """A class for reading in and operating on Emerge galaxy merger trees
    alone or within the context of a Universe.
    """

    def __init__(self, trees_path, add_attrs=None, fields_out=None):
        """Initialize a galaxy_trees object

        Parameters
        ----------
        trees_path : string or list of strings
            A list of tree files to be included into the class.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None
        fields_out : list of strings, optional
            tree columns to be read in, by default None
        """
        self.trees_used = trees_path

        if isinstance(trees_path, str):
            trees_path = [trees_path]

        self.trees_used = sorted(trees_path)

        frames = [None] * len(self.trees_used)
        for i, file in enumerate(tqdm(self.trees_used, desc='Loading galaxy trees')):
            frames[i] = read_tree(file, fields_out=fields_out)
        forest = pd.concat(frames)

        forest.set_index('ID', drop=True, inplace=True)
        self.trees = forest
        self.scales = self.trees.Scale.unique()[::-1]
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])

        self.snapshots = []

        self.__galaxies = {}
        for i in range(len(self.scales)):
            self.snapshots.append('S{:d}'.format(i))
            self.__galaxies['S{:d}'.format(i)] = np.logical_and(self.trees['Scale'] == self.scales[i], self.trees['Stellar_mass'] >= self.OutputMassThreshold)

        self.__roots_mask = self.trees['Scale'] == self.trees['Scale'].max()
        self.__roots_idx = pd.DataFrame(index=self.trees.loc[self.__roots_mask].index)
        self.__roots_idx['root_idx'] = np.argwhere(self.__roots_mask.values).flatten()
        self.__roots_idx['leaf_idx'] = self.__roots_idx['root_idx'] + np.append(np.diff(self.__roots_idx['root_idx']), (len(self.trees) - 1 - self.__roots_idx['root_idx'].iloc[-1]))
        self.sorted = False
        if 'Leaf_ID' in self.trees.keys():
            self.__leaf = True
        else:
            self.__leaf = False

    def __repr__(self):
        """pass through the pandas dataframe __repr__"""
        #TODO: there is likely a more informative usage of this method.
        return repr(self.trees)

    #TODO: the `add` approach for class start up is bad. switch to using properties...
    @classmethod
    def from_universe(cls, Universe, **kwargs):
        """Initiate a `galaxy_tree` within the `Universe` class

        Parameters
        ----------
        Universe : `galaxybox.Universe`
            The outer universe class used to organize these galaxy trees

        Returns
        -------
        galaxybox.galaxy_trees
            galaxy trees set from the universe
        """
        # this is a terrible approach.
        try:
            kwargs
        except:
            kwargs={}

        kwargs['add_attrs'] = {'out_dir': Universe.out_dir,
               'fig_dir': Universe.fig_dir,
               'NumFiles': Universe.params.get_param('NumFilesInParallel'),
               'ModelName': Universe.params.get_param('ModelName'),
               'BoxSize': Universe.params.get_param('BoxSize'),
               'UnitTime_in_yr': Universe.params.get_param('UnitTime_in_yr'),
               'Fraction_Escape_ICM': Universe.params.get_param('Fraction_Escape_ICM'),
               'OutputMassThreshold': Universe.params.get_param('OutputMassThreshold'),
               'TreeRootRedshift': Universe.params.get_param('TreeRootRedshift'),
               'cosmology': Universe.cosmology}
        trees_path = []
        for i in range(Universe.params.get_param('NBoxDivisions')**3):
            if Universe.params.get_param('OutputFormat') == 1:
                trees_path.append(os.path.join(Universe.out_dir, 'trees/tree.{:d}.out'.format(i)))
            else:
                trees_path.append(os.path.join(Universe.out_dir, 'trees/tree.{:d}.h5'.format(i)))
        return cls(trees_path, **kwargs)

    @classmethod
    def from_file(cls, tree_dir, fields_out=None):
        """Initiate a `galaxy_tree` by pointing to the directory containg the trees.

        class attributes will be set using parameters shown in the tree.0.h5 file.
        Currently only hdf5 format trees are supported

        Parameters
        ----------
        tree_dir : str
            path to the directory containing the galaxy trees.
        fields_out : list of strings, optional
            the columns to be read in from the tree files, by default None

        Returns
        -------
        galaxy_tree
            A galaxy tree object
        """
        # TODO: this whole method needs some cleanup
        # TODO: set this to work with ascii trees aswell
        tree_dir = os.path.abspath(tree_dir)
        f = os.path.join(tree_dir,'tree.0.h5') #! Having this hardcoded is a terrible idea...I'll deal with it later.
        keys = h5py.File(f, 'r').attrs.keys()
        values = h5py.File(f, 'r').attrs.values()
        attrs = dict(zip(keys, values))
        cosmo = apcos.LambdaCDM(H0=attrs['HubbleParameter'] * 100,
                                        Om0=attrs['Omega_0'],
                                        Ode0=attrs['Omega_Lambda'],
                                        Ob0=attrs['Omega_Baryon'])
        add = {'out_dir': tree_dir,
               'fig_dir': tree_dir,
               'NumFiles': attrs['NTreeFiles'],
               'ModelName': attrs['Model Name'].astype(str),
               'BoxSize': attrs['BoxSize']*attrs['HubbleParameter'],
               'UnitTime_in_yr': 1000000000.0, #! Having this hardcoded is a terrible idea...I'll deal with it later.
               'OutputMassThreshold': attrs['Tree_root_mass'],
               'TreeRootRedshift': attrs['Tree_root_redshift'],
               'cosmology': cosmo}
        trees_path = []
        for i in range(attrs['NTreeFiles']):
                trees_path.append(os.path.join(tree_dir,'tree.{:d}.h5'.format(i)))
        return cls(trees_path, add_attrs=add, fields_out=fields_out)

    def climb_tree(self, igal, idx=0, G_mask=None, main_branch=True, progenitors=False):
        """Recursive function for climbing merger trees

        Parameters
        ----------
        igal : int
            The ID of the root galaxy in the tree.
        idx : int
            Current linear index in the tree (the order in which the tree was walked).
        G_mask : type
            Description of parameter `G_mask` (the default is None).
        main_branch : bool
            Whether to traverse the main branch during tree climb (the default is True).
        progenitors : bool
            Whether to traverse the coprogenitors during the tree climb (the default is False).

        Returns
        -------
        idx : int
            Current linear index in the tree (the order in which the tree was walked).
        G_mask

        """
        idx += 1
        if G_mask is not None:
            G_mask[idx - 1] = igal

        if progenitors:
            if self.trees.loc[igal]['Coprog_ID'] > 0:
                idx, G_mask = self.climb_tree(int(self.trees.loc[igal]['Coprog_ID']), idx=idx, G_mask=G_mask, main_branch=main_branch, progenitors=progenitors)
        if main_branch:
            if self.trees.loc[igal]['MMP_ID'] > 0:
                idx, G_mask = self.climb_tree(int(self.trees.loc[igal]['MMP_ID']), idx=idx, G_mask=G_mask, main_branch=main_branch, progenitors=progenitors)

        return idx, G_mask

    def tree(self, igal, root=False, unsort=False):
        """Return the merger tree for a galaxy with ID `igal`.

        Parameters
        ----------
        igal : int
            The ID of the root galaxy in the tree.
        root : bool
            Whether this galaxy is at the file root output redshift. (the default is False).
        unsort : bool
            If using sorted IDs, setting unsort true will return the tree sorted by descending scalefactor (the default is False).

        Returns
        -------
        tree : pandas.DataFrame
            Merger tree

        """
        if self.__leaf:
            if self.sorted:
                if unsort:
                    return self.trees.loc[igal:self.trees.loc[igal]['Leaf_ID'].astype(int)].sort_values('Scale', ascending=False)
                else:
                    return self.trees.loc[igal:self.trees.loc[igal]['Leaf_ID'].astype(int)]
            else:
                return self.trees.loc[np.arange(igal, self.trees.loc[igal]['Leaf_ID'].astype(int) + 1)]
        else:
            if root:
                return self.trees.iloc[self.__roots_idx.loc[igal]['root_idx']:self.__roots_idx.loc[igal]['leaf_idx']]
            else:
                N, _ = self.climb_tree(igal, main_branch=True, progenitors=True)
                mask = np.zeros(N).astype(int)
                N, mask = self.climb_tree(igal, G_mask=mask, main_branch=True, progenitors=True)

                return self.trees.loc[mask]

    def main_branch(self, igal, root=False):
        """Return the main evolutionary branch for a galaxy with ID `igal`.

        Parameters
        ----------
        igal : int
            The ID of the galaxy in the tree.
        root : bool
            Whether this galaxy is at the file root output redshift. (the default is False).

        Returns
        -------
        pandas.Dataframe
            Main branch
        """
        if self.__leaf:
            tree = self.tree(igal)
            # Only include most massive progenitors
            mmp_mask = (tree['MMP'] == 1) | (tree['Scale'] == tree['Scale'].max())
            mb = tree.loc[mmp_mask]

            # initialize a mask
            mask = np.full(len(mb), False)
            desc = igal

            # iterate over the mmp tree finding the mmp route leading to the root galaxy
            i = 0
            for row in mb.to_records():
                if i == 0:
                    mask[i] = True
                else:
                    r_id = row['Desc_ID']
                    if r_id == desc:
                        mask[i] = True
                        desc = row['ID']
                i += 1
            return mb[mask]
        else:
            if root:
                tree = self.tree(igal, root=True)
                # Only include most massive progenitors
                mmp_mask = (tree['MMP'] == 1) | (tree['Scale'] == tree['Scale'].max())
                mb = tree.loc[mmp_mask]

                # initialize a mask
                mask = np.full(len(mb), False)
                desc = igal

                # iterate over the mmp tree finding the mmp route leading to the root galaxy
                i = 0
                for row in mb.to_records():
                    if i == 0:
                        mask[i] = True
                    else:
                        r_id = row['Desc_ID']
                        if r_id == desc:
                            mask[i] = True
                            desc = row['ID']
                    i += 1
                return mb[mask]
            else:
                N, _ = self.climb_tree(igal, main_branch=True, progenitors=False)
                mask = np.zeros(N).astype(int)
                N, mask = self.climb_tree(igal, G_mask=mask, main_branch=True, progenitors=False)
                return self.trees.loc[mask]

    def main_branch_progenitors(self, igal, mmp=True, sat=True, **kwargs):
        """Return the progenitors of all mergers occuring along the main branch of galaxy with ID `igal`.

        Parameters
        ----------
        igal : int
            The ID of the galaxy in the tree.
        mmp : bool, optional
            if True return most massive progenitors along main branch, default True
        sat : bool, optional
            if True return satellites merging onto the main branch, default True

        Returns
        -------
        pandas.DataFrame
            Progenitor galaxies along main branch

        """
        tree = self.tree(igal, **kwargs)
        mb = self.main_branch(igal, **kwargs)
        main_desc = mb.loc[mb['Num_prog'] > 1]
        if mmp and sat:
            return tree.loc[tree['Desc_ID'].isin(main_desc.index)]
        elif mmp:
            return tree.loc[(tree['Desc_ID'].isin(main_desc.index)) & (tree['MMP'] == 1)]
        elif sat:
            return tree.loc[(tree['Desc_ID'].isin(main_desc.index)) & (tree['MMP'] == 0)]
        else:
            raise Exception('must specify atleast one progenitor type.')

    def mass_weighted_mass_ratio(self, igal, **kwargs):
        """Compute the mass-weighted mass-ratio for galaxy with ID `igal`

        Parameters
        ----------
        igal : int
            The ID of the galaxy in the tree.

        Returns
        -------
        float
            The mass weighted mass ratio
        """
        progs = self.main_branch_progenitors(igal, **kwargs)
        sats = progs.loc[progs['MMP'] == 0]
        mains = progs.loc[progs['MMP'] == 1]
        numerator = 0
        denominator = 0
        for i, gal in enumerate(sats.to_records(index=False)):
            sat_mass = 10**gal['Stellar_mass']
            numerator += ((sat_mass)**2) / (10**np.float(mains.loc[mains['Desc_ID'] == gal['Desc_ID']]['Stellar_mass']))
            denominator += sat_mass

        if numerator == 0 or denominator == 0:
            return np.nan
        else:
            return float(1 / (numerator / denominator))

    def exsitu_mass(self, igal, frac=None, min_MR=1, max_MR=np.inf, **kwargs):
        """Compute the exsitu mass fraction at the root redshift"""
        progs = self.main_branch_progenitors(igal, **kwargs)
        sats = progs.loc[progs['MMP'] == 0]
        mains = progs.loc[progs['MMP'] == 1]
        exsitu_mass = 0
        for i, gal in enumerate(sats.to_records(index=False)):
            sat_mass = 10**gal['Stellar_mass']
            main_mass = 10**np.float(mains.loc[mains['Desc_ID'] == gal['Desc_ID']]['Stellar_mass'])
            ratio = np.maximum(main_mass / sat_mass, sat_mass / main_mass)
            if (ratio >= min_MR) and (ratio < max_MR):
                exsitu_mass += (10**gal['Stellar_mass_root']) * (1 - self.Fraction_Escape_ICM)

        if frac is None:
            return exsitu_mass, np.sum((10**sats['Stellar_mass_root']) * (1 - self.Fraction_Escape_ICM)), (10**self.trees.loc[igal]['Stellar_mass'])
        elif frac == 'total':
            return exsitu_mass / (10**self.trees.loc[igal]['Stellar_mass'])
        elif frac == 'exsitu':
            return exsitu_mass / np.sum((10**sats['Stellar_mass_root']) * (1 - self.Fraction_Escape_ICM))
        else:
            raise Exception('frac must be either "total" for total stellar mass, or "exsitu" for total exsitu mass.')

    def count_mergers(self, igal, min_MR=1, max_MR=np.inf, min_z=0, max_z=np.inf, **kwargs):
        """Count the number of mergers along the main branch of galaxy with  ID `igal`

        Parameters
        ----------
        igal : int
            The ID of the galaxy in the tree.
        min_MR : float, optional
            The minimum mass ratio to consider for mergers, by default 1
        max_MR : float, optional
            The maximum mass ratio to consider for mergers, by default np.inf
        min_z : int, optional
            The minimum redshift to consider for mergers, by default 0
        max_z : float, optional
            The maximum redshift to consider for mergers, by default np.inf

        Returns
        -------
        N_mergers : int
            Number of mergers
        """
        mb = self.main_branch(igal)
        progs = self.main_branch_progenitors(igal, **kwargs)
        sats = progs.loc[progs['MMP'] == 0]
        mains = progs.loc[progs['MMP'] == 1]

        # find the min time to consider
        if isinstance(max_z, (float, int)):
            min_time = self.cosmology.age(max_z).value
        else:
            color_mask = (np.log10(mb['SFR']) - mb['Stellar_mass']) < np.log10(0.3 / mb['Age'] / self.UnitTime_in_yr)
            first_quench = np.where(color_mask)[0][-1]
            last_quench = np.where(~color_mask)[0][0] - 1
            if max_z == 'first_q':
                min_time = mb.iloc[first_quench]['Age']
            else:
                min_time = mb.iloc[last_quench]['Age']

        if isinstance(min_z, (float, int)):
            if min_z == 0:
                max_time = np.inf
            else:
                max_time = self.cosmology.age(min_z).value
        else:
            if 'color_mask' not in locals():
                color_mask = (np.log10(mb['SFR']) - mb['Stellar_mass']) < np.log10(0.3 / mb['Age'] / self.UnitTime_in_yr)
                first_quench = np.where(color_mask)[0][-1]
                last_quench = np.where(~color_mask)[0][0] - 1
            if min_z == 'first_q':
                max_time = mb.iloc[first_quench]['Age']
            else:
                max_time = mb.iloc[last_quench]['Age']

        N_mergers = 0
        for i, gal in enumerate(sats.to_records(index=False)):
            sat_mass = 10**gal['Stellar_mass']
            main_mass = 10**np.float(mains.loc[mains['Desc_ID'] == gal['Desc_ID']]['Stellar_mass'])
            ratio = np.maximum(main_mass / sat_mass, sat_mass / main_mass)
            if (ratio >= min_MR) and (ratio < max_MR) and (gal['tdf'] >= min_time) and (gal['tdf'] < max_time):
                N_mergers += 1

        return N_mergers

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
        # first lets make all columns case insensitive
        colnames = list(self.trees.keys())
        col_alias = {}
        for k in colnames:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias['Scale'] += ['a', 'scale_factor']
        col_alias['redshift'] = ['redshift', 'z']
        col_alias['snapshot'] = ['snapshot', 'snapnum', 'snap']
        col_alias['color'] = ['color']
        col_alias['color_obs'] = ['color_obs', 'col_obs', 'color_observed']
        col_alias['Stellar_mass'] += ['mass', 'mstar']
        col_alias['Halo_mass'] += ['mvir']
        col_alias['Type'] += ['gtype']
        col_alias['Up_ID'] += ['ihost', 'host_id', 'id_host']
        col_alias['Desc_ID'] += ['idesc', 'id_desc']
        col_alias['Main_ID'] += ['imain', 'id_main']
        col_alias['MMP_ID'] += ['immp', 'id_mmp']
        col_alias['Coprog_ID'] += ['icoprog', 'id_coprog']
        col_alias['Leaf_ID'] += ['ileaf', 'id_leaf']
        col_alias['Original_ID'] += ['ogid', 'rockstar_id', 'id_rockstar', 'id_original', 'rs_id', 'id_rs', 'irs', 'rsid']
        col_alias['Num_prog'] += ['np']
        if 'Intra_cluster_mass' in colnames: col_alias['Intra_cluster_mass'] += ['icm']
        col_alias['Stellar_mass_root'] += ['mstar_root', 'root_mstar', 'rootmass', 'root_mass']
        col_alias['Stellar_mass_obs'] += ['mstar_obs']
        col_alias['Halo_radius'] += ['rvir', 'virial_radius', 'radius']

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError('`{}` has no known alias.'.format(key))

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
            key = keyl.lower().replace('min', '').replace('max', '').lstrip('_').rstrip('_')
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        # I dont like the implementation of `all` right now.
        # Allowing for specified ranges would be better.
        if 'snapshot' in kwargs.keys():
            if kwargs['snapshot'] == 'all':
                galaxy_list = self.trees
            else:
                galaxy_list = self.trees.loc[self.__galaxies[kwargs['snapshot']]]
            kwargs.pop('snapshot')
        elif 'Scale' in kwargs:
            if kwargs['Scale'] == 'all':
                galaxy_list = self.trees
            else:
                scale_arg = (np.abs(self.scales - kwargs['Scale'])).argmin()
                s_key = self.snapshots[scale_arg]
                galaxy_list = self.trees.loc[self.__galaxies[s_key]]
            kwargs.pop('Scale')
        elif 'redshift' in kwargs:
            if kwargs['redshift'] == 'all':
                galaxy_list = self.trees
            else:
                scale_factor = 1 / (kwargs['redshift'] + 1)
                scale_arg = (np.abs(self.scales - scale_factor)).argmin()
                s_key = self.snapshots[scale_arg]
                galaxy_list = self.trees.loc[self.__galaxies[s_key]]
            kwargs.pop('redshift')
        else:
            s_key = self.snapshots[-1]
            galaxy_list = self.trees.loc[self.__galaxies[s_key]]

        # Setup a default `True` mask
        mask = galaxy_list['Scale'] > 0
        # Loop of each column in the tree and check if a min/max value mask should be created.
        for i, key in enumerate(self.trees.keys()):
            for j, kw in enumerate(kwargs.keys()):
                if ('obs' in kw.lower()) & ('obs' not in key.lower()):
                    pass
                elif key.lower() in kw.lower():
                    if 'min' in kw.lower():
                        mask = mask & (galaxy_list[key] >= kwargs[kw])
                    elif 'max' in kw.lower():
                        mask = mask & (galaxy_list[key] < kwargs[kw])
                    else:
                        values = np.atleast_1d(kwargs[kw])
                        # Setup a default `False` mask
                        sub_mask = galaxy_list['Scale'] > 1
                        for v in values:
                            sub_mask = sub_mask | (galaxy_list[key] == v)
                        mask = mask & sub_mask
        # Create masks for derived quantities such as `color`.
        if 'color' in kwargs.keys():
            if kwargs['color'].lower() == 'blue':
                mask = mask & ((np.log10(galaxy_list['SFR']) - galaxy_list['Stellar_mass']) >= np.log10(0.3 / galaxy_list['Age'] / self.UnitTime_in_yr))
            elif kwargs['color'].lower() == 'red':
                mask = mask & ((np.log10(galaxy_list['SFR']) - galaxy_list['Stellar_mass']) < np.log10(0.3 / galaxy_list['Age'] / self.UnitTime_in_yr))
        if 'color_obs' in kwargs.keys():
            if kwargs['color_obs'].lower() == 'blue':
                mask = mask & ((np.log10(galaxy_list['SFR_obs']) - galaxy_list['Stellar_mass_obs']) >= np.log10(0.3 / galaxy_list['Age'] / self.UnitTime_in_yr))
            elif kwargs['color_obs'].lower() == 'red':
                mask = mask & ((np.log10(galaxy_list['SFR_obs']) - galaxy_list['Stellar_mass_obs']) < np.log10(0.3 / galaxy_list['Age'] / self.UnitTime_in_yr))

        if mask_only:
            return mask
        else:
            return galaxy_list.loc[mask]

    def count(self, target_scales=None, dtype=int, **kwargs):
        """Count the number of galaxies at specified scalefactor(s).

        Parameters
        ----------
        target_scales : float, list of floats, optional
            Scale factors at which a galaxy count should be performed, by default None
        dtype : data type, optional
            interpolated values will be cast into this type, by default int

        Returns
        -------
        dtype, numpy array of dtype
            The number of galaxies at each input scale factor
        """
        counts = np.zeros(len(self.scales))
        argsin = []

        N_gal = self.list(z='all', **kwargs)['Scale'].value_counts().sort_index()

        for i, a in enumerate(self.scales):
            if np.isin(a, N_gal.index):
                argsin += [i]
        counts[argsin] = N_gal.values
        if target_scales is None:
            target_scales = self.scales
        target_scales = np.atleast_1d(target_scales)
        func = interp1d(self.scales, counts)
        return func(target_scales).astype(dtype)

    def find_mergers(self, enforce_mass_threshold=True, enforce_positive_MR=True, use_obs=False, desc_mass_correction=False, dropna=True):
        """Create a list of all galaxy mergers in the tree in terms of descendant, main progenitor, minor progenitor and compute the mass ratio.

        Parameters
        ----------
        enforce_mass_threshold : bool, optional
            Exclude mergers where any galaxy component is below the massthreshold set in the parameterfile, by default True
        enforce_positive_MR : bool, optional
            swap main and minor galaxy progenitors to ensure mass rato > 1, by default True
        use_obs : bool, optional
            whether to use quantities with observational scatter as the default, by default False
        desc_mass_correction : bool, optional
            correct descendant mass in the case of mergers being non-binary, by default False
        dropna : bool, optional
            Drop rows that contain nan, by default True

        Returns
        -------
        pandas.DataFame
            DataFrame of galaxy mergers
        """
        print('Making merger list from trees')
        if use_obs:
            add_obs = '_obs'
        else:
            add_obs = ''
        # Initialise dataframe of merging events
        mergers = pd.DataFrame(columns=['Scale', 'tdf', 'Desc_ID', 'Desc_mvir', 'Desc_mstar', 'Desc_sfr', 'Desc_mstar_obs', 'Desc_sfr_obs',
                                        'Desc_type', 'Main_ID', 'Main_mvir', 'Main_mstar', 'Main_sfr', 'Main_mstar_obs', 'Main_sfr_obs', 'Main_type',
                                        'Minor_ID', 'Minor_mvir', 'Minor_mstar', 'Minor_sfr', 'Minor_mstar_obs', 'Minor_sfr_obs', 'Minor_type', 'MR', 'MR_obs'])

        # find merging systems
        minor_progs = self.trees.loc[(self.trees['Flag'] == 1)]
        missing_desc = minor_progs.loc[~minor_progs.Desc_ID.isin(self.trees.index.values).values]
        if len(missing_desc) > 0:
            # this can happen close to the mass threshold or if the main progenitor is stripped in the next timestep.
            warnings.warn('Could not find descendant for galaxies with index {}'.format(missing_desc.index.values))
        minor_progs = minor_progs.loc[minor_progs.Desc_ID.isin(self.trees.index.values).values]
        mergers[['Desc_ID', 'Minor_mvir', 'Minor_mstar', 'Minor_sfr', 'Minor_mstar_obs', 'Minor_sfr_obs', 'tdf', 'Minor_type']] = self.trees.loc[minor_progs.index.values][['Desc_ID', 'Halo_mass', 'Stellar_mass', 'SFR', 'Stellar_mass_obs', 'SFR_obs', 'tdf', 'Type']].copy()
        mergers['Minor_ID'] = mergers.index.values
        mergers.reset_index(inplace=True, drop=True)

        # Find properties of descendent galaxy
        mergers[['Scale', 'Desc_mvir', 'Desc_mstar', 'Desc_sfr', 'Desc_mstar_obs', 'Desc_sfr_obs', 'Desc_type']] = self.trees.loc[mergers['Desc_ID'].values][['Scale', 'Halo_mass', 'Stellar_mass', 'SFR', 'Stellar_mass_obs', 'SFR_obs', 'Type']].values
        mergers['Main_ID'] = self.trees.loc[mergers['Desc_ID'].values]['MMP_ID'].values

        # Find Main progenitor properties
        mergers[['Main_mvir', 'Main_mstar', 'Main_sfr', 'Main_mstar_obs', 'Main_sfr_obs', 'Main_type']] = self.trees.loc[mergers['Main_ID'].values][['Halo_mass', 'Stellar_mass', 'SFR', 'Stellar_mass_obs', 'SFR_obs', 'Type']].values

        # drop rows with NAN (sometimes happens when looking for mergers near lower mass cut)
        if dropna:
            mergers.dropna(subset=mergers.columns[:-2], inplace=True)
        mergers.reset_index(inplace=True, drop=True)

        # Correct the descendent mass for mergers that arent binary in the galaxy merger tree.
        # TODO: This really needs a speed boost its slow and sloppy.
        if desc_mass_correction:
            def nonbinary_correction(index):
                co_progs_mask = (mergers['Desc_ID'] == mergers.loc[index].Desc_ID) & (mergers.index != index) & (mergers['tdf'] >= mergers.loc[index].tdf)
                return np.log10(10**mergers.loc[index].Desc_mstar - (np.sum(10**mergers.loc[co_progs_mask]['Minor_mstar'])) * (1 - self.Fraction_Escape_ICM))

            duplicate_ids = mergers[mergers['Desc_ID'].duplicated()]
            non_binary = mergers.loc[mergers.Desc_ID.isin(duplicate_ids.Desc_ID.unique())]
            for row in non_binary.itertuples():
                mergers.at[row.Index, 'Desc_mstar'] = nonbinary_correction(row.Index)

        # drop systems below mass threshold
        if enforce_mass_threshold:
            low_mass_mask = (mergers['Main_mstar' + add_obs] < self.OutputMassThreshold) | (mergers['Minor_mstar' + add_obs] < self.OutputMassThreshold) | (mergers['Desc_mstar' + add_obs] < self.OutputMassThreshold)
            mergers.drop(mergers[low_mass_mask].index, inplace=True)

        # Enforce M1 >= M2
        if enforce_positive_MR:
            swap = (mergers['Main_mstar' + add_obs] < mergers['Minor_mstar' + add_obs])
            mergers.loc[swap, ['Main_ID', 'Main_mvir', 'Main_mstar', 'Main_sfr', 'Main_mstar_obs', 'Main_sfr_obs', 'Main_type',
                               'Minor_ID', 'Minor_mvir', 'Minor_mstar', 'Minor_sfr', 'Minor_mstar_obs', 'Minor_sfr_obs', 'Minor_type']] = mergers.loc[swap, ['Minor_ID', 'Minor_mvir', 'Minor_mstar', 'Minor_sfr', 'Minor_mstar_obs', 'Minor_sfr_obs', 'Minor_type',
                                                                                                                                                            'Main_ID', 'Main_mvir', 'Main_mstar', 'Main_sfr', 'Main_mstar_obs', 'Main_sfr_obs', 'Main_type']].values
        # compute mass ratio
        mergers['MR'] = 10**(mergers['Main_mstar'] - mergers['Minor_mstar'])
        mergers['MR_obs'] = 10**(mergers['Main_mstar_obs'] - mergers['Minor_mstar_obs'])

        return mergers

    def sort_index(self):
        """Sort galaxy trees by ID (ascending)

        ID sorting allows for faster retreival of individual galaxy trees/growth history.
        Sorting is particularly helpful for galaxies that arent a root.
        """
        if self.__leaf:
            self.trees.sort_index(inplace=True)
            self.sorted = True
            del self.__roots_idx
            self.__roots_mask = self.trees['Scale'] == self.trees['Scale'].max()
            self.__roots_idx = pd.DataFrame(index=self.trees.loc[self.__roots_mask].index)
            self.__roots_idx['root_idx'] = np.argwhere(self.__roots_mask.values).flatten()
            self.__roots_idx['leaf_idx'] = self.__roots_idx['root_idx'] + np.append(np.diff(self.__roots_idx['root_idx']), (len(self.trees) - 1 - self.__roots_idx['root_idx'].iloc[-1]))

            del self.__galaxies
            self.__galaxies = {}
            for i in range(len(self.scales)):
                self.__galaxies['S{:d}'.format(i)] = np.logical_and(self.trees['Scale'] == self.scales[i], self.trees['Stellar_mass'] >= self.OutputMassThreshold)
        else:
            print('...Im not going to sort because you dont have leaves, so you probably dont have the correct ID format....so there is nothing to gain from a sort')

    def make_lightcone_catalog(self, min_z=None, max_z=None, randomize=False, seed=None, fuzzy_bounds=False, method='KW07', lean=True, **kwargs):
        """Construct a galaxy catalog with lightcone geometry.

        Parameters
        ----------
        min_z : float, optional
            Minimum galaxy redshift, by default None
        max_z : float, optional
            Minimum galaxy redshift, by default None
        randomize : bool, optional
            If True box tesellations will have a random axis rotation and translation applied before selecting galaxies, by default False
        seed : int, optional
            Randomization seed, by default None
        fuzzy_bounds : bool, optional
            If `True` satellite galaxies extending beyond snapshot range will be included if their host is within the snapshot range., by default False
        method : str, optional
            Lightcone construction method, by default 'KW07'
        lean : bool, optional
            If True only a limited data set is returned that should be linked to the galaxy trees, by default True

        Returns
        -------
        pandas.DataFrame
            A galaxy catalog fitting into the geometry of the specified lightcone
        
        galaxybox.mock_observables.lightcone.lightcone
            The lightcone object
        tuple
            A tuple containing the min_z, max_z arguments
        """
        #TODO: This entire method could use some reworking and cleanup.
        if method == 'KW07':
            LC_kwargs, kwargs = arg_parser(lightcone.KW07, drop=True, **kwargs)
            LC = lightcone.KW07(**LC_kwargs, Lbox=self.BoxSize / self.cosmology.h)
        elif method == 'full_width':
            LC_kwargs, kwargs = arg_parser(lightcone.full_width, drop=True, **kwargs)
            LC = lightcone.full_width(Lbox=self.BoxSize / self.cosmology.h, **LC_kwargs)
        elif method == 'hybrid':
            LC_kwargs, kwargs = arg_parser(lightcone.hybrid, drop=True, **kwargs)
            LC = lightcone.hybrid(Lbox=self.BoxSize / self.cosmology.h, **LC_kwargs)
        else:
            raise NotImplementedError('Only the \'KW07\', \'full_width\' and \'hybrid\' cone methods have been implemented at this time')
        
        if seed is not None:
            LC.set_seed(seed)

        D = self.cosmology.comoving_distance(1 / self.scales - 1).value[::-1]
        LC.set_snap_distances(D)
        if max_z is None:
            max_z = 1 / self.scales.min() - 1
        if min_z is None:
            min_z = self.TreeRootRedshift

        z = np.linspace(min_z,max_z,1000000)
        d = self.cosmology.comoving_distance(z).value
        redshift_at_D = interp1d(d,z)

        D_min = self.cosmology.comoving_distance(min_z).value
        D_max = self.cosmology.comoving_distance(max_z).value
        vert = LC.tesselate(D_min, D_max)
        snapshots = self.snapshots[::-1]
        redshifts = 1 / self.scales[::-1] - 1

        # loop over all tesselations that intersect lightcone
        for i, og in enumerate(tqdm(vert)):
            snap_arg = LC.get_snapshots(og)
            bc = LC.get_boxcoord(og)

            # loop over all snapshots available to this volume
            for j, sa in enumerate(snap_arg):
                smin, smax = LC.snapshot_extent(sa)
                if lean:
                    # if lean is true, only save the minmumn information necessary
                    # TODO: this could be cut down even futher but I'll deal with that later.
                    galaxies = self.list(snapshot=snapshots[sa], **kwargs)[['Scale', 'Up_ID', 'Halo_radius', 'X_pos', 'Y_pos', 'Z_pos', 'X_vel', 'Y_vel', 'Z_vel', 'Type']].copy()
                else:
                    galaxies = self.list(snapshot=snapshots[sa], **kwargs).copy()
                
                # if not galaxies then theres nothing to do
                if len(galaxies) == 0:
                    continue

                galaxies[['X_pos', 'Y_pos', 'Z_pos']] = LC.transform_position(pos=galaxies[['X_pos', 'Y_pos', 'Z_pos']], randomize=randomize, box_coord=bc)
                if randomize:
                    galaxies[['X_vel', 'Y_vel', 'Z_vel']] = LC.transform_velocity(vel=galaxies[['X_vel', 'Y_vel', 'Z_vel']], box_coord=bc)
                
                mask = LC.contained(galaxies[['X_pos', 'Y_pos', 'Z_pos']], mask_only=True)
                galaxies = galaxies.loc[mask]
                galaxies[['Redshift', 'Redshift_obs', 'X_cone', 'Y_cone', 'Z_cone', 'X_cvel', 'Y_cvel', 'Z_cvel', 'RA', 'Dec']] = pd.DataFrame(columns=['Redshift', 'Redshift_obs', 'X_cone', 'Y_cone', 'Z_cone', 'X_cvel', 'Y_cvel', 'Z_cvel', 'RA', 'Dec'])
                galaxies[['X_cone', 'Y_cone', 'Z_cone']] = LC.cone_cartesian(galaxies[['X_pos', 'Y_pos', 'Z_pos']])
                glist = galaxies.copy()  # everything at this snapshot
                # cut out the right radial extent
                mask = (galaxies['Z_cone'] >= np.max([smin, D_min])) & (galaxies['Z_cone'] < np.min([smax, D_max]))
                galaxies = galaxies.loc[mask]

                # added satellites that break the border
                if fuzzy_bounds & (len(galaxies) > 0):
                    mains = galaxies.loc[galaxies['Type'] == 0]
                    # check if there are any main galaxies
                    if len(mains) > 0:
                        mains = mains.loc[(mains['Z_cone'].values + mains['Halo_radius'].values) > smax]
                        # do any of these extend over the boundary
                        if len(mains) > 0:
                            sats = glist.loc[glist['Up_ID'].isin(mains.index.values)]
                            mask = (sats['Z_cone'] >= smax) & (sats['Z_cone'] < D_max)
                            # Add those satellites to the galaxy list.
                            galaxies = pd.concat([galaxies, sats.loc[mask]])

                # Add columnes for lightcone coordinates and apparent redshift
                galaxies[['RA', 'Dec']] = LC.ang_coords(galaxies[['X_pos', 'Y_pos', 'Z_pos']])
                galaxies[['X_cvel', 'Y_cvel', 'Z_cvel']] = LC.cone_cartesian(galaxies[['X_vel', 'Y_vel', 'Z_vel']])
                galaxies['Redshift'] = galaxies['Z_cone'].apply(redshift_at_D).values
                galaxies['Redshift_obs'] = galaxies['Redshift'] + galaxies['Z_cvel'] * (1 + galaxies['Redshift']) / apconst.c.to('km/s').value

                if (i == 0) & (j == 0):
                    Lightcone_cat = galaxies
                else:
                    Lightcone_cat = pd.concat([Lightcone_cat, galaxies])

        Lightcone_cat.index.rename('ID', inplace=True) # this is a safety step incase the first loop turned out no galaxies
        Lightcone_cat.reset_index(inplace=True)
        Lightcone_cat.rename(columns={'ID': 'Tree_ID'}, inplace=True)
        if lean:
            return Lightcone_cat[['Tree_ID', 'Redshift', 'Redshift_obs', 'X_cone', 'Y_cone', 'Z_cone', 'X_cvel', 'Y_cvel', 'Z_cvel', 'RA', 'Dec']], LC, (min_z, max_z)
        else:
            return Lightcone_cat, LC, (min_z, max_z)

    def merging_time(self, igal_1, igal_2):
        """Find the cosmic time when two galaxies merge.

        Parameters
        ----------
        igal_1 : int
            ID of the first galaxy
        igal_2 : int
            ID of the second galaxy

        Returns
        -------
        float
            The cosmic time when the two input galaxies finally merge. Returns 0 if the galaxies dont ever merge.

        """
        if self.__leaf:
            # First we check if these two are in the same tree
            __roots = self.trees.loc[self.__roots_mask].index.values
            if not self.sorted:
                __roots = np.sort(__roots)

            tree_idx = np.searchsorted(__roots, [igal_1, igal_2], side='right') - 1

            # if they arent in the same tree we can stop here.
            if tree_idx[0] != tree_idx[1]:
                return 0
            else:
                gal_1 = self.trees.loc[igal_1]
                gal_2 = self.trees.loc[igal_2]
                
                if gal_1.Scale != gal_2.Scale:
                    # if these galaxies arent on the same timestep check if one is a progenitor of the other
                    if (gal_2.name > gal_1.name) and (gal_2.name <= gal_1.Leaf_ID):
                        # if its along the main branch return scale of gal_1
                        main_branch = self.main_branch(gal_1.name).index.values
                        if gal_2.name in main_branch:
                            return gal_1.Age
                        else: #otherwise evolve gal_2 until it merges
                            while gal_2.Flag == 0:
                                gal_2 = self.trees.loc[gal_2.Desc_ID]
                            return gal_2.tdf

                    elif (gal_1.name > gal_2.name) and (gal_1.name <= gal_2.Leaf_ID):
                        # if its along the main branch return scale of gal_2
                        main_branch = self.main_branch(gal_2.name).index.values
                        if gal_1.name in main_branch:
                            return gal_2.Age
                        else: #otherwise evolve gal_1 until it merges
                            while gal_1.Flag == 0:
                                gal_1 = self.trees.loc[gal_1.Desc_ID]
                            return gal_1.tdf
                        
                    else: #sync the time steps
                        if gal_1.Scale > gal_2.Scale:
                            while gal_1.Scale != gal_2.Scale:
                                gal_2 = self.trees.loc[gal_2.Desc_ID]

                        elif gal_2.Scale > gal_1.Scale:
                            while gal_1.Scale != gal_2.Scale:
                                gal_1 = self.trees.loc[gal_1.Desc_ID]
                
                # Just step forward until these galaxies have a common desc.
                while gal_1.Desc_ID != gal_2.Desc_ID:
                    if (gal_1.Flag==2) or (gal_2.Flag==2):
                        return 0
                    gal_1 = self.trees.loc[gal_1.Desc_ID]
                    gal_2 = self.trees.loc[gal_2.Desc_ID]
                
                # take the merging time from the infalling system.
                if gal_1.MMP == 0:
                    return gal_1.tdf
                elif gal_2.MMP == 0:
                    return gal_2.tdf
                elif (gal_1.MMP == 0) & (gal_2.MMP==0):
                    return max(gal_1.tdf, gal_2.tdf)
                else: #? Probably better raise an error in this case as something has broken.
                    return 0
        else:
            raise NotImplementedError("This method not currently availble for trees that haven't been reindexed.")

    def plot_tree(self, igal, ax=None, x_pos=0.0, min_scale=0.0, spacing=1.0, desc_pos=None, **kwargs):
            """Create a visual representation for galaxy tree growth.

            Parameters
            ----------
            igal : int
                ID of root galaxy
            ax : matplotlib ax object
                the axis on which the tree should be plotted
            x_pos : float, optional
                current x coordinate for the galaxy on the plot, by default 0.0
            min_scale : float, optional
                The minimum scale factor that should be plotted, by default 0.0
            spacing : float, optional
                x spacing between branches, by default 1.0
            desc_pos : list, optional
                the (x, y) coordinate of the descendant galaxy on the plot, by default None

            Returns
            -------
            x_pos : int, optional
                New x coordinate for the next galaxy on the plot
            """

            # some default plotting configs if no external ax is provided.
            if ax is None:
                vmin = self.OutputMassThreshold
                vmax = self.trees.Stellar_mass.max()
                kwargs['cmap'] = plt.cm.jet
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
                fig, ax = plt.subplots(figsize=(20,13))
                ax.set_ylabel('Scale factor', fontsize=18)
                ax.tick_params(axis='x',  labelbottom=False)
                axins = inset_axes(ax,
                                    width="100%",
                                    height="3%",
                                    loc='lower left',
                                    bbox_to_anchor=(0., 1.02, 1., .75),
                                    bbox_transform=ax.transAxes,
                                    borderpad=0)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                cbar = plt.colorbar(sm, cax=axins, orientation="horizontal")
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.minorticks_on()
                cbar.set_label('$\log_{10}(m/M_{\odot})$', fontsize=18)
                

            scale = self.trees.loc[igal]['Scale']
            # add this galaxy to the plot at position [x_pos, scale]
            ax.scatter([x_pos], [scale], c=[self.trees.loc[igal]['Stellar_mass']], **kwargs)

            if desc_pos is None:
                # if no desc gal then record the current galaxies position to desc_pos
                desc_pos = [x_pos, scale]
            else:
                # otherwise draw a line to the descendant galaxy
                ax.plot([desc_pos[0], x_pos], [desc_pos[1], scale],'k-',zorder=0)
                
            # get the ID of the most massive progenitor and any coprogenitors
            immp = int(self.trees.loc[igal]['MMP_ID'])
            icoprog = int(self.trees.loc[igal]['Coprog_ID'])
            
            if scale > min_scale:
                # walk the main branch first, update the desc_pos argument
                if immp > 0:
                    x_pos = self.plot_tree(immp, ax=ax, x_pos=x_pos, min_scale=min_scale, spacing=spacing, desc_pos=[x_pos,scale], **kwargs)
                # walk the coprogenitors, no update to desc_pos
                if icoprog > 0:
                    x_pos += spacing
                    x_pos = self.plot_tree(icoprog, ax=ax, x_pos=x_pos, min_scale=min_scale, spacing=spacing, desc_pos=desc_pos, **kwargs)
                    
            return x_pos

    def scale_at_massfrac(self, igal, frac, interpolate=False):
        """Determine the scalefactor when a galaxy's mass crossed some fraction of its current mass

        Parameters
        ----------
        igal : int or list of ints
            The ID of the galaxy in the tree.
        frac : float
            Target fraction of current galaxy mass
        interpolate : bool, optional
            If true mass growth is linearly interpolated between simulation time steps, by default False

        Returns
        -------
        list
            List of scale factors

        """
        #TODO: implement interpoliation option
        log_thresh = np.log10(frac)
        # get starting values
        dat = self.trees.loc[igal][['Scale', 'Stellar_mass', 'MMP_ID']].values
        scale, m0, progid = dat[:,0], dat[:,1], dat[:,2]
        log_frac = np.full(len(scale), 0.0)
        progid = progid.astype(int)
        prog_mask = progid > 0
        # while galaxies still have progenitors and they are above the threshold.
        while prog_mask.sum() > 0:
            log_frac[prog_mask] = self.trees.loc[progid[prog_mask]]['Stellar_mass'].values - m0[prog_mask]
            thresh_mask = (log_frac < log_thresh)
            
            # if threshold is crossed save the id and scale
            scale[prog_mask & ~thresh_mask] = self.trees.loc[progid[prog_mask & ~thresh_mask]]['Scale'].values
            igal[prog_mask & ~thresh_mask] = self.trees.loc[progid[prog_mask & ~thresh_mask]].index.values
            
            # otherwise update progs and move on
            progid[prog_mask] = self.trees.loc[progid[prog_mask]]['MMP_ID'].values.astype(int)
            progid[thresh_mask] = 0
            prog_mask = progid > 0

        if interpolate:
            raise NotImplementedError('Interpolated growth between timesteps not yet available')
        
        return scale
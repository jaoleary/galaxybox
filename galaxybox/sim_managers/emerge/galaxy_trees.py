"""Classes for handling Emerge galaxy merger tree data."""
from astropy import cosmology as apcos
from ...io.emerge_io import read_tree
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

    @classmethod
    def from_universe(cls, Universe):
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
        add = {'out_dir': Universe.out_dir,
               'fig_dir': Universe.fig_dir,
               'NumFiles': Universe.NumFilesInParallel,
               'ModelName': Universe.ModelName,
               'BoxSize': Universe.BoxSize,
               'UnitTime_in_yr': Universe.UnitTime_in_yr,
               'Fraction_Escape_ICM': Universe.Fraction_Escape_ICM,
               'OutputMassThreshold': Universe.OutputMassThreshold,
               'TreeRootRedshift': Universe.TreeRootRedshift,
               'cosmology': Universe.cosmology}
        trees_path = []
        for i in range(Universe.NBoxDivisions**3):
            if Universe.OutputFormat == 1:
                trees_path.append(os.path.join(Universe.out_dir, 'trees/tree.{:d}.out'.format(i)))
            else:
                trees_path.append(os.path.join(Universe.out_dir, 'trees/tree.{:d}.h5'.format(i)))
        return cls(trees_path, add_attrs=add)

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
        col_alias['Intra_cluster_mass'] += ['icm']
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

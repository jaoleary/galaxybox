"""Classes for handling Emerge data."""
from ...io.emerge_io import read_outputs

import numpy as np
import pandas as pd
import os
import glob
import h5py

from scipy.interpolate import interp1d

__author__ = ('Joseph O\'Leary', )

#TODO : This whole class needs updating.
class galaxy_catalog:
    """A class for reading in and operating on Emerge galaxy catalogs
    alone or within the context of a Universe.
    """

    def __init__(self, galaxies_path, add_attrs=None):
        """Initialize a galaxy_catalog object

        Parameters
        ----------
        galaxies_path : string or list of strings
            A list of galaxy catalog files to be included into the class.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None
        """
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])

        self.galaxies_used = list(np.atleast_1d(galaxies_path))

        self.snapshots, scalefactor, self.redshifts = [], [], []

        # get snapshot number and redshift for each file.
        for i, fp in enumerate(galaxies_path):
            fbase = fp.split('.')[0]
            self.snapshots += [fp.split('.')[1]]
            with h5py.File(fp, 'r') as f:
                scalefactor += [f['Galaxies'].attrs['Scale Factor']]
                Nfiles = f.attrs['NFiles']

        scalefactor = np.unique(scalefactor)[::-1]
        self.redshifts = 1/scalefactor - 1
        self.snapshots = np.unique(self.snapshots)[::-1]

        self.__galaxies = {}
        for i, skey in enumerate(self.snapshots):
            #skey = 'S{:d}'.format(s)
            self.__galaxies[skey] = {}
            self.__galaxies[skey]['redshift'] = self.redshifts[i]
            if Nfiles > 1:
                self.__galaxies[skey]['data'] = pd.concat([read_outputs('.'.join([fbase,skey,'{:d}'.format(j),'h5']), key='Galaxies') for j in range(Nfiles)], copy=False)
            else:
                self.__galaxies[skey]['data'] = pd.concat([read_outputs('.'.join([fbase,skey,'{:d}'.format(j),'h5']), key='Galaxies') for j in range(Nfiles)], copy=False)
            if 'Halo_ID' in self.__galaxies[skey]['data'].keys(): self.__galaxies[skey]['data'].rename(columns={"Halo_ID": "ID"}, inplace=True)
            self.__galaxies[skey]['data'].set_index('ID', drop=True, inplace=True)
        

    @classmethod
    def from_universe(cls, Universe, galaxies_path=None):
        add = {'out_dir': Universe.out_dir,
               'fig_dir': Universe.fig_dir,
               'NumOutputFiles': Universe.params.get_param('NumOutputFiles'),
               'ModelName': Universe.params.get_param('ModelName'),
               'BoxSize': Universe.params.get_param('BoxSize'),
               'OutputMassThreshold': Universe.params.get_param('OutputMassThreshold'),
               'OutputRedshifts': Universe.params.get_param('OutputRedshifts'),
               'UnitTime_in_yr': Universe.params.get_param('UnitTime_in_yr'),
               'cosmology': Universe.cosmology}

        def empty(fp):
            f = h5py.File(fp, 'r')
            if 'Galaxies' in f.keys():
                return False
            else:
                print('`{}` is empty'.format(fp))
                return True

        if galaxies_path is None:
            galaxies_path = [name for name in glob.glob(os.path.join(Universe.out_dir, 'galaxies/galaxies.*' + '.h5'))]
            galaxies_path.sort(reverse=True, key=lambda x: int(x.split('.')[-2].strip('S')))
            
            # only import non-empty files
            galaxies_path = [fp for fp in galaxies_path if not empty(fp)]

        return cls(galaxies_path, add_attrs=add)

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
        s = [*self.__galaxies.keys()][0]
        colnames = list(self.__galaxies[s]['data'].keys())
        col_alias = {}
        for k in colnames:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias['Scale'] = ['a', 'scale_factor']
        col_alias['redshift'] = ['redshift', 'z']
        col_alias['snapshot'] = ['snapshot', 'snapnum', 'snap']
        col_alias['color'] = ['color']
        col_alias['color_obs'] = ['color_obs', 'col_obs', 'color_observed']
        col_alias['Stellar_mass'] += ['mass', 'mstar']
        col_alias['Halo_mass'] += ['mvir']
        col_alias['Type'] += ['gtype']
        col_alias['Up_ID'] += ['ihost', 'host_id', 'id_host']
        col_alias['Desc_ID'] += ['idesc', 'id_desc']
        col_alias['Original_ID'] += ['ogid', 'rockstar_id', 'id_rockstar', 'id_original', 'rs_id', 'id_rs', 'irs', 'rsid']
        if 'Intra_cluster_mass' in colnames: col_alias['Intra_cluster_mass'] += ['icm']
        col_alias['Stellar_mass_obs'] += ['mstar_obs']
        col_alias['Halo_radius'] += ['rvir', 'virial_radius', 'radius']

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError('`{}` has no known alias.'.format(key))

    def list(self, **kwargs):
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
                redshift = self.__galaxies[kwargs['snapshot']]['redshift']
                galaxy_list = self.__galaxies[kwargs['snapshot']]['data']
                kwargs.pop('snapshot')
            elif 'scale' in kwargs:
                redshift = 1 / kwargs['scale'] - 1
                redshift_arg = (np.abs(self.redshifts - redshift)).argmin()
                s_key = self.snapshots[redshift_arg]
                redshift = self.__galaxies[s_key]['redshift']
                galaxy_list = self.__galaxies[s_key]['data']
                kwargs.pop('Scale')
            elif 'redshift' in kwargs:
                redshift_arg = (np.abs(self.redshifts - kwargs['redshift'])).argmin()
                s_key = self.snapshots[redshift_arg]
                redshift = self.__galaxies[s_key]['redshift']
                galaxy_list = self.__galaxies[s_key]['data']
                kwargs.pop('redshift')
            else:
                s_key = self.snapshots[0]
                redshift = self.__galaxies[s_key]['redshift']
                galaxy_list = self.__galaxies[s_key]['data']

            #print('Using snapshot {} at z={:.3f}'.format(s_key, redshift))
            # Setup a default `True` mask
            mask = galaxy_list['Up_ID'] > -10
            # Loop of each column in the tree and check if a min/max value mask should be created.
            for i, key in enumerate(galaxy_list.keys()):
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
                            sub_mask = galaxy_list['Up_ID'] < -10
                            for v in values:
                                sub_mask = sub_mask | (galaxy_list[key] == v)
                            mask = mask & sub_mask
            # Create masks for derived quantities such as `color`.
            if 'color' in kwargs.keys():
                if kwargs['color'].lower() == 'blue':
                    mask = mask & ((np.log10(galaxy_list['SFR']) - galaxy_list['Stellar_mass']) >= np.log10(0.3 / self.cosmology.age(redshift).value / self.UnitTime_in_yr))
                elif kwargs['color'].lower() == 'red':
                    mask = mask & ((np.log10(galaxy_list['SFR']) - galaxy_list['Stellar_mass']) < np.log10(0.3 / self.cosmology.age(redshift).value / self.UnitTime_in_yr))
            if 'color_obs' in kwargs.keys():
                if kwargs['color_obs'].lower() == 'blue':
                    mask = mask & ((np.log10(galaxy_list['SFR_obs']) - galaxy_list['Stellar_mass_obs']) >= np.log10(0.3 / self.cosmology.age(redshift).value / self.UnitTime_in_yr))
                elif kwargs['color_obs'].lower() == 'red':
                    mask = mask & ((np.log10(galaxy_list['SFR_obs']) - galaxy_list['Stellar_mass_obs']) < np.log10(0.3 / self.cosmology.age(redshift).value / self.UnitTime_in_yr))

            return galaxy_list.loc[mask]
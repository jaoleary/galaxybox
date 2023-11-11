"""Classes for handling Emerge lightcone catalogs."""
from astropy import constants as apconst
import astropy.units as apunits

from ...io.io import *
from ...io.emerge_io import *
from ...helper_functions import *
from ...plot.plot import *
from ...plot.emerge_plot import *
from ...mock_observables import *

import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

import h5py

from scipy.spatial import ckdtree

__author__ = ('Joseph O\'Leary', )

# TODO: This was all thrown together hastily, needs some cleanup and better documentation.

class survey:
    """A wrapper for working on lightcone groups."""
    def __init__(self, file_path, add_attrs=None):
        """A collection of observations"""
        self.file_path = file_path
        self.__group = {}
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])
        
        self.__infile = []
        if not os.path.exists(self.file_path):
            # if the file does exist make it.
            # TODO: attach other usefull properties like snapshot spacing.
            file = h5py.File(self.file_path, 'w')
            file.attrs['ModelName'] = self.ModelName
            file.close()
        else:
            # lets just see what is in the file...
            file = h5py.File(self.file_path, 'r')
            for key in file.keys():
                for dset in file[key].keys():
                    self.__infile += [key+'/'+dset]
            file.close()
        
    def __str__(self):
        pstr = 'Survey: '+self.name+'\n'
        for i, obs in enumerate(self.__observations):
            pstr += 4*' ' + str(getattr(self, obs))
        return pstr
            
    def load_group(self, group, link_trees=False):
        """load an entire group from the survey file."""
        if group not in self.__group.keys():
            self.__group[group] = {}
        for g in self.__infile:
            if g.startswith(group):
                name = g.split('/')[-1]
                self.__group[group][name] = observation.from_file(self.file_path, key=g)
                if link_trees:
                    self.__group[group][name].link_trees(self.__trees, self.__tree_alias)
                    self.__group[group][name].set_cone_slice()
        
    def load_observation(self, group, name, link_trees=False):
        """Load a single observation from the survey file."""
        if group not in self.__group.keys():
            self.__group[group] = {}
        key = group+'/'+name
        self.__group[group][name] = observation.from_file(self.file_path, key=key)
        if link_trees:
            self.__group[group][name].link_trees(self.__trees, self.__tree_alias)
            self.__group[group][name].set_cone_slice()
            
    def avail(self):
        """show which groups/observations are available currently, and in the `survey` file"""
        print('Loaded:')
        for i, g in enumerate(self.__group.keys()):
            print(4*' '+g)
            for j, o in enumerate(self.__group[g].keys()):
                obs = self.__group[g][o]
                print(8*' '+obs.name)
        
        print('In file:')
        file = h5py.File(self.file_path, 'r')
        for key in file.keys():
            print(4*' '+key)
            for dset in file[key].keys():
                print(8*' '+dset)
        file.close()
            
    def new_observation(self, group, name, **kwargs):
        """create a new observation using the make_lightcone_catalog method from galaxy_trees"""
        if hasattr(self, '_survey__trees'):
            if group not in self.__group.keys():
                self.__group[group] = {}
            
            catalog, LC, params = self.__make_lightcone_catalog(**kwargs)
            self.__group[group][name] = observation(catalog, name=name, geometry=LC)
            self.__group[group][name].link_trees(self.__trees, self.__tree_alias)
            self.__group[group][name].min_z = params[0]
            self.__group[group][name].max_z = params[1]
            self.__group[group][name].set_cone_slice()
            
    def link_trees(self, galtree):
        self.__trees = galtree.trees
        self.__tree_alias = galtree.alias
        self.__make_lightcone_catalog = galtree.make_lightcone_catalog
        
    def calc(self, group, func, **kwargs):
        """Execute methods by group"""
        out = []
        for o in self.__group[group].keys():
            obs = self.__group[group][o]
            meth = getattr(obs, func)
            out += [meth(**kwargs)]
        return out

    def obs(self, group, name):
        """Direct access to each observation"""
        return self.__group[group][name]
    
    def save(self, overwrite=False):
        """Save merger class data to an hdf5 file."""
        # TODO: specify file_path
        # TODO: specify which observation should be saved
        file_path = self.file_path
        for i, g in enumerate(self.__group.keys()):
            for j, o in enumerate(self.__group[g].keys()):
                datpath = g+'/'+o
                if overwrite and (datpath in self.__infile):
                    self.delete(group=g, name=o)
                if datpath not in self.__infile:
                    obs = self.__group[g][o]
                    obs.save(file_path, key = datpath)
                    self.__infile += [datpath]
                    
    def delete(self, group, name):
        """Remove an observation from file"""
        datpath = group+'/'+'name'
        if datpath in self.__infile:
            self.__infile.remove(datpath)
            file = h5py.File(self.file_path, 'a')
            del file[datpath]
            file.close()
            
  
class observation:
    """ This is a single lightcone catalog and the associated geometry"""
    def __init__(self, catalog, name, geometry):
        self.__data = catalog
        self.__slice = [self.__data.index.values]
        self.name = name
        self.geometry = geometry
            
    def __str__(self):
        # TODO: add more descriptive output
        pstr = self.name + '\n'
        pstr += 4*' ' + 'N gal: = {}\n'.format(len(self.__data))
        return pstr

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
        # lets see if theres already an alias in the tree.
        try:
            return self.__tree_alias(key)
        except:
            # first lets make all columns case insensitive 
            colnames = list(self.__data.keys())
            col_alias = {}
            for k in colnames:
                col_alias[k] = [k.lower()]

            # add other aliases.
            col_alias['index'] = ['index']
            col_alias['slice'] = ['slice']
            col_alias['Tree_ID'] += ['igal', 'treeid']
            col_alias['Redshift'] += ['z']
            col_alias['Redshift_obs'] += ['z_obs', 'zobs']
            col_alias['RA'] += ['right_ascension', 'alpha']
            col_alias['Dec'] += ['declination', 'delta']

            for k in col_alias.keys():
                if key.lower() in col_alias[k]:
                    return k
            raise KeyError('`{}` has no known alias.'.format(key))
            
    def list(self, merge=True, **kwargs):
        """Return list of galaxy with specified properties.

        This function selects galaxies based on a flexible set of arguments. Any column of the galaxy.trees attribute can
        have a `min` are `max` added as a prefix or suffix to the column name or alias of the column name and passed as
        and argument to this function. This can also be used to selected galaxies based on derived properties such as color.

        Parameters
        ----------
        merge : bool, optional
            If True the lightcone dataframe will be merged to the tree dataframe so that tree columns can also be selected, by default True

        Returns
        -------
        pandas.DataFrame
            A masked subset of the galaxy.trees attribute

        # TODO: expand this docstring and add examples.
        """
        # BUG: This listing method will break if not linked to trees. 
        # First clean up kwargs, replace aliases
        keys = list(kwargs.keys())
        for i, kw in enumerate(keys):
            keyl = kw.lower()
            key = keyl.lower().replace('min', '').replace('max', '').lstrip('_').rstrip('_')
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        # merge the lightcone data with the tree data.
        galaxy_list = self.__data.merge(self.__trees, left_on='Tree_ID', right_on='ID', copy=False, suffixes=(True,False))

        # Slices are just index lists. join those lists together and set them as an index kwarg
        if 'slice' in kwargs:
            kwargs['slice'] = np.atleast_1d(kwargs['slice'])
            kwargs['index']=[]
            for i, s in enumerate(kwargs['slice']):
                kwargs['index'] += list(self.__slice[s])
            kwargs.pop('slice')

        # prioritize masking over index. this is needed for the pair selection.
        if 'index' in kwargs:
            galaxy_list = galaxy_list.loc[kwargs['index']]
            kwargs.pop('index')
        # Setup a default `True` mask
        mask = galaxy_list['Scale'] > 0
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

        if merge:
            return galaxy_list.loc[mask]
        else:
            return self.__data
    
    def save(self, file_path, key=None):
        """Save this observation to an hdf5 file.

        Parameters
        ----------
        file_path : str
            Path to output file
        key : str, optional
            The name of the dataset in the file. If None the observation name will be used, by default None
        """        
        if key is None:
            key = self.name
        file = h5py.File(file_path, 'a')
        
        data = self.__data.to_records(index=False)
        dset = file.create_dataset(key, data=data, compression='gzip', compression_opts=9)
        # save the info needed to recreate the lightcone object.
        dset.attrs['RA'] = self.geometry.da.value
        dset.attrs['Dec'] = self.geometry.dd.value
        dset.attrs['u1'] = self.geometry.u1
        dset.attrs['u2'] = self.geometry.u2
        dset.attrs['u3'] = self.geometry.u3
        dset.attrs['Lbox'] = self.geometry.Lbox
        dset.attrs['full_width'] = self.geometry.full_width
        dset.attrs['seed'] = self.geometry.seed
        dset.attrs['min_z'] = self.min_z
        dset.attrs['max_z'] = self.max_z
        
        file.close()
        
    @classmethod
    def from_file(cls, file_path, key):
        """Read observation from a file."""
        file = h5py.File(file_path, 'r')
        dset = file[key]
        catalog = pd.DataFrame(dset[()])
        da = Angle(dset.attrs['RA'], apunits.radian)
        dd = Angle(dset.attrs['Dec'], apunits.radian)
        u1 = dset.attrs['u1']
        u2 = dset.attrs['u2']
        u3 = dset.attrs['u3']
        Lbox = dset.attrs['Lbox']
        full_width = dset.attrs['full_width']
        seed = dset.attrs['seed']
        min_z = dset.attrs['min_z']
        max_z = dset.attrs['max_z']

        file.close()
        LC = lightcone(da, dd, u1,u2, u3, Lbox, full_width=full_width)
        LC.set_seed(seed=seed)
        
        obs = cls(catalog = catalog, name = key.split('/')[-1], geometry = LC)
        obs.min_z = min_z
        obs.max_z = max_z
        
        return obs
    
    def link_trees(self, galtree, alias):
        """Give the observation access to the tree...but lets just keep this to ourselves.
        This way we can grab the other galaxy properties without increasing filesize."""
        self.__trees = galtree.loc[self.__data['Tree_ID'].values].drop_duplicates()    
        self.__tree_alias = alias

    def set_cone_slice(self, bins=1):
        """Bin the lightcone catalog by redshift

        Parameters
        ----------
        bins : int or sequence of scalars or str, optional
            This function uses `numpy.histogram_bin_edges` refer to this documentation for use of this parameter, by default 1
        """
        bins = np.histogram_bin_edges([self.min_z, self.max_z], bins=bins)
        self.redshift_bins = bins
        self.__slice = [None]*(len(bins)-1)

        # Gather the data indicies that fit in each bin.
        for i, m in enumerate(bins[:-1]):
            self.__slice[i] = self.list(min_redshift=bins[i], max_redshift=bins[i+1]).index.values


    def set_distance_matrix(self, max_distance, coordinate='cartesian', comoving=True, slice=None):
        #? would using the halotools implementation offer any advantage?
        if slice is None:
            slice = [i for i in range(len(self.__slice))]
        self.distmat = []
        for i, s in enumerate(slice):
            galaxy_list = self.list(slice=slice[s])
            if coordinate.lower() == 'cartesian':
                if comoving:
                    sparse = ckdtree.cKDTree(galaxy_list[['X_cone', 'Y_cone']])
                else:
                    x = galaxy_list['X_cone'].values / (galaxy_list['Redshift'].values + 1)
                    y = galaxy_list['Y_cone'].values / (galaxy_list['Redshift'].values + 1)
                    pos = np.vstack([x, y]).T
                    sparse = ckdtree.cKDTree(pos)
            elif coordinate.lower() == 'angular':
                sparse = ckdtree.cKDTree(galaxy_list[['RA', 'Dec']])
            else:
                raise ValueError('`{}` not valid argument for coordinate'.format(coordinate))

            self.distmat += [sparse.sparse_distance_matrix(sparse, max_distance=max_distance)]
    
    def get_pairs(self, r_min, r_max, MR_min=1.0, MR_max=4.0, min_mstar=0.0, dv_max=500.0, slice=None):
        """Find all projected galaxy pairs meeting the specified criteria.

        Parameters
        ----------
        r_min : float
            minimum projected separation between pairs [kpc]
        r_max : float
            maximum projected separation between pairs [kpc]
        MR_min : float, optional
            Minimum stellar mass ratio, by default 1.0
        MR_max : float, optional
            Maximum stellar mass ratio, by default 4.0
        min_mstar : float, optional
            Log stellar mass cut for the main galaxy in the pair, by default 0.0
        dv_max : float, optional
            maximum LoS velocity difference between pairs [km/s], by default 500.0

        Returns
        -------
        list
            A list galaxy pairs by index value in the cone catalogs.
        """
        if slice is None:
            slice = [i for i in range(len(self.__slice))]
        
        pairs = []
        for i, s in enumerate(slice):

            # grab all pairs with a non-zero separation.
            G1_idx, G2_idx = self.distmat[i].nonzero()
            #redshift = self.list()['Redshift'].values
            #z_mean = np.mean([redshift.min(),redshift.max()])
            galaxies = self.list(slice=s)
            # get the properties for each pair
            G1_prop = galaxies.iloc[G1_idx]
            G2_prop = galaxies.iloc[G2_idx]
            # set a false mask
            dmask = np.full(len(G1_idx), False)
            # check each pair
            for j in range(len(dmask)):
                distance = self.distmat[i][(G1_idx[j], G2_idx[j])]
                if (distance >= r_min) & (distance < r_max):
                    main_gal = G1_prop.iloc[j]
                    minor_gal = G2_prop.iloc[j]

                    if main_gal['Stellar_mass'] >= min_mstar:
                        MR = 10**(main_gal['Stellar_mass'] - minor_gal['Stellar_mass'])
                        if ((MR >= MR_min) & (MR < MR_max)):
                            #z_mean = (main_gal['Redshift'] + minor_gal['Redshift'])/2
                            z_mean = main_gal['Redshift']
                            if (np.abs(main_gal['Redshift_obs'] - minor_gal['Redshift_obs']) <= dv_max*(1+z_mean)/apconst.c.to('km/s').value):
                                dmask[j] = True

            G1_index = G1_prop[dmask].index.values
            G2_index = G2_prop[dmask].index.values
            pairs += [[(G1_index[k], G2_index[k]) for k in range(len(G1_index))]]
        
        return pairs
    
    def plot_cone(self, frame='cone', comoving=True, **kwargs):
        """Plot a visual representation of the galaxies contained in the light cone.

        Parameters
        ----------
        frame : str, optional
            If `box` the plot will be generated in the reference from of the cosmological box. If `cone` the plot will be in the refrence frame of the lightcone geometry, by default 'cone'
        comoving : bool, optional
            If False physical coordinates will be used, by default True
        """        
        LC_cat = self.list(**kwargs)
        pos = LC_cat[['X_cone','Y_cone','Z_cone']].values
        unitstr = ' [cMpc]'
        if not comoving:
            unitstr = ' [Mpc]'
            pos[:,0] = pos[:,0]*LC_cat['Scale']
            pos[:,1] = pos[:,1]*LC_cat['Scale']
            pos[:,2] = pos[:,2]*LC_cat['Scale']
        
        if frame == 'box':
            pos = self.geometry.rotation.apply(pos, inverse=True)

        fig, ax = plt.subplots(2,2,figsize=(16,16))
        ax[0,1].set_xlabel('RA [rad]', size=16)
        ax[0,1].set_ylabel('Dec [rad]', size=16)

        ax[1,0].set_xlabel('X'+unitstr, size=16)
        ax[1,0].set_ylabel('Y'+unitstr, size=16)

        ax[0,0].set_xlabel('X'+unitstr, size=16)
        ax[0,0].set_ylabel('Z'+unitstr, size=16)

        ax[1,1].set_xlabel('Z'+unitstr, size=16)
        ax[1,1].set_ylabel('Y'+unitstr, size=16)
        ms=4
        ax[0,1].scatter(LC_cat['RA'], LC_cat['Dec'], c=LC_cat['Redshift'], cmap=plt.cm.jet, s=ms)
        ax[1,0].scatter(pos[:,0], pos[:,1], c=LC_cat['Redshift'], cmap=plt.cm.jet, s=ms)
        ax[0,0].scatter(pos[:,0], pos[:,2], c=LC_cat['Redshift'], cmap=plt.cm.jet, s=ms)
        ax[1,1].scatter(pos[:,2], pos[:,1], c=LC_cat['Redshift'], cmap=plt.cm.jet, s=ms)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=LC_cat['Redshift'].min(), vmax=LC_cat['Redshift'].max()))
        cbar = plt.colorbar(sm)
        cbar.set_label('Redshift', fontsize=16)

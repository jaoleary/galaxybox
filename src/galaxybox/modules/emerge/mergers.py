"""Classes for handling Emerge data."""

import os

import h5py
import numpy as np
import pandas as pd


class GalaxyMergers:
    """Merger list generated using Emerge galaxy trees."""

    def __init__(self, merger_list_path, add_attrs=None, save=False):
        """Initializse a galaxy_mergers object.

        Parameters
        ----------
        merger_list_path : string or pandas.DataFrame
            A list of tree files to be included into the class.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None
        save : bool, optional
            save the merger file on init.

        """
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])
        if isinstance(merger_list_path, str):
            print("Loading merger list:\n" + merger_list_path)
            self._list = pd.read_hdf(merger_list_path, key="Data")
        else:
            self._list = merger_list_path

        if save:
            self.save()

    @classmethod
    def from_galaxy_trees(cls, galaxy_trees, save, **kwargs):
        """Initiate a `galaxy_mergers` from a  `galaxy_tree` class.

        A merger list is created using the `find_mergers` method of the `galaxy_tree` class.

        Parameters
        ----------
        cls : class
            The class object.
        galaxy_trees : `galaxybox.galaxy_trees`
            The galaxy_trees class.
        save : bool
            Whether to save the merger list.
        **kwargs : dict
            Additional keyword arguments to be passed to the `find_mergers` method.

        Returns
        -------
        galaxybox.galaxy_mergers
            galaxy mergers set from galaxy_trees

        """
        add = {
            "out_dir": galaxy_trees.out_dir,
            "fig_dir": galaxy_trees.fig_dir,
            "ModelName": galaxy_trees.ModelName,
            "BoxSize": galaxy_trees.BoxSize,
            "UnitTime_in_yr": galaxy_trees.UnitTime_in_yr,
            "cosmology": galaxy_trees.cosmology,
        }
        merger_list = galaxy_trees.find_mergers(**kwargs)
        return cls(merger_list, add_attrs=add, save=save)

    @classmethod
    def from_file(cls, merger_list_path, add_attrs=None):
        """Initiate a `galaxy_mergers` by pointing to the file containing the lists.

        Parameters
        ----------
        merger_list_path : str
            path to the file containing the galaxy mergers file.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None

        Returns
        -------
        galaxy_mergers
            A galaxy mergers object

        """
        return cls(merger_list_path, add_attrs=add_attrs)

    def save(self):
        """Save merger class data to an hdf5 file."""
        file_out = os.path.join(self.out_dir, "mergers.h5")
        try:
            os.remove(file_out)
        except OSError:
            pass

        f = h5py.File(file_out, "w")
        data = self._list.to_records(index=False)
        f.create_dataset("Data", data=data, compression="gzip", compression_opts=9)
        f.close()

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
        colnames = list(self._list.keys())
        col_alias = {}
        for k in colnames:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias["Scale"] += ["a", "scale_factor"]
        col_alias["tdf"] += ["tmerge", "time"]

        col_alias["Desc_mstar"] += ["descmass", "desc_mass", "desc_stellar_mass"]
        col_alias["Desc_mstar_obs"] += [
            "descmass_obs",
            "descmass_observed",
            "desc_mass_obs",
            "desc_mass_observed",
            "desc_stellar_mass_obs",
            "desc_stellar_mass_observed",
        ]
        col_alias["Desc_sfr"] += ["descsfr", "desc_star_formation_rate"]
        col_alias["Desc_sfr_obs"] += [
            "descsfr_obs",
            "descsfr_observed",
            "desc_star_formation_rate_obs",
            "desc_star_formation_rate_observed",
        ]
        col_alias["Desc_mvir"] += ["descmvir", "desc_halo_mass"]
        col_alias["Desc_ID"] += ["idesc", "id_desc"]
        col_alias["desc_color"] = ["desc_color"]
        col_alias["desc_color_obs"] = ["desc_color_obs", "desc_color_observed"]

        col_alias["Main_mstar"] += ["mainmass", "main_mass", "main_stellar_mass"]
        col_alias["Main_mstar_obs"] += [
            "mainmass_obs",
            "mainmass_observed",
            "main_mass_obs",
            "main_mass_observed",
            "main_stellar_mass_obs",
            "main_stellar_mass_observed",
        ]
        col_alias["Main_sfr"] += ["mainsfr", "main_star_formation_rate"]
        col_alias["Main_sfr_obs"] += [
            "mainsfr_obs",
            "mainsfr_observed",
            "main_star_formation_rate_obs",
            "main_star_formation_rate_observed",
        ]
        col_alias["Main_mvir"] += ["mainmvir", "main_halo_mass"]
        col_alias["Main_ID"] += ["imain", "id_main"]
        col_alias["main_color"] = ["main_color"]
        col_alias["main_color_obs"] = ["main_color_obs", "main_color_observed"]

        col_alias["Minor_mstar"] += ["minormass", "minor_mass", "minor_stellar_mass"]
        col_alias["Minor_mstar_obs"] += [
            "minormass_obs",
            "minormass_observed",
            "minor_mass_obs",
            "minor_mass_observed",
            "minor_stellar_mass_obs",
            "minor_stellar_mass_observed",
        ]
        col_alias["Minor_sfr"] += ["minorsfr", "minor_star_formation_rate"]
        col_alias["Minor_sfr_obs"] += [
            "minorsfr_obs",
            "minorsfr_observed",
            "minor_star_formation_rate_obs",
            "minor_star_formation_rate_observed",
        ]
        col_alias["Minor_mvir"] += ["minormvir", "minor_halo_mass"]
        col_alias["Minor_ID"] += ["iminor", "id_minor"]
        col_alias["minor_color"] = ["minor_color"]
        col_alias["minor_color_obs"] = ["minor_color_obs", "minor_color_observed"]

        col_alias["MR"] += ["massratio", "mass_ratio", "mu", "ratio"]
        col_alias["MR_obs"] += [
            "massratio_obs",
            "massratio_observed",
            "mass_ratio_obs",
            "mass_ratio_observed",
            "mu_obs",
            "mu_observed",
        ]

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))

    def list(self, mask_only=False, **kwargs):
        """Return list of galaxy mergers with specified properties.

        This function selects galay mergers based on a flexible set of arguments. Any column of the
        galaxy_mergers.__list attribute can have a `min` are `max` added as a prefix or suffix to
        the column name or alias of the column name and passed as and argument to this function.
        This can also be used to selected galaxy mergers based on derived properties such as color.

        Parameters
        ----------
        mask_only : bool, optional
            Return the dataframe, or a mask for the dataframe, by default False
        **kwargs : dict
            Additional keyword arguments used to filter the galaxy mergers.

        Returns
        -------
        pandas.DataFrame
            A masked subset of the galaxy_mergers.__list attribute

        """
        # First clean up kwargs, replace aliases
        keys = list(kwargs.keys())
        for i, kw in enumerate(keys):
            keyl = kw.lower()
            key = (
                keyl.lower()
                .replace("min_", "")
                .replace("_min", "")
                .replace("max_", "")
                .replace("_max", "")
            )
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        # Setup a default `True` mask
        mask = self._list["Scale"] > 0
        # Loop of each column in the tree and check if a min/max value mask should be created.
        for i, key in enumerate(self._list.keys()):
            for j, kw in enumerate(kwargs.keys()):
                if ("obs" in kw.lower()) & ("obs" not in key.lower()):
                    pass
                elif key.lower() in kw.lower():
                    if ("min_" in kw.lower()) or ("_min" in kw.lower()):
                        mask = mask & (self._list[key] >= kwargs[kw])
                    elif ("max_" in kw.lower()) or ("_max" in kw.lower()):
                        mask = mask & (self._list[key] < kwargs[kw])
                    else:
                        values = np.atleast_1d(kwargs[kw])
                        # Setup a default `False` mask
                        sub_mask = self._list["Scale"] > 1
                        for v in values:
                            sub_mask = sub_mask | (self._list[key] == v)
                        mask = mask & sub_mask
        # Create masks for derived quantities such as `color`.
        for i, kw in enumerate(kwargs.keys()):
            if "color" in kw:
                if "obs" in kw:
                    obs = "_obs"
                else:
                    obs = ""
                gal = kw.split("_")[0]
                sfr_key = self.alias(gal + "_sfr" + obs)
                mass_key = self.alias(gal + "_mstar" + obs)
                if kwargs[kw].lower() == "blue":
                    mask = mask & (
                        (np.log10(self._list[sfr_key]) - self._list[mass_key])
                        >= np.log10(0.3 / self._list["tdf"] / self.UnitTime_in_yr)
                    )
                elif kwargs[kw].lower() == "red":
                    mask = mask & (
                        (np.log10(self._list[sfr_key]) - self._list[mass_key])
                        < np.log10(0.3 / self._list["tdf"] / self.UnitTime_in_yr)
                    )

        if mask_only:
            return mask
        else:
            return self._list.loc[mask]

    def hist(self, axis, bins=None, inverse=False, log=False, **kwargs):
        """Histogram of mergers along some specified axis.

        Parameters
        ----------
        axis : string
            the axis along which to make a histogram.
            Can use any column of the merger list.

        bins : int or 1-D array, optional
            An ascending array of bins to be used for the histogram

        inverse : bool, optional
            If True, compute the histogram of the inverse of the values along the specified axis.
            Default is False.

        log : bool, optional
            If True, compute the histogram of the logarithm of the values along the specified axis.
            Default is False.

        **kwargs : dict
            Additional keyword arguments used to filter the galaxy mergers.

        Returns
        -------
        N_mergers : 1-D array
            A histogram containing the number of galaxy-galaxy mergers located in the specified
            cosmic time bins

        bin_edges : array of dtype float
            Return the bin edges ``(length(hist)+1)``.

        """
        # TODO: compute MR on the fly....maybe
        # ? Is this method really even needed?

        # allow aliasing for axis argument
        axis = self.alias(axis)

        # Combine the mass and mass ratio masks
        mask = self.list(mask_only=True, **kwargs)

        try:
            if inverse and log:
                return np.histogram(np.log10(1 / self._list.loc[mask][axis].values), bins)
            elif inverse:
                return np.histogram(1 / self._list.loc[mask][axis].values, bins)
            elif log:
                return np.histogram(np.log10(self._list.loc[mask][axis].values), bins)
            else:
                return np.histogram(self._list.loc[mask][axis].values, bins)
        except:  # noqa E722
            raise Exception("Unrecognized axis type")


class HaloMergers:
    """Merger list generated using Emerge halo trees."""

    def __init__(self, merger_list_path, add_attrs=None, save=False):
        """Initialize a halo_mergers object.

        Parameters
        ----------
        merger_list_path : string or pandas.DataFrame
            A list of tree files to be included into the class.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None
        save : bool, optional
            save the merger file on init.

        """
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])
        if isinstance(merger_list_path, str):
            print("Loading halo merger list:\n" + merger_list_path)
            self._list = pd.read_hdf(merger_list_path, key="Data")
        else:
            self._list = merger_list_path

        if save:
            self.save()

    @classmethod
    def from_halo_trees(cls, halo_trees, save):
        """Initiate a `galaxy_mergers` from a  `galaxy_tree` class.

        A merger list is created using the `find_mergers` method of the `halo_tree` class.

        Parameters
        ----------
        halo_trees : `galaxybox.halo_trees`
            The galaxy_trees class
        save : bool
            Whether to save the merger class data to an hdf5 file.

        Returns
        -------
        galaxybox.halo_mergers
            halo mergers set from halo_trees

        """
        add = {
            "out_dir": halo_trees.out_dir,
            "fig_dir": halo_trees.fig_dir,
            "ModelName": halo_trees.ModelName,
            "BoxSize": halo_trees.BoxSize,
            "UnitTime_in_yr": halo_trees.UnitTime_in_yr,
            "cosmology": halo_trees.cosmology,
        }
        merger_list = halo_trees.find_mergers()
        return cls(merger_list, add_attrs=add, save=save)

    @classmethod
    def from_file(cls, merger_list_path, add_attrs=None):
        """Initiate a `halo_mergers` by pointing to the file containing the lists.

        Parameters
        ----------
        merger_list_path : str
            path to the file containing the halo mergers file.
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None

        Returns
        -------
        halo_mergers
            A halo mergers object

        """
        return cls(merger_list_path, add_attrs=add_attrs)

    def save(self):
        """Save merger class data to an hdf5 file."""
        file_out = os.path.join(self.out_dir, "halo_mergers.h5")
        try:
            os.remove(file_out)
        except OSError:
            pass

        f = h5py.File(file_out, "w")
        data = self._list.to_records(index=False)
        f.create_dataset("Data", data=data, compression="gzip", compression_opts=9)
        f.close()

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
        colnames = list(self._list.keys())
        col_alias = {}
        for k in colnames:
            col_alias[k] = [k.lower()]

        # add other aliases.
        col_alias["Scale"] += ["a", "scale_factor"]

        col_alias["Desc_mvir"] += ["descmvir", "desc_halo_mass"]
        col_alias["Desc_ID"] += ["idesc", "id_desc"]

        col_alias["Main_mvir"] += ["mainmvir", "main_halo_mass"]
        col_alias["Main_ID"] += ["imain", "id_main"]

        col_alias["Minor_mvir"] += ["minormvir", "minor_halo_mass"]
        col_alias["Minor_ID"] += ["iminor", "id_minor"]

        col_alias["MR"] += ["massratio", "mass_ratio", "mu", "ratio"]

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))

    def list(self, mask_only=False, **kwargs):
        """Return list of halo mergers with specified properties.

        This function selects halo mergers based on a flexible set of arguments. Any column of the
        halo_mergers._list attribute can have a `min` are `max` added as a prefix or suffix to the
        column name or alias of the column name and passed as snd argument to this function. This
        can also be used to selected halo mergers based on derived properties such as color.

        Parameters
        ----------
        mask_only : bool, optional
            Return the dataframe, or a mask for the dataframe, by default False
        **kwargs : dict
            Additional keyword arguments to filter the halo mergers.

        Returns
        -------
        pandas.DataFrame
            A masked subset of the halo_mergers.__list attribute

        """
        # First clean up kwargs, replace aliases
        keys = list(kwargs.keys())
        for i, kw in enumerate(keys):
            keyl = kw.lower()
            key = (
                keyl.lower()
                .replace("min_", "")
                .replace("_min", "")
                .replace("max_", "")
                .replace("_max", "")
            )
            new_key = kw.replace(key, self.alias(key))
            kwargs[new_key] = kwargs.pop(kw)

        # Setup a default `True` mask
        mask = self._list["Scale"] > 0
        # Loop of each column in the tree and check if a min/max value mask should be created.
        for i, key in enumerate(self._list.keys()):
            for j, kw in enumerate(kwargs.keys()):
                if key.lower() in kw.lower():
                    if ("min_" in kw.lower()) or ("_min" in kw.lower()):
                        mask = mask & (self._list[key] >= kwargs[kw])
                    elif ("max_" in kw.lower()) or ("_max" in kw.lower()):
                        mask = mask & (self._list[key] < kwargs[kw])
                    else:
                        values = np.atleast_1d(kwargs[kw])
                        # Setup a default `False` mask
                        sub_mask = self._list["Scale"] > 1
                        for v in values:
                            sub_mask = sub_mask | (self._list[key] == v)
                        mask = mask & sub_mask

        if mask_only:
            return mask
        else:
            return self._list.loc[mask]

    def hist(self, axis, bins=10, inverse=False, log=False, **kwargs):
        """Histogram of mergers along some specified axis.

        Parameters
        ----------
        axis : string
            the axis along which to make a histogram.
            Can use any column of the merger list.

        bins : 1-D array, optional
            An ascending array of bins to be used for the histogram

        inverse : bool, optional
            If True, compute the histogram of the inverse of the values along the specified axis.
            Default is False.

        log : bool, optional
            If True, compute the histogram of the logarithm of the values along the specified axis.
            Default is False.

        **kwargs : dict
            Additional keyword arguments to filter the halo mergers.

        Returns
        -------
        N_mergers : 1-D array
            A histogram containing the number of galaxy-galaxy mergers located in the specified
            cosmic time bins

        bin_edges : array of dtype float
            Return the bin edges ``(length(hist)+1)``.

        """
        # allow aliasing for axis argument
        axis = self.alias(axis)

        # Combine the mass and mass ratio masks
        mask = self.list(mask_only=True, **kwargs)

        try:
            if inverse and log:
                return np.histogram(np.log10(1 / self._list.loc[mask][axis].values), bins)
            elif inverse:
                return np.histogram(1 / self._list.loc[mask][axis].values, bins)
            elif log:
                return np.histogram(np.log10(self._list.loc[mask][axis].values), bins)
            else:
                return np.histogram(self._list.loc[mask][axis].values, bins)
        except ValueError:
            print("Unrecognized axis type")

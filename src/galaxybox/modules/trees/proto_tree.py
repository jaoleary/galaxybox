"""module contains the definition of the ProtoTree classes."""

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import pandas as pd
import yaml
from astropy import constants as apconst
from halotools.mock_observables import radial_distance_and_velocity
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from galaxybox.data.utils import key_alias, kwarg_parser, kwargs_to_filters, minmax_kwarg_swap_alias
from galaxybox.modules.mocks.lightcone import Lightcone


class ProtoTree(ABC):
    """Abstract base class for a tree."""

    @abstractmethod
    def tree(self, *args, **kwargs) -> pd.DataFrame:
        """Abstract method to define a tree."""
        raise NotImplementedError

    @abstractmethod
    def branch(self, *args, **kwargs) -> pd.DataFrame:
        """Abstract method to define a branch."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query, columns) -> List[pd.DataFrame]:
        """Abstract method to query the tree with a given query and columns."""
        raise NotImplementedError

    @abstractmethod
    def list(self, columns=None, **kwargs) -> pd.DataFrame:
        """Abstract method to list galaxies with specified properties."""
        raise NotImplementedError

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """Absract property for a list containing the columns names of the tree."""
        raise NotImplementedError

    @property
    @abstractmethod
    def box_size(self) -> float:
        """Abstract property for the size of the simulation box."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cosmology(self) -> None:
        """Abstract property for cosmology used for the simulation."""
        raise NotImplementedError


class ProtoGalaxyTree(ProtoTree):
    """Class representing a prototype galaxy tree.

    Parameters
    ----------
    filepath : str
        The path to the file containing the galaxy tree data.
    alias_path : Optional[str], default=None
        The path to the file containing the alias data. If None, no alias data will be used.
    unittime_in_yr : float, default=1.0e9
        The unit of time, in years, used in the galaxy tree data.

    """

    def __init__(
        self, filepath: str, alias_path: str | None = None, unittime_in_yr: float = 1.0e9
    ) -> None:
        """Initialize the ProtoGalaxyTree."""
        self.filepath = filepath
        self.alias_path = alias_path
        self.unittime_in_yr = unittime_in_yr

        if self.alias_path is not None:
            with open(self.alias_path, "r") as f:
                self.col_alias = yaml.safe_load(f)

    @property
    def box_size(self) -> float:
        """Side length of the simulation volume.

        Assumes cubic volume where all sides are equal length.
        """
        return self._box_size

    @box_size.setter
    def box_size(self, value):
        try:
            self._box_size = float(value)
        except ValueError:
            raise TypeError("`box_size` must be a float or convertible to float")

    @property
    def columns(self) -> list[str]:
        """Data columns available in the galaxy tree.

        Returns
        -------
        list[str]
            list of column names

        """
        return self._columns

    @columns.setter
    def columns(self, value):
        if not isinstance(value, list):
            raise TypeError("`columns` must be a list of strings")
        self._columns = value

    def _df_query(
        self, query: List[tuple[str, str, str]], columns: List[str]
    ) -> List[pd.DataFrame]:
        """Execute a query on a pandas dataframe.

        Parameters
        ----------
        query : list of tuple
            The query conditions.
        columns : list of str
            The columns to return.

        Returns
        -------
        list of pandas.DataFrame
            The query results.

        """
        query = " & ".join([" ".join(map(str, tup)) for tup in query])
        return [self._loader(fp).query(query)[columns] for fp in self.filepath]

    def _parquet_query(
        self, query: List[tuple[str, str, str]], columns: List[str]
    ) -> List[pd.DataFrame]:
        """Query the Parquet data.

        Parameters
        ----------
        query : list of tuple
            The query conditions.
        columns : list of str
            The columns to return.

        Returns
        -------
        list of pandas.DataFrame
            The query results.

        """
        return [self._loader(fp, filters=query, columns=columns) for fp in self.filepath]

    def kwarg_swap_alias(self, kwargs):
        """Replace the keys in the `kwargs` dictionary with their aliases based on column aliases.

        This method delegates to `minmax_kwarg_swap_alias`, passing the current instance's column
        alias mapping (`self.col_alias`) along with the provided `kwargs`. It is designed to handle
        keys with 'min' or 'max' prefixes or suffixes by preserving these modifiers while replacing
        the base key with its alias from `self.col_alias`.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments where each key is a column name that may be
            potentially prefixed or suffixed with 'min' or 'max', indicating a range query. Each
            value is the corresponding value for the query.

        Returns
        -------
        dict
            A new dictionary with the keys replaced by their aliases as defined in `self.col_alias`,
            preserving any 'min' or 'max' modifiers.

        """
        return minmax_kwarg_swap_alias(kwargs, self.col_alias)

    def list(self, columns: List[str] | None = None, **kwargs) -> pd.DataFrame:
        """Return list of galaxies with specified properties.

        This function selects galaxies based on a flexible set of arguments. Any column of the
        galaxy.trees attribute can have a `min` or `max` added as a prefix or suffix to the column
        name or alias of the column name and passed as an argument to this function. This can also
        be used to select galaxies based on derived properties such as color.

        Parameters
        ----------
        columns : list, optional
            list of columns to include in the returned DataFrame, by default None
        kwargs : dict
            Keyword arguments to filter the galaxies based on column values

        Returns
        -------
        pandas.DataFrame
            A masked subset of the galaxy.trees attribute

        Examples
        --------
        # Select galaxies with redshift between 0.5 and 1.0
        galaxies = proto_tree.list(redshift_min=0.5, redshift_max=1.0)

        # Select galaxies with stellar mass greater than 1e10
        galaxies = proto_tree.list(Stellar_mass_min=1e10)

        # Select galaxies with color_obs equal to 'blue' or 'red'
        galaxies = proto_tree.list(color_obs=['blue', 'red'])

        """
        if columns is None:
            columns = self.columns
        # First clean up kwargs, replace aliases
        kwargs = self.kwarg_swap_alias(kwargs)
        filters = kwargs_to_filters(kwargs)

        return self.query(query=filters, columns=columns)

    def count(
        self,
        time_column: str,
        target_time: float | Sequence[float] = None,
        dtype=int,
        **kwargs,
    ) -> np.ndarray:
        """Count the number of galaxies at specified scalefactor(s).

        Parameters
        ----------
        time_column : str
            The column to use for the time values.
        target_time : float, list of floats, optional
            time at which a galaxy count should be performed, by default None
        dtype : data type, optional
            interpolated values will be cast into this type, by default int
        **kwargs : dict
            Additional keyword arguments for use in the `list` method.

        Returns
        -------
        dtype, numpy array of dtype
            The number of galaxies at each input scale factor

        """
        n_gal = self.list(columns=[time_column], **kwargs).value_counts().sort_index().reset_index()

        target_time = np.atleast_1d(target_time)
        func = interp1d(n_gal[time_column], n_gal["count"], fill_value="extrapolate")
        counts = func(target_time).astype(dtype)
        counts[counts < 0] = 0
        return counts

    def hist(
        self,
        axis: str,
        bins: int | Sequence[float | int] = 10,
        inverse: bool = False,
        log: bool = False,
        which_list: str = "list",
        **kwargs,
    ):
        """Compute a histogram of values along a specified axis.

        Parameters
        ----------
        axis : str
            The axis along which to compute the histogram. This can be any column of the merger
            list.

        bins : int | Sequence[float | int], optional
            If an int, it defines the number of equal-width bins in the range. If a sequence, it
            defines the bin edges, including the rightmost edge, allowing for non-uniform bin
            widths. Default is 10.

        inverse : bool, optional
            If True, computes the histogram of the inverse of the values along the specified axis.
            Default is False.

        log : bool, optional
            If True, computes the histogram of the logarithm of the values along the specified axis.
            Default is False.

        which_list : str, optional
            The list method to use for getting the values. Default is "list".

        **kwargs : dict, optional
            Additional keyword arguments that are passed to the `which_list` method. These can be
            used to filter the values that are included in the histogram.


        Returns
        -------
        tuple
            A tuple of two arrays. The first array represents the values of the histogram,
            containing the number of galaxy-galaxy mergers located in the specified cosmic time
            bins. The second array represents the edges of the bins.

        """
        # allow aliasing for axis argument
        axis = key_alias(axis, self.col_alias)

        list_method = getattr(self, which_list)
        values = list_method(**kwargs, columns=[axis])

        if inverse and log:
            return np.histogram(np.log10(1 / values), bins)
        elif inverse:
            return np.histogram(1 / values, bins)
        elif log:
            return np.histogram(np.log10(values), bins)
        else:
            return np.histogram(values, bins)

    def create_lightcone(
        self,
        time_column: str = "Scale",  # This is hardcoded for now
        redshift_bounds: List[float | None] = [0, None],
        randomize: bool = True,
        seed: int | None = None,
        fuzzy_bounds: bool = False,
        method: str = "kw07",
        **kwargs,
    ) -> pd.DataFrame:
        """Temp docstring."""
        if time_column != "Scale":
            raise NotImplementedError("Only 'Scale' column is supported for now.")

        # Attempt to get the class method using the method name string
        try:
            lightcone_init = getattr(Lightcone, method)
        except AttributeError:
            raise NotImplementedError(f"The lightcone init method '{method}' is not implemented.")

        lc_kwargs, kwargs = kwarg_parser(lightcone_init, drop=True, **kwargs)
        lightcone = lightcone_init(lbox=self.box_size / self.cosmology.h, **lc_kwargs)

        # TODO: update to support other time columns (i.e. the output time for each sim snapshot)
        min_z, max_z = redshift_bounds

        # scale factor is decending indicating the first snapshot will be the smallest scale factor
        sim_redshifts = 1 / self.scales - 1

        # if no max redshift is provided infer from the data
        max_z = sim_redshifts.max() if max_z is None else max_z

        if max_z < min_z:
            raise ValueError("The maximum redshift cannot be lower than the minimum redshift")
        if max_z < sim_redshifts.min():
            raise ValueError("The maximum redshift is less than the minimum redshift in the data")

        # the comoving distance to each snapshot in Mpc
        sim_cm_dist = self.cosmology.comoving_distance(sim_redshifts)
        lightcone.set_snap_distances(sim_cm_dist.value[::-1])

        # Create a function to get the comoving distance from a redshift.
        # this is the approach recommended by `astropy` when large number of computations are needed
        # We will need to convert comoving distance to redshift for all galaxies that fall within
        # the lightcone bounds
        _redshift = np.linspace(min_z, max_z, 1000000)
        _cm_dist = self.cosmology.comoving_distance(_redshift).value
        cm_dist_to_redshift = interp1d(_cm_dist, _redshift)

        # minimum and maximum comoving distances for the lightcone
        cm_dist_min, cm_dist_max = self.cosmology.comoving_distance([min_z, max_z]).value

        # determine where the new volumes will be placed
        vert = lightcone.tesselate(cm_dist_min, cm_dist_max)
        galaxy_chunks = []  # Use list to collect DataFrames for efficient concatenation

        # loop over all tesselations that intersect lightcone
        for i, og in enumerate(tqdm(vert)):
            # snap_arg = lightcone.get_snapshots(og)
            bc = lightcone.get_boxcoord(og)

            for j, snapshot_idx in enumerate(lightcone.get_snapshots(og)):
                galaxies = self.list(
                    scale=self.scales[snapshot_idx],
                    columns=[
                        "Scale",
                        "Up_ID",
                        "Halo_radius",
                        "X_pos",
                        "Y_pos",
                        "Z_pos",
                        "X_vel",
                        "Y_vel",
                        "Z_vel",
                        "Type",
                    ],
                    **kwargs,
                )
                # if not galaxies then theres nothing to do
                if len(galaxies) == 0:
                    continue

                galaxies = _transform_and_filter_galaxies(galaxies, lightcone, randomize, bc)
                galaxies = _calculate_galaxy_properties(galaxies, lightcone, method)
                galaxies = _include_fuzzy_bound_galaxies(galaxies, lightcone, randomize, bc)

                # Add columns for lightcone coordinates and apparent redshift
                galaxies[["RA", "Dec"]] = lightcone.ang_coords(
                    galaxies[["X_pos", "Y_pos", "Z_pos"]]
                )

                # Vectorized redshift calculation (more efficient than apply)
                galaxies["Redshift"] = cm_dist_to_redshift(galaxies["R_dist"])
                galaxies["Redshift_obs"] = (
                    galaxies["Redshift"]
                    + galaxies["R_vel"] * (1 + galaxies["Redshift"]) / apconst.c.to("km/s").value
                )

                # Collect DataFrame chunks for efficient concatenation
                galaxy_chunks.append(galaxies)

        if galaxy_chunks:
            lightcone_cat = pd.concat(galaxy_chunks, ignore_index=True)
        else:
            lightcone_cat = pd.DataFrame()  # Empty catalog if no galaxies found

        # Define output columns for consistency
        output_columns = [
            "Tree_ID",
            "Redshift",
            "Redshift_obs",
            "X_cone",
            "Y_cone",
            "Z_cone",
            "R_dist",
            "X_cvel",
            "Y_cvel",
            "Z_cvel",
            "R_vel",
            "RA",
            "Dec",
        ]

        # Handle empty catalog case early
        if lightcone_cat.empty:
            # Return empty catalog with consistent structure
            empty_catalog = pd.DataFrame(columns=output_columns)
            return empty_catalog, lightcone, (min_z, max_z)

        # Add Tree_ID as a proper sequential identifier (more efficient than index manipulation)
        lightcone_cat = lightcone_cat.assign(Tree_ID=range(len(lightcone_cat)))

        # Select only required columns (more memory efficient)
        final_catalog = lightcone_cat[output_columns]

        return final_catalog, lightcone, (min_z, max_z)


# TODO: find a better location for these
def _transform_and_filter_galaxies(galaxies, lightcone, randomize, bc):
    """Transform galaxy positions and velocities, then filter galaxies within the lightcone.

    This function applies coordinate transformations to galaxy positions and optionally
    to velocities if randomization is enabled. It then filters the galaxies to keep
    only those that are contained within the specified lightcone geometry.

    Parameters
    ----------
    galaxies : pandas.DataFrame
        DataFrame containing galaxy data with position columns ["X_pos", "Y_pos", "Z_pos"]
        and velocity columns ["X_vel", "Y_vel", "Z_vel"] if randomize is True.
    lightcone : object
        Lightcone object with methods for transforming positions/velocities and
        checking containment within the lightcone geometry.
    randomize : bool
        If True, also transforms galaxy velocities in addition to positions.
    bc : object
        Box coordinate system object used for the transformations.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only galaxies that are within the lightcone
        after transformation, with updated position (and velocity if randomize=True)
        coordinates.

    """
    galaxies[["X_pos", "Y_pos", "Z_pos"]] = lightcone.transform_position(
        pos=galaxies[["X_pos", "Y_pos", "Z_pos"]].to_numpy(float), randomize=randomize, box_coord=bc
    )

    if randomize:
        galaxies[["X_vel", "Y_vel", "Z_vel"]] = lightcone.transform_velocity(
            vel=galaxies[["X_vel", "Y_vel", "Z_vel"]].to_numpy(float), box_coord=bc
        )

    mask = lightcone.contained(
        galaxies[["X_pos", "Y_pos", "Z_pos"]].to_numpy(float), mask_only=True
    )
    return galaxies.loc[mask]


def _calculate_galaxy_properties(galaxies, lightcone, method):
    """Calculate lightcone-specific properties for galaxies.

    Parameters
    ----------
    galaxies : pd.DataFrame
        DataFrame containing galaxy data with position and velocity columns
    lightcone : Lightcone
        Lightcone object for coordinate transformations
    method : str
        Method for calculating radial distances ("full_width" or other)

    Returns
    -------
    pd.DataFrame
        Updated galaxies DataFrame with additional lightcone properties

    """
    # Define columns to be added
    new_columns = [
        "Redshift",
        "Redshift_obs",
        "X_cone",
        "Y_cone",
        "Z_cone",
        "R_dist",
        "X_cvel",
        "Y_cvel",
        "Z_cvel",
        "R_vel",
        "RA",
        "Dec",
    ]

    # Initialize new columns (more efficient than creating empty DataFrame)
    for col in new_columns:
        galaxies[col] = np.nan

    # Transform positions and velocities to cone coordinates
    position_cols = ["X_pos", "Y_pos", "Z_pos"]
    velocity_cols = ["X_vel", "Y_vel", "Z_vel"]

    cone_positions = lightcone.cone_cartesian(galaxies[position_cols].to_numpy(float))
    cone_velocities = lightcone.cone_cartesian(galaxies[velocity_cols].to_numpy(float))

    galaxies[["X_cone", "Y_cone", "Z_cone"]] = cone_positions
    galaxies[["X_cvel", "Y_cvel", "Z_cvel"]] = cone_velocities

    # Calculate radial distances and velocities
    if method == "full_width":
        galaxies["R_dist"] = galaxies["Z_cone"]
        galaxies["R_vel"] = galaxies["Z_cvel"]
    else:
        # Extract position and velocity arrays
        positions = galaxies[position_cols].to_numpy(float)
        velocities = galaxies[velocity_cols].to_numpy(float)

        # Observer at origin (stationary)
        observer_pos = np.zeros(3)
        observer_vel = np.zeros(3)

        # Calculate radial distance and velocity using halotools
        galaxies["R_dist"], galaxies["R_vel"] = radial_distance_and_velocity(
            *positions.T, *velocities.T, *observer_pos, *observer_vel, np.inf
        )

    return galaxies


def _include_fuzzy_bound_galaxies(galaxies, smin, smax, d_min, d_max, fuzzy_bounds):
    """Include satellites from halos that extend beyond the survey boundary.

    This function applies radial distance cuts and optionally includes satellites
    from central galaxies whose halos extend beyond the maximum survey boundary.

    Parameters
    ----------
    galaxies : pd.DataFrame
        Galaxy catalog with position and halo information
    smin : float
        Minimum survey boundary distance
    smax : float
        Maximum survey boundary distance
    d_min : float
        Minimum data boundary distance
    d_max : float
        Maximum data boundary distance
    fuzzy_bounds : bool
        Whether to include satellites from halos extending beyond smax

    Returns
    -------
    pd.DataFrame
        Filtered galaxy catalog with optional fuzzy boundary satellites

    """
    # Apply radial distance cuts based on survey and data boundaries
    min_distance = np.max([smin, d_min])
    max_distance = np.min([smax, d_max])

    radial_mask = (galaxies["R_dist"] >= min_distance) & (galaxies["R_dist"] < max_distance)

    # Early return if fuzzy bounds not requested
    if not fuzzy_bounds:
        return galaxies.loc[radial_mask]

    # Find central galaxies that extend beyond survey boundary (before applying radial mask)
    central_mask = galaxies["Type"] == 0
    halo_extent = galaxies["R_dist"] + galaxies["Halo_radius"]
    boundary_crossing_mask = central_mask & (halo_extent > smax)

    # Early return if no boundary-crossing centrals
    if not boundary_crossing_mask.any():
        return galaxies.loc[radial_mask]

    # Get IDs of boundary-crossing central galaxies
    boundary_central_ids = galaxies.loc[boundary_crossing_mask].index.values

    # Create mask for satellites of boundary-crossing centrals that are beyond survey boundary
    satellite_fuzzy_mask = (
        galaxies["Up_ID"].isin(boundary_central_ids)
        & (galaxies["R_dist"] >= smax)
        & (galaxies["R_dist"] < d_max)
    )

    # Combine original radial mask with fuzzy boundary satellites
    final_mask = radial_mask | satellite_fuzzy_mask

    return galaxies.loc[final_mask]

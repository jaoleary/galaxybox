"""Classes for handling Emerge data."""

import os

from galaxybox.data.emerge_io import read_statistics
from galaxybox.visualization.emerge_plot import (
    plot_stats_clustering,
    plot_stats_csfrd,
    plot_stats_fq,
    plot_stats_smf,
    plot_stats_ssfr,
)


class EmergeStatistics:
    """Class for Emerge `statistics.h5` file."""

    def __init__(self, statistics_path, add_attrs=None):
        """Initialize an emerge statistics object.

        Parameters
        ----------
        statistics_path : string
            Path to the `statsitics.h5` file
        add_attrs : dict, optional
            A dictionary of additonal attributes to be attach to the class, by default None

        """
        print(statistics_path)
        self.statistics_path = statistics_path
        fdata = read_statistics(self.statistics_path)
        fdatakeys = [key for key in fdata.keys()]
        for k in fdatakeys:
            setattr(self, k, fdata[k])
        if add_attrs:
            for k in add_attrs.keys():
                setattr(self, k, add_attrs[k])

    @classmethod
    def from_universe(cls, universe):
        """Initiate `statistics` within the `Universe` class.

        Parameters
        ----------
        universe : `galaxybox.Universe`
            The outer universe class used to organize this `statistics` class

        Returns
        -------
        galaxybox.statistics
            statistics set from the universe

        """
        add = {
            "out_dir": universe.out_dir,
            "fig_dir": universe.fig_dir,
            "ModelName": universe.ModelName,
            "cosmology": universe.cosmology,
        }
        return cls(os.path.join(universe.out_dir, "statistics.h5"), add)

    def plot(self, obs=None, save=False):
        """Wrap `plot_stats` functions in `emerge_plot.py`.

        Parameters
        ----------
        obs : string or list of strings, optional
            which observables should be plotted. If none all observables are plotted, by default
            None.
        save : bool, optional
            If true observationgs will be saved as a pdf in the designated figure directory, by
            default False.

        """
        # TODO: not a fan of this method, clean up later.

        if obs is None:
            obs = ["CSFRD", "CLUSTERING", "FQ", "SMF", "SSFR"]
        elif isinstance(obs, str):
            obs = [obs]

        obs = [idx.upper() for idx in obs]

        for name in obs:
            if name == "CSFRD":
                plot_stats_csfrd(self.CSFRD, save=save, fig_out=self.fig_dir)
            if name == "CLUSTERING":
                plot_stats_clustering(self.Clustering, save=save, fig_out=self.fig_dir)
            if name == "FQ":
                plot_stats_fq(self.FQ, save=save, fig_out=self.fig_dir)
            if name == "SMF":
                plot_stats_smf(self.SMF, save=save, fig_out=self.fig_dir)
            if name == "SSFR":
                plot_stats_ssfr(self.SSFR, save=save, fig_out=self.fig_dir)

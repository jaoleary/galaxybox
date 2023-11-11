"""Classes for handling Emerge data."""
from ...io.emerge_io import read_statistics
from ...plot.emerge_plot import (
    plot_stats_csfrd,
    plot_stats_clustering,
    plot_stats_fq,
    plot_stats_smf,
    plot_stats_ssfr,
)
import os

__author__ = ("Joseph O'Leary",)


class statistics:
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
        for i, k in enumerate(fdatakeys):
            setattr(self, k, fdata[k])
        if add_attrs:
            for i, k in enumerate(add_attrs.keys()):
                setattr(self, k, add_attrs[k])

    @classmethod
    def from_universe(cls, Universe):
        """Initiate `statistics` within the `Universe` class

        Parameters
        ----------
        Universe : `galaxybox.Universe`
            The outer universe class used to organize this `statistics` class

        Returns
        -------
        galaxybox.statistics
            statistics set from the universe
        """
        add = {
            "out_dir": Universe.out_dir,
            "fig_dir": Universe.fig_dir,
            "ModelName": Universe.ModelName,
            "cosmology": Universe.cosmology,
        }
        return cls(os.path.join(Universe.out_dir, "statistics.h5"), add)

    def plot(self, obs=None, save=False):
        """A wrapper function the for the `plot_stats` functions in `emerge_plot.py`.

        Parameters
        ----------
        obs : string or list of strings, optional
            which observables should be plotted. If none all observables are plotted, by default None.
        save : bool, optional
            If true observationgs will be saved as a pdf in the designated figure directory, by default False.
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

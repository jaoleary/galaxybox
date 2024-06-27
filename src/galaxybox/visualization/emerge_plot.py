"""Module containing common plotting functions for emerge data."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from galaxybox.data.emerge import read_statistics


def plot_stats_csfrd(
    statsfile,
    ax=None,
    universe_num=0,
    observations=True,
    obs_alpha=1.0,
    save=False,
    fig_out=None,
    **kwargs,
):
    """Plot the cosmic star formation rate density (CSFRD) data.

    Parameters
    ----------
    statsfile : str or h5py.Group
        The path to the statistics file or an HDF5 group containing the CSFRD data.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If not provided, a new figure and axes will be created.
    universe_num : int, optional
        The universe number to plot. Default is 0.
    observations : bool, optional
        Whether to plot the observed CSFRD data. Default is True.
    obs_alpha : float, optional
        The transparency of the observed data markers. Default is 1.0.
    save : bool, optional
        Whether to save the plot as a PDF file. Default is False.
    fig_out : str, optional
        The output directory to save the PDF file. If not provided, the file will be saved in the
        current directory.
    **kwargs : dict
        Additional keyword arguments to pass to the `ax.plot` function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    Raises
    ------
    ValueError
        If the `statsfile` is not a valid path or HDF5 group.

    Notes
    -----
    This function plots the cosmic star formation rate density (CSFRD) data from the given
    `statsfile`. The CSFRD data can be either observed or modeled.

    If `statsfile` is a path to an HDF5 file, the CSFRD data will be read from the file.
    The CSFRD data should be stored in the "CSFRD" group of the HDF5 file.

    If `statsfile` is an HDF5 group, it is assumed to contain the CSFRD data directly.

    The observed CSFRD data is plotted as error bars, where the x-axis represents the redshift and
    the y-axis represents the logarithm of the star formation rate density.

    The modeled CSFRD data is plotted as a line.

    Examples
    --------
    >>> plot_stats_csfrd("stats.h5")
    >>> plot_stats_csfrd("stats.h5", ax=my_axes, observations=False, save=True, fig_out="output/")

    """
    mark = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "olive",
        "brown",
        "gold",
        "deepskyblue",
        "lime",
        "orange",
        "navy",
    ]
    labelsize = 16
    # check if the statsfile is an HDF5 group or filepath
    if isinstance(statsfile, str):
        statsfile = read_statistics(statsfile, universe_num=universe_num)["CSFRD"]

    csfrd = statsfile

    # Open plot
    if ax is None:
        xmin = 1.0
        xmax = 15.0
        ymin = -3.9
        ymax = -0.4
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xscale("log")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(r"$z$", size=labelsize)
        ax.set_ylabel(
            (
                r"$\log_{10}( {\rho}_* \, / \, \mathrm{M}_{\odot}"
                r" \, \mathrm{yr}^{-1} \, \mathrm{Mpc}^{-3})$"
            ),
            size=labelsize,
        )
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1, 14, 1)))
        ax.set_xticklabels(("0", "1", "2", "3", "4", "5", "6", " ", "8", " ", "10", " ", "12", " "))

    # Open Data group
    csfrddata = csfrd["Data"]
    csfrddatakeys = [key for key in csfrddata.keys()]

    # Go through each data set and plot each point
    if observations:
        for i, setnum in enumerate(csfrddatakeys):
            csfrdset = csfrddata[setnum]
            x = csfrdset["Redshift"] + 1
            y = csfrdset["Csfrd_observed"]
            s = csfrdset["Sigma_observed"]
            ax.errorbar(
                x,
                y,
                yerr=s,
                marker=mark[i % len(mark)],
                ls="none",
                color=colors[i % len(colors)],
            )

    # Open Model dataset
    csfrdmodel = csfrd["Model"][()]
    x = csfrdmodel[:, 0] + 1
    y = csfrdmodel[:, 1]
    ax.plot(x, y, **kwargs)

    if save is True:
        if fig_out:
            plt.savefig(os.path.join(fig_out, "CSFRD.pdf"), bbox_inches="tight")
        else:
            plt.savefig("CSFRD.pdf", bbox_inches="tight")
    return ax


def plot_stats_clustering(
    statsfile,
    ax=None,
    annotate=False,
    universe_num=0,
    observations=True,
    obs_alpha=1.0,
    save=False,
    fig_out=None,
    **kwargs,
):
    """Plot the clustering statistics from an emerge statistics file.

    Parameters
    ----------
    statsfile : dictionary, string
        A dictionary with required universe statistics. Alternatively, a file
        path can be given.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new figure and axes will be created.

    annotate : bool, optional
        If True, annotations will be added to the plot.

    universe_num : int, optional
        Specifies which universe statistics will be used. Default is 0.

    observations : bool, optional
        If True, observations will be plotted.

    obs_alpha : float, optional
        The alpha value for the observed data points.

    save : bool, optional
        If True, the figure will be saved.

    fig_out : str, optional
        The output directory for saving the figure.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to the plot function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    """
    labelsize = 16
    # check if the statsfile is an HDF5 group or filepath
    if isinstance(statsfile, str):
        statsfile = read_statistics(statsfile, universe_num=universe_num)["Clustering"]

    wp = statsfile

    # Plot the WP sets
    if ax is None:
        annotate = True
        fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")
        xmin = 0.002
        xmax = 90.0
        ymin = 0.8
        ymax = 22000.0
        ax[0, 0].set_xlim([xmin, xmax])
        ax[0, 0].set_ylim([ymin, ymax])

        ax[0, 0].tick_params(
            axis="both",
            direction="in",
            which="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        ax[0, 0].yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax[0, 0].yaxis.set_major_locator(ticker.FixedLocator([1.0, 10.0, 100.0, 1000.0]))
        ax[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator([0.01, 0.1, 1.0, 10.0]))

        ax[0, 0].set_xticklabels(("", "0.1", "1.0", "10.0"))

        ax[0, 0].set_ylabel(r"$w_p \, / \, \mathrm{Mpc}$", size=labelsize)
        ax[1, 0].set_ylabel(r"$w_p \, / \, \mathrm{Mpc}$", size=labelsize)
        ax[1, 0].set_xlabel(r"$r_p \, / \, \mathrm{Mpc}$", size=labelsize)
        ax[1, 1].set_xlabel(r"$r_p \, / \, \mathrm{Mpc}$", size=labelsize)
        ax[1, 2].set_xlabel(r"$r_p \, / \, \mathrm{Mpc}$", size=labelsize)

    # Open Data group
    wpdata = wp["Data"]
    wpdatakeys = [key for key in wpdata.keys()]

    # Define the number of subplots per axis
    nsets = len(wpdatakeys)

    # Go through each data set and make a subplot for each
    for i, axi in enumerate(ax.reshape(-1)):
        if i < nsets:
            wpset = wpdata[wpdatakeys[i]]
            xo = wpset["Radius"]
            ym = wpset["Wp_model"]

            if observations:
                yo = wpset["Wp_observed"]
                so = wpset["Sigma_observed"]
                axi.errorbar(xo, yo, yerr=so, marker="x", ls="none", color="red", alpha=obs_alpha)
            axi.plot(xo, ym, **kwargs)
            if annotate:
                axi.annotate(
                    wpdatakeys[i][4:-10],
                    xy=(0.05, 1 - 0.05),
                    xycoords="axes fraction",
                    size=14,
                    ha="left",
                    va="top",
                )

    if save is True:
        if fig_out:
            plt.savefig(os.path.join(fig_out, "clustering.pdf"), bbox_inches="tight")
        else:
            plt.savefig("clustering.pdf", bbox_inches="tight")
    return ax


def plot_stats_fq(
    statsfile,
    ax=None,
    annotate=False,
    universe_num=0,
    observations=True,
    obs_alpha=1.0,
    save=False,
    fig_out=None,
    **kwargs,
):
    """Plot the quenched fraction from an emerge statistics file.

    Parameters
    ----------
    statsfile : dictionary, string
        A dictionary with required universe statistics. Alternatively, a file
        path can be given for the statistics file.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the quenched fraction. If not provided, a new
        set of axes will be created.

    annotate : bool, optional
        If True, annotate each subplot with the redshift range.

    universe_num : int, optional
        Specifies which universe statistics will be used. Default is 0.

    observations : bool, optional
        If True, plot the observed data points along with the model.

    obs_alpha : float, optional
        The transparency of the observed data points. Default is 1.0.

    save : bool, optional
        If True, save the figure as 'quenched_fraction.pdf' in the current working directory.

    fig_out : str, optional
        The directory where the figure will be saved. If provided, the figure will be saved
        in this directory instead of the current working directory.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `plot` function of matplotlib.

    Returns
    -------
    ax : matplotlib.axes.Axes
        ax object

    """
    mark = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "olive",
        "brown",
        "gold",
        "deepskyblue",
        "lime",
        "orange",
        "navy",
    ]
    labelsize = 16
    # check if the statsfile is an HDF5 group or filepath
    if isinstance(statsfile, str):
        statsfile = read_statistics(statsfile, universe_num=universe_num)["FQ"]

    fq = statsfile

    # Plot the FQ sets

    if ax is None:
        annotate = True
        fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        xmin = 8.3
        xmax = 12.1
        ymin = 0.0
        ymax = 1.0
        ax[0, 0].set_xlim([xmin, xmax])
        ax[0, 0].set_ylim([ymin, ymax])
        ax[0, 0].tick_params(
            axis="both",
            direction="in",
            which="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )

        ax[0, 0].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, 1, 0.05)))
        ax[0, 0].yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1, 0.2)))
        ax[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(7, 13, 0.25)))
        ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator(np.arange(7, 13, 1)))
        ax[0, 0].set_ylabel(r"$f_q$", size=labelsize)
        ax[1, 0].set_ylabel(r"$f_q$", size=labelsize)
        ax[1, 0].set_xlabel("$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)
        ax[1, 1].set_xlabel("$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)
        ax[1, 2].set_xlabel("$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)

    zrange = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    zrange = np.array(zrange)
    zmean = 0.5 * (zrange[:-1] + zrange[1:])

    # Open Data group
    fqdata = fq["Data"]
    fqdatakeys = [key for key in fqdata.keys()]

    # Open Set compound
    fqset = fq["Sets"]
    fqsetzmin = fqset["Redshift_min"]
    fqsetzmax = fqset["Redshift_max"]
    fqsetzmean = 0.5 * (fqsetzmin + fqsetzmax)

    # Open Model dataset
    fqmodel = fq["Model"]
    zmodel = 1.0 / fqmodel[()][0, 1:] - 1.0
    msmodel = fqmodel[()][1:, 0]
    fqmodel = fqmodel[()][1:, 1:]

    # Find the model index for each redshift bin
    idz = np.zeros(zmean.size, dtype=int)
    for iz in range(0, zmean.size):
        idz[iz] = (np.abs(zmodel - zmean[iz])).argmin()

    # Go through each data set and make a subplot for each
    for i, axi in enumerate(ax.reshape(-1)):
        axi.plot(msmodel, fqmodel[:, idz[i]], **kwargs)
        if annotate:
            axi.annotate(
                "{} < z < {}".format(zrange[i], zrange[i + 1]),
                xy=(0.05, 1 - 0.05),
                xycoords="axes fraction",
                size=labelsize,
                ha="left",
                va="top",
            )
        if observations:
            # Go through all sets
            for iset, setnum in enumerate(fqdatakeys):
                # If set is in right redshift range for this subplot
                if fqsetzmean[iset] >= zrange[i] and fqsetzmean[iset] < zrange[i + 1]:
                    # Load set
                    fqset = fqdata[setnum]
                    x = fqset["Stellar_mass"]
                    y = fqset["Fq_observed"]
                    s = fqset["Sigma_observed"]
                    axi.errorbar(
                        x,
                        y,
                        yerr=s,
                        marker=mark[iset % len(mark)],
                        ls="none",
                        color=colors[iset % len(colors)],
                        alpha=obs_alpha,
                    )

    if save is True:
        if fig_out:
            plt.savefig(os.path.join(fig_out, "quenched_fraction.pdf"), bbox_inches="tight")
        else:
            plt.savefig("quenched_fraction.pdf", bbox_inches="tight")
    return ax


def plot_stats_smf(
    statsfile,
    ax=None,
    annotate=False,
    universe_num=0,
    observations=True,
    obs_alpha=1.0,
    save=False,
    fig_out=None,
    **kwargs,
):
    """Plot the stellar mass function from an emerge statistics file.

    Parameters
    ----------
    statsfile : dictionary, string
        A dictionary with required universe statistics. Alternatively, a file
        path can be given.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the stellar mass function. If not provided,
        a new set of axes will be created.

    annotate : bool, optional
        If True, annotate each subplot with the redshift range.

    universe_num : int, optional
        Specifies which universe statistics will be used. Default is 0.

    observations : bool, optional
        If True, plot the observed data points along with the model.

    obs_alpha : float, optional
        The transparency of the observed data points. Default is 1.0.

    save : bool, optional
        If True, save the figure as 'SMF.pdf' in the current working directory.

    fig_out : str, optional
        The output directory to save the figure. If provided, the figure will be
        saved as 'SMF.pdf' in the specified directory.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `plot` function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the stellar mass function is plotted.

    """
    mark = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "olive",
        "brown",
        "gold",
        "deepskyblue",
        "lime",
        "orange",
        "navy",
    ]
    labelsize = 16
    # check if the statsfile is an HDF5 group or filepath
    if isinstance(statsfile, str):
        statsfile = read_statistics(statsfile, universe_num=universe_num)["SMF"]

    smf = statsfile

    zrange = np.array([0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    zmean = 0.5 * (zrange[:-1] + zrange[1:])

    # Open Data group
    smfdata = smf["Data"]
    smfdatakeys = [key for key in smfdata.keys()]

    # Open Set compound
    smfset = smf["Sets"]
    smfsetzmin = smfset["Redshift_min"]
    smfsetzmax = smfset["Redshift_max"]
    smfsetzmean = 0.5 * (smfsetzmin + smfsetzmax)

    # Open Model dataset
    smfmodel = smf["Model"]
    zmodel = 1.0 / smfmodel[()][0, 1:] - 1.0
    msmodel = smfmodel[()][1:, 0]
    smfmodel = smfmodel[()][1:, 1:]

    # Find the model index for each redshift bin
    idz = np.zeros(zmean.size, dtype=int)
    for iz in range(0, zmean.size):
        idz[iz] = (np.abs(zmodel - zmean[iz])).argmin()

    if ax is None:
        annotate = True
        fig, ax = plt.subplots(3, 4, figsize=(16, 10), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        xmin = 7.1
        xmax = 12.4
        ymin = -5.9
        ymax = -0.8

        ax[0, 0].tick_params(
            axis="both",
            direction="in",
            which="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        ax[0, 0].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-6, 0, 0.25)))
        ax[0, 0].yaxis.set_major_locator(ticker.FixedLocator(np.arange(-6, 0, 1)))
        ax[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(7, 13, 0.25)))
        ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator(np.arange(7, 13, 1)))
        ax[0, 0].axis([xmin, xmax, ymin, ymax])

        ax[2, 0].set_xlabel(r"$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)
        ax[2, 1].set_xlabel(r"$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)
        ax[2, 2].set_xlabel(r"$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)
        ax[2, 3].set_xlabel(r"$\log_{10}(m_* / \mathrm{M}_{\odot})$", size=labelsize)

        ax[0, 0].set_ylabel(
            r"$\log_{10}(\Phi \, / \, \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$",
            size=labelsize,
        )
        ax[1, 0].set_ylabel(
            r"$\log_{10}(\Phi \, / \, \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$",
            size=labelsize,
        )
        ax[2, 0].set_ylabel(
            r"$\log_{10}(\Phi \, / \, \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$",
            size=labelsize,
        )

    for i, axi in enumerate(ax.reshape(-1)):
        # Go through all sets
        axi.plot(msmodel, smfmodel[:, idz[i]], **kwargs)
        if observations:
            for iset, setnum in enumerate(smfdatakeys):
                # If set is in right redshift range for this subplot
                if smfsetzmean[iset] >= zrange[i] and smfsetzmean[iset] < zrange[i + 1]:
                    # Load set
                    smfset = smfdata[setnum]
                    x = smfset["Stellar_mass"]
                    y = smfset["Phi_observed"]
                    s = smfset["Sigma_observed"]
                    axi.errorbar(
                        x,
                        y,
                        yerr=s,
                        marker=mark[iset % len(mark)],
                        ls="none",
                        color=colors[iset % len(colors)],
                        alpha=obs_alpha,
                    )

        if annotate:
            axi.annotate(
                "${} < z \leq {}$".format(zrange[i], zrange[i + 1]),
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                size=labelsize,
                ha="left",
                va="bottom",
            )

    if save is True:
        if fig_out:
            plt.savefig(os.path.join(fig_out, "SMF.pdf"), bbox_inches="tight")
        else:
            plt.savefig("SMF.pdf", bbox_inches="tight")

    return ax


def plot_stats_ssfr(
    statsfile,
    ax=None,
    annotate=False,
    universe_num=0,
    observations=True,
    obs_alpha=1.0,
    save=False,
    fig_out=None,
    **kwargs,
):
    """Plot the specific star formation rates from an emerge statistics file.

    Parameters
    ----------
    statsfile : dictionary, string
        A dictionary with required universe statistics. Alternatively, a file
        path can be given for and HDF5 group.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the data. If not provided, a new figure and axes will be created.

    annotate : bool, optional
        If True, annotations will be added to the plot.

    universe_num : int, optional
        Specifies which universe statistics will be used. Default is 0.

    observations : bool, optional
        If True, observed data will be plotted along with the model data.

    obs_alpha : float, optional
        The transparency of the observed data markers. Default is 1.0.

    save : bool, optional
        If True, the figure will be saved to the specified file path or the current working
        directory as 'SSFR.pdf'.

    fig_out : str, optional
        The file path to save the figure. If not provided, the figure will be saved in the current
        working directory.

    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `plot` function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the data is plotted.

    """
    mark = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "olive",
        "brown",
        "gold",
        "deepskyblue",
        "lime",
        "orange",
        "navy",
    ]
    labelsize = 16
    # check if the statsfile is an HDF5 group or filepath
    if isinstance(statsfile, str):
        statsfile = read_statistics(statsfile, universe_num=universe_num)["SSFR"]

    ssfr = statsfile

    mrange = [8.0, 9.0, 10.0, 11.0, 12.0]
    mrange = np.array(mrange)
    mmean = 0.5 * (mrange[:-1] + mrange[1:])

    # Open Data group
    ssfrdata = ssfr["Data"]
    ssfrdatakeys = [key for key in ssfrdata.keys()]

    # Open Model dataset
    ssfrmodel = ssfr["Model"]
    msmodel = ssfrmodel[()][0, 1:]
    zmodel = ssfrmodel[()][1:, 0]
    ssfrmodel = ssfrmodel[()][1:, 1:]

    # Find the model index for each mass bin
    idm = np.zeros(mmean.size, dtype=int)
    for im in range(0, mmean.size):
        idm[im] = (np.abs(msmodel - mmean[im])).argmin()

    if ax is None:
        annotate = True
        fig, ax = plt.subplots(2, 2, figsize=(16, 10), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        ax[0, 0].tick_params(
            axis="both",
            direction="in",
            which="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        xmin = 1.0
        xmax = 15.0
        ymin = -12.3
        ymax = -7.6
        ax[0, 0].axis([xmin, xmax, ymin, ymax])
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_ylabel(r"$\log_{10}(\mathrm{sSFR} / \mathrm{yr}^{-1})$", size=labelsize)
        ax[1, 0].set_ylabel(r"$\log_{10}(\mathrm{sSFR} / \mathrm{yr}^{-1})$", size=labelsize)
        ax[1, 0].set_xlabel(r"$z$", size=labelsize)
        ax[1, 1].set_xlabel(r"$z$", size=labelsize)

    # Go through each data set and make a subplot for each

    for i, axi in enumerate(ax.reshape(-1)):
        axi.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1, 14, 1)))
        axi.set_xticklabels(
            ("0", "1", "2", "3", "4", "5", "6", " ", "8", " ", "10", " ", "12", " ")
        )
        # Make subplot for this redshift bin
        axi.plot(zmodel + 1, ssfrmodel[:, idm[i]], **kwargs)

        if observations:
            # Go through all sets
            for iset, setnum in enumerate(ssfrdatakeys):
                ssfrset = ssfrdata[setnum]
                x = ssfrset["Redshift"]
                y = ssfrset["Ssfr_observed"]
                s = ssfrset["Sigma_observed"]
                m = ssfrset["Stellar_mass"]
                # Go through all data points in this set
                for issfr in range(0, ssfrset.size):
                    # If the stellar mass of the point is in the subplot
                    if m[issfr] >= mrange[i] and m[issfr] < mrange[i + 1]:
                        # print(x[issfr],y[issfr],s[issfr])
                        axi.errorbar(
                            x[issfr] + 1.0,
                            y[issfr],
                            yerr=s[issfr],
                            marker=mark[iset % len(mark)],
                            ls="none",
                            color=colors[iset % len(colors)],
                            alpha=obs_alpha,
                        )

        if annotate:
            axi.annotate(
                "${} < \log(M/M_\odot) < {}$".format(mrange[i], mrange[i + 1]),
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                size=labelsize,
                ha="left",
                va="bottom",
            )

    if save is True:
        if fig_out:
            plt.savefig(os.path.join(fig_out, "SSFR.pdf"), bbox_inches="tight")
        else:
            plt.savefig("SSFR.pdf", bbox_inches="tight")

    return ax


def plot_efficiency(
    glist: pd.DataFrame,
    redshift: float,
    f_baryon: float,
    ax: Optional[plt.Axes] = None,
    frac: float = 0.15,
    min_mass: float = 10.5,
    max_mass: float = np.inf,
    mass_def: str = "Halo",
    peak_mass: bool = True,
    use_obs: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = False,
    labelfs: int = 18,
) -> plt.Axes:
    """Plot the efficiency of star formation.

    Parameters
    ----------
    glist : pd.DataFrame
        DataFrame containing the galaxy properties.
    redshift : float
        The redshift value.
    f_baryon : float
        The baryon fraction.
    ax : plt.Axes, optional
        The axes on which to plot the data. If not provided, a new figure and axes will be created.
    frac : float, optional
        The fraction of galaxies to sample. Default is 0.15.
    min_mass : float, optional
        The minimum mass threshold. Default is 10.5.
    max_mass : float, optional
        The maximum mass threshold. Default is np.inf.
    mass_def : str, optional
        The mass definition. Default is "Halo".
    peak_mass : bool, optional
        Whether to use peak mass. Default is True.
    use_obs : bool, optional
        Whether to use observed data. Default is True.
    vmin : float, optional
        The minimum value for the color scale. Default is None.
    vmax : float, optional
        The maximum value for the color scale. Default is None.
    colorbar : bool, optional
        Whether to show the colorbar. Default is False.
    labelfs : int, optional
        The font size for labels. Default is 18.

    Returns
    -------
    plt.Axes
        The axes on which the data is plotted.

    """
    if peak_mass & (mass_def == "Halo"):
        mt = "_peak"
    else:
        mt = ""

    if use_obs:
        obs = "_obs"
    else:
        obs = ""

    mass_mask = (glist[mass_def + "_mass" + mt] >= min_mass) & (
        glist[mass_def + "_mass" + mt] < max_mass
    )
    data = glist.loc[mass_mask].sample(frac=frac)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.set_yscale("log")
        ax.set_ylabel("$m_{*}/m_{\mathrm{b}}$", size=labelfs)
        ax.set_xlabel("$\log_{10}(M_{\mathrm{h}}/\mathrm{M}_{\odot})$", size=labelfs)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, pos: ("{{:.{:1d}f}}".format(int(np.maximum(-np.log10(y), 0)))).format(y)
            )
        )

    ax.tick_params(
        axis="both",
        direction="in",
        which="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelsize=labelfs,
    )
    ax.set_ylim([0.005, 1])
    ax.set_xticks(np.arange(11, 15 + 1, 1))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(11.5, 14.5 + 1, 1)))
    ax.set_xlim([min_mass, 15])

    color = np.log10(data["SFR_obs"] / (10 ** data["Stellar_mass_obs"]))
    if vmin is None:
        color.loc[color == -np.inf] = -15
    else:
        color.loc[color == -np.inf] = vmin

    ln = ax.scatter(
        data["Halo_mass" + mt],
        10 ** data["Stellar_mass" + obs] / ((10 ** data["Halo_mass" + mt]) * f_baryon),
        s=5,
        c=color,
        cmap=plt.cm.jet_r,
        vmin=vmin,
        vmax=vmax,
    )
    ax.annotate(
        "$z={:.1f}$".format(redshift),
        xy=(1 - 0.05, 1 - 0.05),
        xycoords="axes fraction",
        size=labelfs,
        ha="right",
        va="top",
    )
    if colorbar:
        cbar = plt.colorbar(ln)
        cbar.set_label("$\log_{10}(\mathrm{sSFR})$", fontsize=labelfs)
        cbar.ax.tick_params(labelsize=labelfs)

    return ln

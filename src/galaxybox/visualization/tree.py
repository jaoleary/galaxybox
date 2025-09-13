"""Module for visualizing trees."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes


class TemporalTreePlotter:
    """A class for plotting temporal trees with matplotlib.

    This class assumes the trees have depth first indexing and are sorted by scale factor.

    Parameters
    ----------
    min_scale : int, optional
        The minimum scale factor to consider for plotting, by default 0.
    max_mu : int, optional
        The maximum mass ratio (mu) to consider for plotting, by default 10000.
    plot_style : str, optional
        The style of the plotted tree. Can be "centered" or "linear", by default "centered".

    Attributes
    ----------
    min_scale : int
        The minimum scale factor to consider for plotting.
    max_mu : int
        The maximum mass ratio (mu) to consider for plotting.
    plot_style : str
        The style of the plotted tree. Can be "centered" or "linear", by default "centered".
    time_col : str
        Column name for the time scale in the tree data.
    coprog_id_str : str
        Column name for the coprogenitor ID in the tree data.
    desc_id_str : str
        Column name for the descendant ID in the tree data.
    mmp_str : str
        Column name for the most massive progenitor flag in the tree data.
    num_prog_str : str
        Column name for the number of progenitors in the tree data.
    mass_str : str
        Column name for the stellar mass in the tree data.
    tree : pandas.DataFrame or None
        The tree data.
    pos : pandas.DataFrame or None
        The positions of nodes in the tree.
    node : int or None
        The current node being processed.
    current_y : int
        The current y position in the plot.
    scale : int
        The scale factor for plotting.

    """

    def __init__(self, min_scale: int = 0, max_mu: float = 10000, plot_style: str = "centered"):
        if min_scale < 0:
            raise ValueError("Minimum scale factor must be greater than or equal to 0.")
        if max_mu < 1:
            raise ValueError("Maximum mass ratio must be greater than or equal to 1.")
        if plot_style not in ["centered", "linear"]:
            raise ValueError("Invalid plot style. Must be 'centered' or 'linear'.")

        self.min_scale = min_scale
        self.max_mu = max_mu
        self.plot_style = plot_style

        # TODO: allow for custom column names / aliasing
        # Strings for the tree columns
        self.time_col: str = "Scale"
        self.coprog_id_str: str = "Coprog_ID"
        self.desc_id_str: str = "Desc_ID"
        self.mmp_str: str = "MMP"
        self.num_prog_str: str = "Num_prog"
        self.mass_str: str = "Stellar_mass"

        self.tree: pd.DataFrame | None = None
        self.pos: pd.DataFrame | None = None
        self.node: int | None = None
        self.current_y: int = 0

        self.scale: float = 1.0

    def __call__(self, tree: pd.DataFrame, ax: Axes | None = None):
        """Call the Tree object as a function.

        Parameters
        ----------
        tree : pd.DataFrame
            The input data for the tree.
        ax : Optional[matplotlib.axes.Axes], optional
            The matplotlib axes to render the tree on. If not provided, a new axes will be created.

        Returns
        -------
        tuple(fig, ax)
            The matplotlib figure and axes.

        Raises
        ------
        ValueError
            If the input data is invalid.

        Notes
        -----
        This method initializes the tree, sets the position, renders the tree, and resets the
        internal state.

        Examples
        --------
        >>> plot_tree = TemporalTreePlotter()
        >>> plot_tree(data)

        """
        self._init_tree(tree)

        if ax is None:
            self._init_ax()
            fig = self.fig
        else:
            self.ax = ax
            fig = None

        self.set_pos()
        self.render()
        self._reset()
        return fig, ax

    def _reset(self):
        """Reset the internal state of the plotter."""
        self.tree = None
        self.pos = None
        self.node = None
        self.current_y = 0

    def _init_tree(self, tree: pd.DataFrame):
        """Initialize the tree with the given DataFrame.

        Parameters
        ----------
        tree : pd.DataFrame
            The input data for the tree.

        """
        self.tree = tree

        self.pos = pd.DataFrame(index=self.tree.index.values, columns=["x", "y"])
        self.pos.index.name = "id"
        self.pos["x"] = self.tree[self.time_col].values

        self.node = self.tree.index[0]

    def set_pos(self):
        """Set the position of the current node."""
        # set the position of current node
        self.pos.loc[self.node, "y"] = self.current_y
        self.scale = self.tree.loc[self.node, self.time_col]

        # get the ID of the most massive progenitor
        immp = 0
        if self.tree.loc[self.node, self.num_prog_str] > 0:
            immp = self.tree.loc[
                (self.tree[self.desc_id_str] == self.node) & (self.tree[self.mmp_str] == 1)
            ].index.values[0]

        # doing a mass ratio check on coprogs
        icoprog = self._get_coprog_id()

        if self.scale > self.min_scale:
            # walk the main branch first
            if immp > 0:
                self.node = immp
                self.set_pos()
            # walk the coprogenitors
            if icoprog > 0:
                self.current_y += 1
                self.node = icoprog
                self.set_pos()

            if self.plot_style == "centered":
                self._update_desc_pos()

    def _get_coprog_id(self):
        """Get the ID of the coprogenitor node.

        Returns
        -------
        int
            The ID of the coprogenitor node.

        """
        icoprog = int(self.tree.loc[self.node][self.coprog_id_str])
        if icoprog > 0:
            icoprog = self.tree.loc[self.tree.index == icoprog].index.values[0]
            dat_c = self.tree.loc[icoprog]
            did_c = int(dat_c[self.desc_id_str])
            main_prog = self.tree.loc[
                (self.tree[self.desc_id_str] == did_c) & (self.tree[self.mmp_str] == 1)
            ]
            mu = 10 ** (main_prog[self.mass_str].values[0] - dat_c[self.mass_str])
            if mu > self.max_mu:
                icoprog = 0
        return icoprog

    def _get_mmp_id(self, idx: int):
        """Get the ID of the most massive progenitor (MMP).

        Parameters
        ----------
        idx : int
            The index of the current node.

        Returns
        -------
        int
            The ID of the MMP.

        """
        if self.tree.loc[idx, self.num_prog_str] > 0:
            return self.tree.loc[
                (self.tree[self.desc_id_str] == idx) & (self.tree[self.mmp_str] == 1)
            ].index.values[0]

    def _update_desc_pos(self):
        """Update the position of descendant nodes."""
        desc_id = self.tree.loc[self.node, self.desc_id_str]
        max_depth = 1
        depth = 0

        while (desc_id > 0) and (depth < max_depth):
            if self.tree.loc[self.node, self.mmp_str] == 0:
                immp = self._get_mmp_id(desc_id)
                mmp_y = self.pos.loc[immp, "y"]
                now_y = self.pos.loc[self.node, "y"]

                self.pos.loc[desc_id, "y"] = (now_y + mmp_y) / 2
                depth += 1
            else:
                self.pos.loc[desc_id, "y"] = self.pos.loc[self.node, "y"]

            self.node = desc_id
            desc_id = self.tree.loc[self.node, self.desc_id_str]

    def render(self):
        """Render the tree visualization."""
        # some indices are blank since they didnt meet mu or scale cuts
        # so we need to ignore those
        valid_idx = ~self.pos["y"].isna()
        for idx in self.pos[valid_idx].index.values:
            if self.tree.loc[idx, self.desc_id_str] == 0:
                continue

            desc_id = self.tree.loc[idx, self.desc_id_str]
            x = [self.pos.loc[idx, "x"], self.pos.loc[desc_id, "x"]]
            y = [self.pos.loc[idx, "y"], self.pos.loc[desc_id, "y"]]
            self.ax.plot(x, y, color="k", zorder=0)

        self.ax.scatter(
            self.pos.loc[valid_idx, "x"],
            self.pos.loc[valid_idx, "y"],
            c=self.tree.loc[valid_idx, self.mass_str],
            zorder=10,
            vmin=9,
            vmax=self.tree[self.mass_str].max(),
            cmap=plt.cm.jet,
        )

    def _init_ax(self):
        """Initialize the matplotlib axes for the plot."""
        vmin = 7
        vmax = self.tree[self.mass_str].max()
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.1, 0.1)))
        self.ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, 1, 0.025)))
        self.ax.set_xlabel("Scale factor", fontsize=18)
        self.ax.tick_params(axis="y", labelleft=False)
        self.ax.set_yticks([])

        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        self.cbar = plt.colorbar(sm, ax=self.ax, pad=0.01, aspect=30)
        self.cbar.ax.xaxis.set_label_position("top")
        self.cbar.ax.xaxis.set_ticks_position("top")
        self.cbar.ax.minorticks_on()
        self.cbar.set_label(r"$\log_{10}(M_{\mathrm{*}}/M_{\odot})$", fontsize=18)

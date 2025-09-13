"""Some general use plotting functions."""

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def linestyle(sequence=None, dash=None, dot=None, space=None, buffer=False, offset=0):
    """Generate more linetypes from `-`,`.` combinations with added IDL like linetype shortcuts.

    Parameters
    ----------
    sequence : string
        An arbitary sequence of `-` and `.` to construct linestyles. Default
        behavior acts just like normal matplotlib `linestyle` arguments. Added
        IDl like shortcuts for long dashes `__` and dash-dot-dot-dot `-:`.
        Underscore is a double dash, and semicolon is a tripple dot.(the default is None).
    dash : float
        length of line dashes (the default is None).
    dot : float
        length of line dots (the default is None).
    space : float
        length of blank spaces in lines (the default is None).
    buffer : float
        lenght of trailing buffer space for a given sequence (the default is False).
    offset : float
        Description of parameter `offset` (the default is 0).

    Returns
    -------
    linestyle : tuple
        tuple containing the matplotlib linestyle `offset`, and `onoffseq`

    """
    if (sequence is None) or (sequence == "-") or (sequence == "_"):
        return "-"
    elif sequence == ":":
        return sequence
    else:
        if dash is None:
            dash = plt.rcParamsDefault["lines.dashed_pattern"][0]
            if (sequence.count(".") > 0) or (sequence.count(":") > 0):
                dash = plt.rcParamsDefault["lines.dashdot_pattern"][0]

        if dot is None:
            dot = plt.rcParamsDefault["lines.dotted_pattern"][0]

        if space is None:
            space = plt.rcParamsDefault["lines.dashed_pattern"][1]

        reftype = {}
        reftype["-"] = [dash, space]
        reftype["_"] = [2 * dash, space]
        reftype["."] = [dot, space]
        reftype[":"] = [dot, space, dot, space, dot, space]
        onoffseq = []
        for i, s in enumerate(sequence):
            onoffseq.extend(reftype[s])
        if buffer:
            onoffseq[-1] = buffer
        return (offset, onoffseq)


def ls(sequence, **kwargs):
    """Shortcut to the linestyle function."""
    return linestyle(sequence, **kwargs)


def render_cube(ax, origin, length):
    """Render a cube in a 3D plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D plot axes.
    origin : array-like
        The origin of the cube.
    length : float
        The length of the cube's edges.

    Returns
    -------
    ln : matplotlib.lines.Line3D
        The lines representing the cube.

    """
    # Adapted from
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    r = [0, length]
    og = np.atleast_1d(origin)
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ln = ax.plot3D(*zip(og + np.array(s), og + np.array(e)), color="b", alpha=0.25)
    return ln


class Arrow3D(FancyArrowPatch):
    """A 3D arrow class for matplotlib plots."""

    def __init__(self, xs: float, ys: float, zs: float, *args, **kwargs):
        """Initialize the Arrow3D object.

        Parameters
        ----------
        xs : float
            The x-coordinate of the arrow's start point.
        ys : float
            The y-coordinate of the arrow's start point.
        zs : float
            The z-coordinate of the arrow's start point.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the Arrow3D object.

        Parameters
        ----------
        renderer : RendererBase
            The renderer object used for drawing.

        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

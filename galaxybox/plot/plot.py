"""
Some general use plotting functions
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from itertools import product, combinations

__author__ = ('Joseph O\'Leary', )

def linestyle(sequence=None, dash=None, dot=None, space=None, buffer=False, offset=0):
    """Generate more linetypes from arbitrary `-`,`.` combinations with added
       IDL like linetype shortcuts.

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
    if (sequence is None) or (sequence == '-') or (sequence == '_'):
        return '-'
    elif sequence == ':':
        return sequence
    else:
        if dash is None:
            dash = plt.rcParamsDefault['lines.dashed_pattern'][0]
            if (sequence.count('.') > 0) or (sequence.count(':') > 0):
                dash = plt.rcParamsDefault['lines.dashdot_pattern'][0]

        if dot is None:
            dot = plt.rcParamsDefault['lines.dotted_pattern'][0]

        if space is None:
            space = plt.rcParamsDefault['lines.dashed_pattern'][1]

        reftype = {}
        reftype['-'] = [dash, space]
        reftype['_'] = [2 * dash, space]
        reftype['.'] = [dot, space]
        reftype[':'] = [dot, space, dot, space, dot, space]
        onoffseq = []
        for i, s in enumerate(sequence):
            onoffseq.extend(reftype[s])
        if buffer:
            onoffseq[-1] = buffer
        return (offset, onoffseq)


def ls(sequence, **kwargs):
    """Shortcut to the linestyle function."""
    return linestyle(sequence, **kwargs)


def render_cube(ax,O,L):
    # Adapted from
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    r = [0, L]
    og = np.atleast_1d(O)
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ln = ax.plot3D(*zip(og+np.array(s), og+np.array(e)), color="b",alpha=0.25)
    return ln


class Arrow3D(FancyArrowPatch):
    # Adapted from
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

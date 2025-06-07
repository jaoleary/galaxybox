"""Various useful functions."""

import os
import subprocess

import numpy as np


def cmd(args, path=None, encoding="utf-8", **kwargs):
    """Execute command line operations with live output to notebook cells.

    Parameters
    ----------
    args : string, sequence
        A string, or a sequence of program arguments.
    path : string
        the directory in which the commands should be executed (the default is None).
    encoding : string
        Byte encoding for screen output (the default is 'utf-8').
    **kwargs : dict
        Other arguments accepted by `subprocess.Popen`.

    """
    old_dir = os.getcwd()
    if path is not None:
        os.chdir(path)
    else:
        os.chdir(old_dir)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, **kwargs)
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        print(line.decode(encoding))
    os.chdir(old_dir)


def shuffle_string(s: str) -> str:
    """Randomly shuffle a string.

    Parameters
    ----------
    s : str
        String to be shuffled

    Returns
    -------
    str
        The shuffled string

    """
    s = list(s)
    np.random.shuffle(s)
    return "".join(s)


def translate(pos, lbox, dx=0, axes=0):
    """Translate coordinates.

    Parameters
    ----------
    pos : 2-D array
        An array of size (N, 3) containing the 3D cartesian positions for N
        points in space.
    lbox : float
        Comoving cosmological box side length.
    dx : float, array_like
        The magnitude of translation along each axis
    axes : int, array_like
        Which axes the translation should be applied to

    Returns
    -------
    pos : 2-D array
        The new positions after translation

    """
    pos = np.atleast_2d(pos)
    dx = np.atleast_1d(dx)
    axes = np.atleast_1d(axes)
    for i, a in enumerate(axes):
        pos[:, a] += dx[i]
        above = np.where(pos[:, a] >= lbox)
        pos[above, a] -= lbox
    return pos


def poly_traverse(coords: np.ndarray, ccw: bool = True):
    """Return a (counter) clockwise coordinate list.

    This function takes a list of coordinates corresponding to the 2D vertex
    coordinates of a convex polygon and reorders them such that the verticies
    are traversed in a (counter) clockwise order.

    Parameters
    ----------
    coords : 2-D np.ndarray
        An array of size (N, 2) where N corresponds to the number of vertices on
        a polygon.
    ccw : boolean
        If true the coordinates will be returned in counter-clockwise order.
        Otherwise coordinates will be returned in clockwise order (the default is True).

    Returns
    -------
    ordered_coords : 2-D array
        Reordered coordinate list of size (N,2)

    """
    # First locate the geometric center of the polygon.
    center = np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1])])
    # Next, find the angle with respect to the geometric center and each vertex.
    # Coordinates are then sorted in order of ascending or descending angle
    # depending on whethere clockwise or counter-clockwise orientation is desired.
    order = np.argsort(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))
    if ccw:
        return coords[order]
    else:
        return coords[order[::-1]]


def coordinate_plane(origin=np.array([0, 0, 0]), lbox=1, axes=[0, 1]):
    """Create a square coordinate plane along desired axes starting at some origin.

    Parameters
    ----------
    origin : array_like
        The desired origin for the plane in 3D space (the default is np.array([0, 0, 0])).
    lbox : type
        The side length of the plane (the default is 1).
    axes : array_like
        The principle axes used to create the plane (the default is [0, 1]).

    Returns
    -------
    points : array_like
        The vertex locations of the plane in 3D space.

    """
    points = np.zeros((4, 3))
    y = np.arange(3)
    axes = np.atleast_1d(axes)

    norm = np.setdiff1d(y, axes)[0]
    points[:, norm] = origin[norm]
    points[0, :] = origin

    i = 0
    for aj in range(2):
        for ai in range(2):
            points[i, [axes[0], axes[1]]] = (
                origin[axes[0]] + ai * lbox,
                origin[axes[1]] + aj * lbox,
            )
            i += 1

    return points


def rotate(vec, angle=0, u=[1, 0, 0]):
    """Rotate a vector wrt an arbitrary unit vector.

    Rotates an input vector about some arbitrary unit vector `u` using the
    Rodrigues rotation formula.

    Parameters
    ----------
    vec : 1-D array
        3D cartesian vector coordinates.
    angle : float
        the angle to be rotated in radians (the default is 0).
    u : type
        unit vector to rotate around (the default is [1, 0 ,0]).

    Returns
    -------
    rotated_vec : 1-D array
        The 3D cartesian coordinates of the rotated vector

    """
    I = np.identity(3)  # noqa E741
    W = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])  # noqa E741

    R = I + np.sin(angle) * W + 2 * (np.sin(angle / 2) ** 2) * np.matmul(W, W)  # noqa E741
    return np.matmul(vec, R)

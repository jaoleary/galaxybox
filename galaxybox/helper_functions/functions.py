"""Various useful functions."""
import os
import subprocess
import numpy as np
import inspect
from astropy import cosmology as apcos
import astropy.units as apunits

__author__ = ('Joseph O\'Leary', 'Benjamin Moster', )


def clevels(z, level=0.68, bins=1000):
    """confidence levels"""
    y, x = np.histogram(z, bins=bins)
    x = 0.5 * (x[1:] + x[:-1])
    y = np.cumsum(y / np.sum(y))
    mean = np.argwhere(y > 0.5).reshape(-1)[0]
    low = np.argwhere(y > (1. - level) / 2.0).reshape(-1)[0]
    high = np.argwhere(y > 1. - (1. - level) / 2.0).reshape(-1)[0]

    i1 = mean
    i0 = np.max([0, i1 - 1])
    if (i1 > i0):
        m = (x[i1] - x[i0]) / (y[i1] - y[i0])
        b = x[i0] - m * y[i0]
    else:
        m = 0
        b = x[i0]
    xmean = m * 0.5 + b

    i1 = low
    i0 = np.max([0, i1 - 1])
    if (i1 > i0):
        m = (x[i1] - x[i0]) / (y[i1] - y[i0])
        b = x[i0] - m * y[i0]
    else:
        m = 0
        b = x[i0]
    xlow = m * (1. - level) / 2.0 + b

    i1 = high
    i0 = np.max([0, i1 - 1])
    if (i1 > i0):
        m = (x[i1] - x[i0]) / (y[i1] - y[i0])
        b = x[i0] - m * y[i0]
    else:
        m = 0
        b = x[i0]
    xhigh = m * (1. - (1. - level) / 2.0) + b

    return xmean, xlow, xhigh


def is_sorted(L):
    """
    Check if an array is ascending.

    Parameters
    ----------
    L : 1-D array or sequence
        Input array.

    Returns
    -------
    True/False : Boolean

    """
    return np.all(L[:-1] <= L[1:])


def make_time_bins(cosmology, bins=None, centered=False, db=0.01, db_type='da', max_redshift=8):
    """
    Make histogram bins with equivalent separation for scale factor, redshift and cosmic time.

    Parameters
    ----------
    cosmology : astropy cosmology.LambdaCDM object
        The cosmology set by our emerge parameter file.

    bins : 1-D array or sequence
        If a set of bins are already provided they will be converted directly.
        Otherwise db will be used to construct bins.

    centered : Boolean
        Specifies whether the bins provided are edges or centers.

    db : float
        Bin width

    db_type : string
        This specifies to which unit the bin width is applied. from there the other units will be computed

    max_redshift : float
        maximum redshift for bin construction.
        If explicit bins are provided this is ignored.

    Returns
    -------
    scalefactor_bins : 1-D array
        Bin edges in units of scale factor (a). Bins returned are always set to ascending scale factor

    redshift_bins : 1-D array
        Bin edges in units of redshift (z)

    time_bins  : 1-D array
        Bin edges in units of cosmic age [Gyr]

    """
    db_type = db_type.upper()
    # list of acceptable bin type strings
    snap_accept = ['SNAP', 'SNAPS', 'SNAPSHOT', 'SNAPSHOTS', 'SIMULATION', 'EMERGE']
    scale_accept = ['A', 'DA', 'SCALE', 'SCALES', 'SCALEFACTOR', 'SCALEFACTORS', 'SCALE FACTOR', 'SCALE FACTORS']
    time_accept = ['T', 'DT', 'TIME', 'AGE', 'GYR']
    redshift_accept = ['Z', 'DZ', 'REDSHIFT', 'REDSHIFTS']

    if any(key == db_type for key in snap_accept):
        # for type snap, bin must be provided.
        print('Making bins centered on snapshot redshifts')
        scalefactor = bins
        scalefactor.sort()
        temp = (scalefactor[1:] + scalefactor[:-1]) / 2

        scalefactor_bins = np.append(np.insert(temp, 0, scalefactor[0]), scalefactor[-1])

        redshift_bins = 1 / scalefactor_bins - 1
        time_bins = cosmology.age(redshift_bins).value
        return scalefactor_bins, redshift_bins, time_bins

    elif any(key == db_type for key in scale_accept):
        print('Making bins evenly spaced in scale factor')
        min_scale = cosmology.scale_factor(max_redshift)
        scalefactor_bins = np.append(np.arange(1, min_scale, -db), min_scale)
        redshift_bins = 1 / scalefactor_bins - 1
        time_bins = cosmology.age(redshift_bins).value

        return scalefactor_bins[::-1], redshift_bins[::-1], time_bins[::-1]

    elif any(key == db_type for key in time_accept):
        print('Making bins evenly spaced in time')
        # make time bins (Gyr)
        max_time = cosmology.lookback_time(max_redshift).value
        time_bins = np.append(np.arange(0, max_time, db), max_time)
        # make redshift bins
        redshift_bins = np.zeros(len(time_bins))
        for i, time in enumerate(time_bins[1:]):
            redshift_bins[i + 1] = apcos.z_at_value(cosmology.lookback_time, time * apunits.Gyr)
        # make scale factor bins
        scalefactor_bins = cosmology.scale_factor(redshift_bins)
        return scalefactor_bins[::-1], redshift_bins[::-1], cosmology.age(0).value - time_bins[::-1]

    elif any(key == db_type for key in redshift_accept):
        print('Making bins evenly spaced in redshift')
        redshift_bins = np.append(np.arange(0, max_redshift, db), max_redshift)
        scalefactor_bins = cosmology.scale_factor(redshift_bins)
        time_bins = cosmology.age(redshift_bins).value

        return scalefactor_bins[::-1], redshift_bins[::-1], time_bins[::-1]


def target_redshift_bin(redshift, cosmology, bins=None, db=0.75, db_type='dt'):
    """Short summary.

    Parameters
    ----------
    redshift : type
        Description of parameter `redshift`.
    cosmology : type
        Description of parameter `cosmology`.
    bins : type
        Description of parameter `bins`.
    db : type
        Description of parameter `db`.
    db_type : type
        Description of parameter `db_type`.

    Returns
    -------
    type
        Description of returned object.

    """
    db_type = db_type.upper()
    # list of acceptable bin type strings
    snap_accept = ['SNAP', 'SNAPS', 'SNAPSHOT', 'SNAPSHOTS', 'SIMULATION', 'EMERGE']
    scale_accept = ['A', 'DA', 'SCALE', 'SCALES', 'SCALEFACTOR', 'SCALEFACTORS', 'SCALE FACTOR', 'SCALE FACTORS']
    time_accept = ['T', 'DT', 'TIME', 'AGE', 'GYR']
    redshift_accept = ['Z', 'DZ', 'REDSHIFT', 'REDSHIFTS']

    if any(key == db_type for key in snap_accept):
        # for type snap, bin must be provided.
        if bins is None:
            raise Exception('Must provide an ascending array of snapshot scale factors when db_type = snap')

        scalefactor_bins, redshift_bins, cosmictime_bins = make_time_bins(cosmology=cosmology, bins=bins, db_type='snap')
        if redshift == 0:
            return scalefactor_bins[-2:], redshift_bins[-2:], cosmictime_bins[-2:]
        else:
            arg = np.searchsorted(scalefactor_bins, 1 / (redshift + 1))
        return scalefactor_bins[arg - 1:arg + 1], redshift_bins[arg - 1:arg + 1], cosmictime_bins[arg - 1:arg + 1]

    elif any(key == db_type for key in scale_accept):
        print('Making bin evenly spaced in scale factor')

        scale = 1 / (redshift + 1)
        if (scale + db / 2) > 1.0:
            db = (1.0 - scale) * 2
        if (scale - db / 2) < 0:
            db = scale * 2
        scalefactor_bins = np.array([scale - db / 2, scale + db / 2])
        redshift_bins = 1 / scalefactor_bins - 1
        time_bins = cosmology.age(redshift_bins).value

        return scalefactor_bins, redshift_bins, time_bins

    elif any(key == db_type for key in time_accept):
        print('Making bin evenly spaced in time')
        # -------- THIS NEEDS A BETTER SOLUTION! -------- #
        if redshift == 0:
            db = db
        elif (cosmology.age(redshift).value + db / 2) > cosmology.age(0).value:
            db = (cosmology.age(0).value - cosmology.age(redshift).value) * 2
        elif (cosmology.age(redshift).value - db / 2) < 0:
            db = cosmology.age(redshift).value * 2

        if redshift == 0:
            time_bins = np.array([cosmology.age(0).value - db, cosmology.age(0).value])
            redshift_bins = np.array([apcos.z_at_value(cosmology.age, time_bins[0] * apunits.Gyr), 0])
        else:
            time_bins = np.array([cosmology.age(redshift).value - db / 2, cosmology.age(redshift).value + db / 2])
            redshift_bins = np.zeros(len(time_bins))
            for i, time in enumerate(time_bins):
                if time == cosmology.age(0).value:
                    redshift_bins[i] = 0
                else:
                    redshift_bins[i] = apcos.z_at_value(cosmology.age, time * apunits.Gyr)
        # make scale factor bins
        scalefactor_bins = cosmology.scale_factor(redshift_bins)
        # return time_bins
        return scalefactor_bins, redshift_bins, time_bins

    elif any(key == db_type for key in redshift_accept):
        print('Making bin evenly spaced in redshift')
        if (redshift - db / 2) < 0:
            db = redshift * 2
        redshift_bins = np.array([redshift + db / 2, redshift - db / 2])
        scalefactor_bins = cosmology.scale_factor(redshift_bins)
        time_bins = cosmology.age(redshift_bins).value

        return scalefactor_bins, redshift_bins, time_bins

def cmd(args, path=None, encoding='utf-8', **kwargs):
    """Wrapper function for executing command line opertions with live output to notebook cells.

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



def arg_parser(func, drop=False, **kwargs):
    """From a dict of key word agruments, seperate those relevant to input function.

    Parameters
    ----------
    func : function
        The function for which kwargs should be extracted.
    drop : bool
        if True kwargs for func will be dropped from the orginal list (the default is False).
    **kwargs : dict
        keyword arguments.

    Returns
    -------
    func_kwargs : dict
        kwargs only associated with input function.
    kwargs : dict
        original or modifed dict of kwargs.

    """
    arg_names = inspect.getfullargspec(func)[0]
    func_kwargs = {}
    for i, k in enumerate(arg_names):
        if k in kwargs:
            func_kwargs[k] = kwargs[k]
            if drop:
                kwargs.pop(k)
    return func_kwargs, kwargs

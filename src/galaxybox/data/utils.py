"""Module containing general input/output functions."""

import h5py


def hdf5_to_dict(group: h5py.Group) -> dict:
    """Recursively unpack and hdf5 file into a dict.

    Parameters
    ----------
    group : HDF5 group object
        Description of parameter `group`.

    Returns
    -------
    data : dict
        dict containing group data (including subgroups)

    """
    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Dataset):
            data[key] = group[key][()]
        else:
            data[key] = hdf5_to_dict(group[key])
    return data

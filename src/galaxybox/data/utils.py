"""Module containing general input/output functions."""

import h5py


def hdf5_to_dict(group: h5py.Group) -> dict:
    """Recursively unpack an HDF5 file into a dictionary.

    This function takes an HDF5 group object and recursively unpacks it into a dictionary.
    It iterates over the keys in the group and checks if each key corresponds to a dataset or a
    subgroup. If it is a dataset, the corresponding value is extracted and stored in the dictionary.
    If it is a subgroup, the function calls itself recursively to unpack the subgroup.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group object to be unpacked.

    Returns
    -------
    dict
        A dictionary containing the data from the HDF5 group, including subgroups.

    """
    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Dataset):
            data[key] = group[key][()]
        else:
            data[key] = hdf5_to_dict(group[key])
    return data

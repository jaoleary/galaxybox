"""Module containing functions used to read emerge input/output files."""
import h5py
import os
import numpy as np
import pandas as pd
import struct
from .io import recursive_hdf5

__author__ = ("Joseph O'Leary",)


def parse_header(file_path):
    """
    Read the header of txt Emerge output and ruturn columns as a list of strings.

    Parameters
    ----------
    file_path : string
        Path of file to be read

    Returns
    -------
    col_names : list
        A list containing the column names.

    """
    col_names = pd.read_csv(
        file_path, header=None, nrows=1, sep="\s+", engine="python"
    ).values.tolist()[0]

    for i, key in enumerate(col_names):
        # check if what is between the parenthesis is col number of not
        paren = key[key.find("(") + 1 : key.find(")")]
        if np.char.isnumeric(paren):
            col_names[i] = key.split("(", 1)[0]
        for char in "?#":
            col_names[i] = col_names[i].replace(char, "")
    return col_names


def read_merger_list(file_path):
    """Read an Emerge galaxy merger list."""
    merger_list = pd.read_hdf(file_path)
    return merger_list


def read_tree(file_path, fields_out=None):
    """Read an Emerge galaxy merger tree.

    Parameters
    ----------
    file_path : string
        Path of file to be read

    Returns
    -------
    galaxy_tree : pandas.DataFrame
        A data frame containing a galaxy merger tree.

    """
    if file_path.endswith(".h5"):
        galaxy_tree = pd.read_hdf(file_path, key="MergerTree/Galaxy")
        if fields_out is None:
            return galaxy_tree
        else:
            # This is less than ideal, should really set the data columns in read_hdf.
            # That method seems to not work with this HDF5 data layout.
            return galaxy_tree[fields_out]
    else:
        col_names = parse_header(file_path)
        if fields_out is None:
            fields_out = col_names

        galaxy_tree = pd.read_csv(
            file_path,
            names=col_names,
            usecols=fields_out,
            header=None,
            comment="#",
            skiprows=1,
            sep="\s+",
        )
        return galaxy_tree


def read_statistics(file_path, universe_num=0):
    """Read an Emerge statistics.h5 file.

    Parameters
    ----------
    file_path : string
        Path of file to be read

    universe_num: int, or boolean, optional
        If int only the specified universe will be loaded, otherwise all are loaded.

    Returns
    -------
    statfile : HDF5 group
        An h5py hdf5 group object containing statistics for simulated universe.

    """
    if file_path.endswith(".h5"):
        statfile = h5py.File(file_path, "r")
        statfilekeys = [key for key in statfile.keys()]
        h5f = statfile[statfilekeys[universe_num]]
        stats = recursive_hdf5(h5f)
        statfile.close()
        return stats
    else:
        stats = {
            "CSFRD": [
                ("Redshift", float),
                ("Csfrd_observed", float),
                ("Sigma_observed", float),
            ],
            "FQ": [
                ("Stellar_mass", float),
                ("Fq_observed", float),
                ("Sigma_observed", float),
                ("Mean_ScaleFactor", float),
                ("Fq_model", float),
                ("Sigma_model", float),
            ],
            "SMF": [
                ("Stellar_mass", float),
                ("Phi_observed", float),
                ("Sigma_observed", float),
                ("Mean_ScaleFactor", float),
                ("Phi_model", float),
                ("Sigma_model", float),
            ],
            "SSFR": [
                ("Redshift", float),
                ("Ssfr_observed", float),
                ("Sigma_observed", float),
                ("Stellar_mass", float),
                ("Ssfr_model", float),
                ("Sigma_model", float),
            ],
        }

        # for now this is hardcoded....in the future will allow for keyword imports
        stat_keys = None
        if stat_keys is not None:
            stat_keys = np.atleast_1d(stat_keys)
            stat_bool = []
            for i, k in enumerate(stat_keys):
                if k in stats.keys():
                    stat_bool.append(True)
                else:
                    stat_bool.append(False)

            if not all(stat_bool):
                raise KeyError(
                    "{} are not valid statistics keys!".format(
                        stat_keys[[not i for i in stat_bool]]
                    )
                )
        else:
            stat_keys = list(stats.keys())

        obs = {}
        for k in stat_keys:
            obs[k] = {}
            obs[k]["Data"] = {}
            if k is "Clustering":
                filepath = file_path + "wpobs.{:d}.out".format(universe_num)
            else:
                filepath = file_path + k.lower() + "obs.{:d}.out".format(universe_num)
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    if line.startswith("#"):
                        line = line.replace("\n", "")
                        key = line.replace("# ", "")
                        data = []
                    elif line.strip():
                        line = np.fromstring(line, sep=" ")
                        data.append(line)
                    else:
                        obs[k]["Data"][key] = np.array(data, dtype=stats[k])
                    line = fp.readline()

            if k is "Clustering":
                filepath = file_path + "wpmod.{:d}.out".format(universe_num)
            else:
                filepath = file_path + k.lower() + "mod.{:d}.out".format(universe_num)
            obs[k]["Model"] = np.loadtxt(filepath)

        return obs


def read_chain(Seed, Run=0, Path="./", Mode="MCMC", sample_size=None, lnprobmin=None):
    """Read in an Emerge chain.txt.

    Parameters
    ----------
    Seed : int
        The seed value used to create the chain, defining the file name

    Run : int
        Index of the run used to define the file name (for the first run: Run==0)

    Path : string
        Path to the chain file

    Mode : string
        Which fitting algorithm was used to create the chain. Three options:
        1) 'MCMC' for affine invariant ensemble sampler
        2) 'HYBRID' for the hybrid code that combines MCMC and swarm optimisation
        3) 'PT' for parallel tempering

    sample_size : int
        The number of chain samples to retur. Typically on the last set of walkers are used.
        Default here is to return all walkers.

    lnprobmin : float
        Minimum lnprob for a walker to get returned

    Returns
    -------
    coldchain : 2-D numpy array
        The cold chain.

    """
    # Set the file name
    if Mode == "MCMC":
        file = Path + "mcmc{}.{:03d}.out".format(Seed, Run)
    elif Mode == "HYBRID":
        file = Path + "hybrid{}.{:03d}.out".format(Seed, Run)
    elif Mode == "PT":
        file = Path + "pt{}.{:03d}.out".format(Seed, Run)
    else:
        return

    # Check if file exists
    if os.path.isfile(file) is False:
        print("File " + file + " does not exist!")
        return

    # Load chain
    chain = np.loadtxt(file)

    # Process chain
    if Mode == "MCMC":
        coldchain = chain
    if Mode == "HYBRID":
        coldchain = chain
        # Use lnProb instead of Chi2
        coldchain[:, 0] = -coldchain[:, 0] / 2.0
    if Mode == "PT":
        # Select only cold walkers
        temp = chain[:, 12]
        coldchain = chain[temp == 1.0]

    # Select only sample
    if sample_size is not None:
        coldchain = coldchain[-sample_size:]

    # Select only walkers above a minimum lnprob
    if lnprobmin is not None:
        coldchain = coldchain[coldchain[:, 0] > lnprobmin]

    # Return (cold) Chain
    return coldchain


def read_parameter_file(file_path):
    """Read in an Emerge parameter file.

    This function reads in and parses a standard Emerge parameter file.
    Each paramter name is a key in a parameter dict along with the corresponding
    values.

    Parameters
    ----------
    file_path : string
        Location of the parameter file `file_path`.

    Returns
    -------
    params : dict
        A dicitonary containing contents of the parameter file.

    """
    # ? Not sure this function is still needed given the parameter file class....
    param_table = pd.read_csv(
        file_path,
        names=["key", "value"],
        header=None,
        comment="%",
        sep="\s+",
        engine="python",
    )
    params = {}
    for i, row in param_table.iterrows():
        params[row["key"]] = pd.to_numeric(row["value"], errors="ignore")
    values = np.array([i for i in params["OutputRedshifts"].split(",")])
    params["OutputRedshifts"] = values.astype(float)
    return params


def read_halo_trees(file_path, file_format="EMERGE", fields_out=None):
    """Read in halo merger trees.

    Parameters
    ----------
    filepath : type
        Description of parameter `filepath`.
    file_format : type
        Description of parameter `file_format` (the default is 'EMERGE').

    Returns
    -------
    type
        Description of returned object.

    """
    file_format = file_format.upper()
    if file_format == "EMERGE":

        def SKIP(fname):
            return fname.read(4)

        dt = np.dtype(
            [
                ("haloid", np.uintc),
                ("descid", np.uintc),
                ("upid", np.uintc),
                ("np", np.ushort),
                ("mmp", np.ushort),
                ("scale", np.single),
                ("mvir", np.single),
                ("rvir", np.single),
                ("concentration", np.single),
                ("lambda", np.single),
                ("X_pos", np.single),
                ("Y_pos", np.single),
                ("Z_pos", np.single),
                ("X_vel", np.single),
                ("Y_vel", np.single),
                ("Z_vel", np.single),
            ]
        )
        file = open(file_path, "rb")

        SKIP(file)
        Ntrees = struct.unpack("i", file.read(4))[0]
        SKIP(file)
        SKIP(file)
        Nhalos = struct.unpack("i" * Ntrees, file.read(4 * Ntrees))
        SKIP(file)
        SKIP(file)
        TreeID = struct.unpack("I" * Ntrees, file.read(4 * Ntrees))
        SKIP(file)
        SKIP(file)
        data = np.fromfile(file, dtype=dt)
        halo_trees = pd.DataFrame(data)
        file.close()

        halo_trees["Type"] = 0
        type_mask = halo_trees["upid"] > 0
        halo_trees.loc[type_mask, "Type"] = 1
        fnum = int(file_path.split(".")[-1])
        halo_trees["FileNumber"] = fnum
        return Ntrees, Nhalos, TreeID, halo_trees
    elif file_format == "ROCKSTAR":
        col_names = parse_header(file_path)
        if fields_out is None:
            fields_out = col_names
        return pd.read_csv(
            file_path,
            names=col_names,
            usecols=fields_out,
            header=0,
            comment="#",
            sep="\s+",
        )
    else:
        raise NotImplementedError(
            "Only Rockstar and Emerge halo tree formats are currently supported."
        )


def read_halo_forests(file_path, ID_format="EMERGE"):
    forests = np.loadtxt(file_path, dtype=int)
    forests = pd.DataFrame(forests, columns=["rootid", "forestid"])

    if ID_format == "EMERGE":
        forests += 1
    forests.set_index("rootid", inplace=True)
    return forests


def write_halo_trees(file_path, tree, Ntrees, Nhalos, TreeID, file_format="EMERGE"):
    file_format = file_format.upper()
    if file_format == "EMERGE":

        def SKIP(fname):
            return fname.write(struct.pack("i", 4))

        dt = np.dtype(
            [
                ("haloid", np.uintc),
                ("descid", np.uintc),
                ("upid", np.uintc),
                ("np", np.ushort),
                ("mmp", np.ushort),
                ("scale", np.single),
                ("mvir", np.single),
                ("rvir", np.single),
                ("concentration", np.single),
                ("lambda", np.single),
                ("X_pos", np.single),
                ("Y_pos", np.single),
                ("Z_pos", np.single),
                ("X_vel", np.single),
                ("Y_vel", np.single),
                ("Z_vel", np.single),
            ]
        )
        file = open(file_path, "wb")
        SKIP(file)
        np.array(Ntrees, dtype=np.intc).tofile(file)
        SKIP(file)
        SKIP(file)
        np.array(tuple(Nhalos), dtype=np.intc).tofile(file)
        SKIP(file)
        SKIP(file)
        np.array(tuple(TreeID), dtype="I").tofile(file)
        SKIP(file)
        SKIP(file)
        np.array(tree.to_records(), dtype=dt).tofile(file)
        SKIP(file)
        file.close()

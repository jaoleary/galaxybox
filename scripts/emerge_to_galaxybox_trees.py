"""Convert Emerge HDF5 files to Parquet format.

The script defines two functions:
- hdf5_to_parquet: This function takes an input directory and an output directory as arguments. It
  finds all HDF5 files in the input directory, converts each file to Parquet format, and saves the
  converted files in the output directory.
- main: This function calls hdf5_to_parquet with the arguments provided when the script is run from
  the command line.

Usage:
    python emerge_to_galaxybox_trees.py <input_dir> <output_dir>

Where:
    <input_dir> is the path to the directory containing the input HDF5 files.
    <output_dir> is the path to the directory where the output Parquet files will be saved.
"""

import argparse
import os
from glob import glob

import pandas as pd
from tqdm.auto import tqdm


def hdf5_to_parquet(input_dir: str, output_dir: str) -> None:
    """Convert HDF5 files in the input directory to Parquet format in the output directory.

    Parameters
    ----------
    input_dir : str
        The directory containing the input HDF5 files.
    output_dir : str
        The directory where the output Parquet files will be saved.

    """
    glob_path = f"{input_dir}/*.h5"
    input_files = sorted(glob(glob_path))
    for input_file in (pbar := tqdm(input_files)):
        pbar.set_description(input_file.split("/")[-1])
        output_file = input_file.replace(input_dir, output_dir)
        output_file = os.path.join(output_dir, input_file.split("/")[-1].replace(".h5", ".parquet"))
        df = pd.read_hdf(input_file, key="MergerTree/Galaxy")
        df.set_index("ID", inplace=True, drop=True)
        df.to_parquet(output_file, index=True)


def main(*args, **kwargs):  # noqa D103
    hdf5_to_parquet(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some directories.")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)

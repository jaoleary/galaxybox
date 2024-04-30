import pandas as pd
import numpy as np

class RockstarDataset:
    def __init__(self):
        pass

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

def read_rockstar(filepath, fields_out=None):
    col_names = parse_header(filepath)
    if fields_out is None:
        fields_out = col_names
    return pd.read_csv(
        filepath,
        names=col_names,
        usecols=fields_out,
        header=0,
        comment="#",
        sep="\s+",
    )




def rockstar_to_parquet():
    return
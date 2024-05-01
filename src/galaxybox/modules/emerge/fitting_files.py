"""Classes for handling Emerge outputs from fitting."""

import numpy as np
import pandas as pd
from IPython.display import Math, display


class EmergeFitFile:
    """A class for reading in and operating on Emerge fitting files.

    Fit files can be read in alone or within the context of the `Universe` class.

    Parameters
    ----------
    filepath : str
        Filepath to the emerge fitting files (plain text output)

    """

    def __init__(self, filepath: str):
        """Initialize the EmergeFitFile class.

        Parameters
        ----------
        filepath : str
            Filepath to the emerge fitting files (plain text output)

        """
        self.filepath = filepath
        data = []
        for fp in filepath:
            header = self._colnames(fp)
            header = header.strip("# ")
            header = header.strip("\n")
            header = header.split(" ")
            for i, key in enumerate(header):
                header[i] = key.split(")")[-1]
            data += [pd.read_csv(fp, comment="#", delimiter="\s+", engine="python", names=header)]
        self.data = pd.concat(data)

    def _colnames(self, filepath: str, skiprows: int = 1) -> str:
        """Grab column names line from fitting file header.

        Parameters
        ----------
        filepath : str
            Filepath to the fitting file
        skiprows : int, optional
            Number of rows to skip before reading the column names, by default 1

        Returns
        -------
        str
            The column names line from the fitting file header

        """
        with open(filepath) as fp:
            for i, line in enumerate(fp):
                if i == skiprows:
                    return line

    def best(
        self,
        data: pd.DataFrame,
        percentile: float = 68.0,
        ipython: bool = False,
        latex: bool = False,
    ):
        """Determine best fit parameter values given a table of mcmc parameters.

        Parameters
        ----------
        data : pandas.DataFrame
            A table of parameter values
        percentile : float, optional
            Percentile value for determining parameter range, by default 68.0
        ipython : bool, optional
            Set true to print fancy math when working in Jupyter Notebook, by default False
        latex : bool, optional
            Return a latex string of best fit parameter values and ranges, by default False

        Returns
        -------
        numpy.array or string
            Return array containing best fit parameter values and ranges (default behavior) or a
            latex formatted string if `latex` is `True`

        """
        txt = []
        bf = []
        for key in data.keys():
            if key not in self._reserved:
                mcmc = np.percentile(
                    data[key].values, [50 - percentile / 2, 50, 50 + percentile / 2]
                )
                q = np.diff(mcmc)
                s = "{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{+{2:.4f}}}"
                txt += [s.format(mcmc[1], q[0], q[1], self.latex_alias(key))]
                bf += [[mcmc[1], q[0], q[1]]]
        if ipython:
            display(Math(",\;".join(txt)))
        if latex:
            return txt
        else:
            return np.array(bf)

    def latex_alias(self, key: str) -> str:
        """Return a latex formatted string for model parameters.

        Parameters
        ----------
        key : str
            Shorthand model parameter name used in fitting files.

        Returns
        -------
        str
            Latex string for use in displays

        """
        col_alias = {}
        # add other aliases.
        col_alias["M0"] = "M_{0}"
        col_alias["MZ"] = "M_{z}"
        col_alias["E0"] = "\\epsilon_{0}"
        col_alias["EZ"] = "\\epsilon_{z}"
        col_alias["EM"] = "\\epsilon_{m}"
        col_alias["B0"] = "\\beta_{0}"
        col_alias["BZ"] = "\\beta_{z}"
        col_alias["G0"] = "\\gamma_{0}"
        col_alias["GZ"] = "\\gamma_{z}"
        col_alias["Fesc"] = "f_{\\mathrm{esc}}"
        col_alias["Fstrip"] = "f_{\\mathrm{s}}"
        col_alias["Tau0"] = "\\tau_{0}"
        col_alias["TauS"] = "\\tau_{s}"
        col_alias["TauD"] = "\\tau_{d}"
        col_alias["IonZ"] = "z_{\\mathrm{ion}}"
        col_alias["IonM"] = "M_{\\mathrm{ion}}"
        col_alias["IonR"] = "R_{\\mathrm{ion}}"
        col_alias["MRI0"] = "M_{\\mathrm{RI}}"
        col_alias["MRIZ"] = "M_{\\mathrm{RI} z}"
        col_alias["A0"] = "\\alpha_{0}"
        col_alias["AZ"] = "\\alpha_{z}"

        if key in col_alias.keys():
            return col_alias[key]
        else:
            return key

    def free_params(self, latex: bool = False) -> list[str]:
        """Show all emerge model free parameters.

        Parameters
        ----------
        latex : bool, optional
            Return a latex string of best fit parameter values and ranges, by default False

        Returns
        -------
        list[str]
            List of model parameters

        """
        fp = []
        for key in self.data.keys():
            if key not in self._reserved:
                if latex:
                    fp += [self.latex_alias(key)]
                else:
                    fp += [key]
        return fp


class EmergeMCMC(EmergeFitFile):
    """A class for handling Emerge MCMC fitting files."""

    def __init__(self, filepath: str):
        """Initialize the EmergeMCMC class.

        Parameters
        ----------
        filepath : str
            Filepath to the emerge MCMC fitting files (plain text output)

        """
        super().__init__(filepath)

    def best(self, **kwargs):
        """Determine best fit parameter values given a table of MCMC parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `best` method of the parent class.

        Returns
        -------
        numpy.array or string
            Return array containing best fit parameter values and ranges (default behavior) or a
            latex formatted string if `latex` is `True`.

        """
        return super().best(self.data, **kwargs)


class EmergeHybrid(EmergeFitFile):
    """A class for handling Emerge Hybrid fitting files."""

    def __init__(self, filepath: str):
        """Initialize the EmergeHybrid class.

        Parameters
        ----------
        filepath : str
            Filepath to the emerge Hybrid fitting files (plain text output)

        """
        super().__init__(filepath)

    def best(self, **kwargs):
        """Determine best fit parameter values given a table of Hybrid parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `best` method of the parent class.

        Returns
        -------
        numpy.array or string
            Return array containing best fit parameter values and ranges (default behavior) or a
            latex formatted string if `latex` is `True`.

        """
        return super().best(self.data, **kwargs)


class EmergeParallelTempering(EmergeFitFile):
    """A class for handling Emerge Parallel Tempering fitting files."""

    def __init__(self, filepath: str):
        """Initialize the EmergeParallelTempering class.

        Parameters
        ----------
        filepath : str
            Filepath to the emerge Parallel Tempering fitting files (plain text output)

        """
        super().__init__(filepath)

    def best(self, **kwargs):
        """Determine best fit parameter values given a table of Parallel Tempering parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `best` method of the parent class.

        Returns
        -------
        numpy.array or string
            Return array containing best fit parameter values and ranges (default behavior) or a
            latex formatted string if `latex` is `True`.

        """
        return super().best(self.data.loc[self.data.Temp == 1], **kwargs)

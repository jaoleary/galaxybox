import numpy as np
import pandas as pd
from IPython.display import display, Math

__author__ = ("Joseph O'Leary",)


class fit:
    """A class for reading in and operating on Emerge fitting files
    alone or within the context of a Universe.
    """

    # all column names that ARENT model parameters
    _reserved = ["lnprob", "Temp", "Scale", "Frac_accept", "Step", "Epoch", "iwalker"]

    def __init__(self, filepath):
        self.filepath = filepath
        data = []
        for fp in filepath:
            header = self._colnames(fp)
            header = header.strip("# ")
            header = header.strip("\n")
            header = header.split(" ")
            for i, key in enumerate(header):
                header[i] = key.split(")")[-1]
            data += [
                pd.read_csv(
                    fp, comment="#", delimiter="\s+", engine="python", names=header
                )
            ]
        self.data = pd.concat(data)

    def _colnames(self, filepath, skiprows=1):
        """Grab column names line from fitting file header"""
        with open(filepath) as fp:
            for i, line in enumerate(fp):
                if i == skiprows:
                    return line

    def _best(self, data, percentile=68.0, ipython=False, latex=False):
        """Determine best fit parameter values given a table of mcmc parameters.

        Parameters
        ----------
        data : pandas.DataFrame
            A table of parameter values
        percentile : float, optional
            Percentile value for determining parameter range, by default 68.0
        ipython : bool, optional
            Set true to print fancy math when working in Jupyter Notebook, by default False
        return_latex : bool, optional
            Return a latex string of best fit parameter values and ranges, by default False

        Returns
        -------
        numpy.array
            Return array containting best fit parameter values and ranges (default behavior)
        string
            Return latex formated parameter strings if `return_latex` is `True`
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

    def latex_alias(self, key):
        """Return a latex formated string for model parameters

        Parameters
        ----------
        key : string
            Shorthand model parameter name used in fitting files.

        Returns
        -------
        string
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

    def free_params(self, latex=False):
        fp = []
        for key in self.data.keys():
            if key not in self._reserved:
                if latex:
                    fp += [self.latex_alias(key)]
                else:
                    fp += [key]
        return fp


class mcmc(fit):
    def __init__(self, filepath):
        super().__init__(filepath)

    def best(self, **kwargs):
        """See `_best` in parent class"""
        return super()._best(self.data, **kwargs)


class hybrid(fit):
    def __init__(self, filepath):
        super().__init__(filepath)

    def best(self, **kwargs):
        """See `_best` in parent class"""
        return super()._best(self.data, **kwargs)


class parallel_tempering(fit):
    def __init__(self, filepath):
        super().__init__(filepath)

    def best(self, **kwargs):
        """See `_best` in parent class"""
        # we only care about cold walkers for parameter estimation.
        return super()._best(self.data.loc[self.data.Temp == 1], **kwargs)

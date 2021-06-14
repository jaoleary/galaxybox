import numpy as np
import pandas as pd
from IPython.display import display, Math

__author__ = ('Joseph O\'Leary', )

class fit:
    """A class for reading in and operating on Emerge fitting files
    alone or within the context of a Universe.
    """
    # all column names that ARENT model parameters
    _reserved = ['lnprob', 'Temp', 'Scale', 'Frac_accept', 'Step', 'Epoch', 'iwalker']
    def __init__(self, filepath):
        self.filepath = filepath
        data = []
        for fp in filepath:
            header = self._colnames(fp)
            header = header.strip('# ')
            header = header.strip('\n')
            header = header.split()
            for i, key in enumerate(header):
                header[i] = key.split(')')[-1]
            data += [pd.read_csv(fp, comment='#', delimiter = '\s+', engine = 'python', names=header)]
        self.data = pd.concat(data)       

    def _colnames(self, filepath, skiprows=1):
        """Grab column names line from fitting file header
        """        
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
        txt=[]
        bf =[]
        for key in data.keys():
            if key not in self._reserved:
                mcmc = np.percentile(data[key].values, [50-percentile/2, 50, 50+percentile/2])
                q = np.diff(mcmc)
                s = '{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{+{2:.4f}}}'
                txt += [s.format(mcmc[1], q[0], q[1], self.alias(key))]
                bf += [[mcmc[1], q[0], q[1]]]
        if ipython:
            display(Math(',\;'.join(txt)))
        if latex:
            return txt
        else:
            return np.array(bf)
        
    def alias(self, key, form='latex'):
        """Return a latex or parameter file formated string for model parameters

        Parameters
        ----------
        key : string
            Shorthand model parameter name used in fitting files.
        form : string
            Set to `param` to return the parameter file value, by default `latex`

        Returns
        -------
        string
            Latex string for use in displays
        """

        # return a latex alias or the parameter file key name
        idx = {'latex':0,
               'param':1}

        col_alias = {}
        # set aliases.
        col_alias['M0'] = ['M_{0}', 'Eff_MassPeak']
        col_alias['MZ'] = ['M_{z}', 'Eff_MassPeak_Z']
        col_alias['E0'] = ['\\epsilon_{0}', 'Eff_Normalisation']
        col_alias['EZ'] = ['\\epsilon_{z}', 'Eff_Normalisation_Z']
        col_alias['EM'] = ['\\epsilon_{m}', 'Eff_Normalisation_M']
        col_alias['B0'] = ['\\beta_{0}', 'Eff_LowMassSlope']
        col_alias['BZ'] = ['\\beta_{z}', 'Eff_LowMassSlope_Z']
        col_alias['G0'] = ['\\gamma_{0}', 'Eff_HighMassSlope']
        col_alias['GZ'] = ['\\gamma_{z}', 'Eff_HighMassSlope_Z']
        col_alias['Fesc'] = ['f_{\\mathrm{esc}}', 'Fraction_Escape_ICM']
        col_alias['Fstrip'] = ['f_{\\mathrm{s}}', 'Fraction_Stripping']
        col_alias['Tau0'] = ['\\tau_{0}', 'Timescale_Quenching']
        col_alias['TauS'] = ['\\tau_{s}', 'Slope_Quenching']
        col_alias['TauD'] = ['\\tau_{d}', 'Decay_Quenching']
        col_alias['IonA'] = ['a_{\\mathrm{ion}}', 'Reionization_Scale']
        col_alias['IonM'] = ['M_{\\mathrm{ion}}', 'Reionization_Mass']
        col_alias['IonR'] = ['R_{\\mathrm{ion}}', 'Reionization_Rate']
        col_alias['MRI0'] = ['M_{\\mathrm{RI}}', 'Eff_ReionizationMass']
        col_alias['MRIZ'] = ['M_{\\mathrm{RI} z}', 'Eff_ReionizationMass_Z']
        col_alias['A0']   = ['\\alpha_{0}', 'Eff_ReionizationSlope']
        col_alias['AZ']   = ['\\alpha_{z}', 'Eff_ReionizationSlope_Z']

        if key in col_alias.keys():
            return col_alias[key][idx[form]]
        else:
            return key

    def free_params(self, latex=False):
        fp = []
        for key in self.data.keys():
            if key not in self._reserved:
                if latex:
                    fp += [self.alias(key)]
                else:
                    fp += [key]
        return fp

    def update_params(self):
        """Update the free parameters in the parameter file with best fit values"""
        if hasattr(self, '_params'):
            b = np.around(self.best(), decimals=5)
            for i, p in enumerate(self.free_params()):
                self._params.update(option=self.alias(p, form='param'), value=b[i,0])

class mcmc(fit):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def best(self,**kwargs):
        """See `_best` in parent class
        """    
        return super()._best(self.data, **kwargs)
        
class hybrid(fit):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def best(self,**kwargs):
        """See `_best` in parent class
        """        
        return super()._best(self.data, **kwargs)

class parallel_tempering(fit):
    def __init__(self, filepath):
        super().__init__(filepath) 

    def best(self,**kwargs):
        """See `_best` in parent class
        """        
        # we only care about cold walkers for parameter estimation.
        return super()._best(self.data.loc[self.data.Temp==1], **kwargs)

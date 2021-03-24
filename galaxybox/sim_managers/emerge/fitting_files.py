import numpy as np
import pandas as pd
from IPython.display import display, Math

__author__ = ('Joseph O\'Leary', )

class fit:
    # all column names that ARENT model parameters
    _reserved = ['lnprob', 'Temp', 'Scale', 'Frac_accept', 'Step', 'Epoch', 'iwalker']
    def __init__(self, filepath):
        self.filepath = filepath
        data = []
        for fp in filepath:
            header = self._colnames(fp)
            header = header.strip('# ')
            header = header.strip('\n')
            header = header.split(' ')
            for i, key in enumerate(header):
                header[i] = key.split(')')[-1]
            data += [pd.read_csv(fp, comment='#', delimiter = '\s+', engine = 'python', names=header)]
        self.data = pd.concat(data)       

    def _colnames(self, filepath, skiprows=1):
        with open(filepath) as fp:
            for i, line in enumerate(fp):
                if i == skiprows:
                    return line
    
    def _best(self, data, percentile=68, ipython=False, return_latex=False):
        txt=''
        bf =[]
        for key in data.keys():
            if key not in self._reserved:
                mcmc = np.percentile(data[key].values, [50-percentile/2, 50, 50+percentile/2])
                q = np.diff(mcmc)
                s = "{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{+{2:.4f}}},\;"
                txt += s.format(mcmc[1], q[0], q[1], self.latex_alias(key))
                bf += [[mcmc[1], q[0], q[1]]]
        if ipython:
            display(Math(txt))
        if return_latex:
            return txt
        else:
            return np.array(bf)
        
    def latex_alias(self, key):
        col_alias = {}
        # add other aliases.
        col_alias['M0'] = 'M_{0}'
        col_alias['MZ'] = 'M_{z}'
        col_alias['E0'] = '\\epsilon_{0}'
        col_alias['EZ'] = '\\epsilon_{z}'
        col_alias['EM'] = '\\epsilon_{m}'
        col_alias['B0'] = '\\beta_{0}'
        col_alias['BZ'] = '\\beta_{z}'
        col_alias['G0'] = '\\gamma_{0}'
        col_alias['GZ'] = '\\gamma_{z}'
        col_alias['Fesc'] = 'f_{\\mathrm{esc}}'
        col_alias['Fstrip'] = 'f_{\\mathrm{s}}'
        col_alias['Tau0'] = '\\tau_{0}'
        col_alias['TauS'] = '\\tau_{s}'
        col_alias['TauD'] = '\\tau_{d}'
        col_alias['IonZ'] = 'z_{\\mathrm{ion}}'
        col_alias['IonM'] = 'M_{\\mathrm{ion}}'
        col_alias['IonR'] = 'R_{\\mathrm{ion}}'
        col_alias['MRI0'] = 'M_{\\mathrm{RI}}'
        col_alias['MRIZ'] = 'M_{\\mathrm{RI} z}'
        col_alias['A0']   = '\\alpha_{0}'
        col_alias['AZ']   = '\\alpha_{z}'

        if key in col_alias.keys():
            return col_alias[key]
        else:
            return key

class mcmc(fit):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def best(self,**kwargs):
        return super()._best(self.data, **kwargs)
        
class hybrid(fit):
    def __init__(self, filepath):
        super().__init__(filepath)

class parallel_tempering(fit):
    def __init__(self, filepath):
        super().__init__(filepath) 

    def best(self,**kwargs):
        # we only care about cold walkers for parameter estimation.
        return super()._best(self.data.loc[self.data.Temp==1], **kwargs)

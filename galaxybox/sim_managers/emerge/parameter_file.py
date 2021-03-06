"""Classes for handling Emerge data."""
import numpy as np
import copy
from astropy import cosmology as apcos

__author__ = ('Joseph O\'Leary', )


class params:
    """Load emerege parameterfile."""

    def __init__(self, filepath):
        """Initialize the configuration."""
        self.__val_indent = 29
        self.__desc_indent = 49
        self.load_params(filepath)
        self.__bkp = copy.deepcopy(self.__blocks)
    
        self.cosmology = apcos.LambdaCDM(H0=self.get_param('HubbleParam') * 100,
                                        Om0=self.get_param('Omega0'),
                                        Ode0=self.get_param('OmegaLambda'),
                                        Ob0=self.get_param('OmegaBaryon')) 
    
    def __repr__(self):
        """Report current parameter file setup."""
        str = ''
        str += self.header()
        for blk in self.__blocks:
            str += self.blkheader(name=blk)  + '\n'
            for opt in self.__blocks[blk]:
                str += self.optwrite(block=blk, option=opt)  + '\n'
        return str

    def load_params(self, filepath):
        """Import emerge `.param` file.

        Parameters
        ----------
        filepath : string
            File path to `.param` file

        """
        self.__blocks = {}
        with open(filepath) as fp:
            for line in fp:
                line = line.strip('\n')
                if (line.startswith('%') and line.endswith('%')) or (not line.strip()):
                    continue
                if line.startswith('% '):
                    blkkey = line.strip('%').strip()
                    self.__blocks[blkkey] = {}
                else:
                    if line.startswith('%'):
                        enable = False
                        line = line[1:]
                    else:
                        enable = True

                    line = line.split('%')
                    if len(line) > 1:
                        description = line[1][1:]
                        del line[-1]
                    line = line[0].split()
                    name = line[0]
                    value = line[1]
                    if (value[0].isdigit()) and (name is not 'ModelName'):
                        # this must be numeric
                        if value.isdigit():
                            # if its a digit it must be int
                            value = np.int(value)
                        else:
                            value = np.fromstring(value, sep=',')
                            if len(value) == 1:
                                value = value[0]

                    self.__blocks[blkkey][name] = {}
                    self.__blocks[blkkey][name]['enable'] = enable
                    self.__blocks[blkkey][name]['value'] = value
                    self.__blocks[blkkey][name]['description'] = description

    def write(self, file_path):
        """Write the current parameter configuration to a file.

        Parameters
        ----------
        file_path : string
            The path of the ouput parameter file.

        """
        fp = open(file_path, 'w')
        fp.write(self.header())
        for blk in self.__blocks:
            fp.write('\n' + self.blkheader(name=blk))
            for opt in self.__blocks[blk]:
                fp.write('\n' + self.optwrite(block=blk, option=opt))
        fp.write('\n')
        fp.close()

    def reset(self):
        """Reset config class to initial loaded state."""
        del self.__blocks
        self.__blocks = copy.deepcopy(self.__bkp)

    def blank(self, length=1):
        """Create a blank space of variable lenth.

        Parameters
        ----------
        length : int
            The size of the blank space.

        Returns
        -------
        string
            a blank space of size `length`.

        """
        return ' ' * length

    def cmtln(self, length=1, char='%'):
        """Create a comment line of variable length.

        Parameters
        ----------
        length : int
            Description of parameter `length` (the default is 1).
        char : string
            A the character to use for comments (the default is '#').

        Returns
        -------
        string
            A string repeating `char` of size `length`.

        """
        return char * length

    def blkheader(self, name='', mode=None):
        """Create a formated section header for a parameter file.

        Parameters
        ----------
        name : string
            The name of the section header (the default is '').

        Returns
        -------
        headerstr : string
            A formated section header string.

        """
        headerstr = '\n' * 2
        headerstr += '% ' + name + '\n'
        return headerstr

    def header(self):
        """Create a formated header for printing to a parameter file.

        Returns
        -------
        headerstr : string
            A formated header string.

        """
        headerstr = self.cmtln(124)
        headerstr += '\n%' + self.blank(122) + '%'
        headerstr += '\n% Parameter file for the EMERGE code' + self.blank(87) + '%'
        headerstr += '\n%' + self.blank(122) + '%\n'
        headerstr += self.cmtln(124)
        return headerstr

    def optwrite(self, block, option):
        """Create a formated string for printing a parameter option to file.

        Parameters
        ----------
        block : type
            The section of the parameter file that the option will be added to.
        option : type
            parameter file option name.

        Returns
        -------
        string
            A formated string for printing a parameter option to single file line

        """
        if self.__blocks[block][option]['enable']:
            enable = ''
        else:
            enable = '%'
        value = self.__blocks[block][option]['value']
        if self.__blocks[block][option]['value'] is None:
            optstr = ''
        else:
            if isinstance(value, str):
                optstr = value
            else:
                if block == 'Units':
                    tmp = max(np.abs(value), 1 / np.abs(value))
                    if np.abs(tmp) >= 1000:
                        string = '{:e}'.format(value)
                        front = string.split('e')[0]

                        if np.abs(np.float(front)) == 1:
                            front = '{:.1f}'.format(np.float(front))
                        else:
                            front = front.strip('0')
                        front = front.strip()

                        back = string.split('e')[1]
                        back = '{:d}'.format(np.int(back))
                        if np.int(back) == 0:
                            optstr = front
                        else:
                            optstr = 'e'.join([front, back])
                    else:
                        if value % 1 == 0:
                            optstr = '{:.1f}'.format(value)
                        else:
                            optstr = '{:f}'.format(value)
                            optstr = optstr.strip('0')
                else:
                    value = np.atleast_1d(value)
                    optstr = ','.join(map(str, list(value)))

        vspace = self.blank(self.__val_indent - len(enable + option))
        b = self.__desc_indent - len(enable + option + vspace + optstr)
        if b <= 0:
            b = 1
        dspace = self.blank(b)
        description = '% ' + self.__blocks[block][option]['description']

        return enable + option + vspace + optstr + dspace + description

    def optadd(self, block, option, enable=False, value=None, description=' '):
        """Add a new parameter.

        Parameters
        ----------
        block : type
            The section of the configuration file that the option will be added to.
        option : type
            Configuration file option name.
        enable : string
            Whether an option should be enabled in config file, or commented out.(the default is False).
        value : int, float
            If the compile option accepts an assignable value set it (the default is None).
        description : string
            A description of the of the compile option. (the default is ' ').

        """
        if block not in self.__blocks.keys():
            self.__blocks[block] = {}

        for k in self.__blocks:
            if option in self.__blocks[k].keys():
                raise KeyError('Option ' + option + ' already exists. Use `.optupdate()` method.')

        self.__blocks[block][option] = {}
        self.__blocks[block][option]['enable'] = enable
        self.__blocks[block][option]['value'] = value
        self.__blocks[block][option]['description'] = description

    def optupdate(self, option, force=False, **kwargs):
        """Update a parameter file option.

        Parameters
        ----------
        option : string
            Configuration file option name
        enable : bool, optional
            Whether an option should be enabled in config file, or commented out.
        value : int, float, optional
            If the compile option accepts an assignable value set it, otherwise set to None.
        description : string
            A description of the of the compile option.

        """
        for k in self.__blocks:
            if option in self.__blocks[k].keys():
                if (option == 'ModelName') and (not force):
                    raise PermissionError('Changing the model name is generally a bad idea...set `force=True` to override')
                for v in kwargs:
                    if v in self.__blocks[k][option].keys():
                        self.__blocks[k][option][v] = kwargs[v]
                    else:
                        raise KeyError('`{}`'.format(v) + ' is not a valid option key')

    def get_param(self, option):
        for b in self.__blocks:
            if option in self.__blocks[b].keys():
                return self.__blocks[b][option]['value']

        
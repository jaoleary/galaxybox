"""Classes for handling Emerge data."""

import copy

import numpy as np

# TODO: replace the functionality of this module with something more sensible


class EmergeConfig:
    """Import/store/update/write emerge configuration files."""

    def __init__(self, filepath):
        """Initialize the configuration."""
        self._indent = 29
        self.load_config(filepath)
        self._bkp = copy.deepcopy(self._blocks)

    def __repr__(self):
        """Report current configuration file setup."""
        str = ""
        str += self.header()
        for blk in self._blocks:
            str += self.blkheader(name=blk) + "\n"
            for opt in self._blocks[blk]:
                str += self.optwrite(block=blk, option=opt) + "\n"
        str += "#\n" + self.cmtln(100)
        return str

    def write(self, file_path):
        """Write the current configuration to a file.

        Parameters
        ----------
        file_path : string
            The path of the ouput configuration file.

        """
        fp = open(file_path, "w")
        fp.write(self.header())
        for blk in self._blocks:
            fp.write("\n" + self.blkheader(name=blk))
            for opt in self._blocks[blk]:
                fp.write("\n" + self.optwrite(block=blk, option=opt))
        fp.write("\n#\n" + self.cmtln(100))
        fp.close()

    def write_compiled(self, file_path):
        """Write the current state to a file in the emerge `compile_options.txt` format.

        Parameters
        ----------
        file_path : string
            Output file path.

        """
        fp = open(file_path, "w")
        fp.write("#compiled config created but `galaxybox`.\n")
        for blk in self._blocks:
            for opt in self._blocks[blk]:
                if self._blocks[blk][opt]["enable"]:
                    if self._blocks[blk][opt]["value"] is not None:
                        line = opt + "={}".format(self._blocks[blk][opt]["value"])
                    else:
                        line = opt
                    fp.write(line + "\n")
        fp.close()

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
        return " " * length

    def cmtln(self, length=1, char="#"):
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

    def blkheader(self, name=""):
        """Create a formated section header for a configuration file.

        Parameters
        ----------
        name : string
            The name of the section header (the default is '').


        Returns
        -------
        headerstr : string
            A formated section header string.

        """
        headerstr = "#\n"
        headerstr += "# " + self.cmtln(length=98, char="-") + "\n"
        headerstr += "#" + self.blank(self._indent - 1) + name + "\n"
        headerstr += "# " + self.cmtln(length=98, char="-")
        return headerstr

    def header(self):
        """Create a formated header for printing to a configuration file.

        Returns
        -------
        headerstr : string
            A formated header string.

        """
        headerstr = self.cmtln(100)
        headerstr += "\n#!/bin/bash"
        headerstr += self.blank(12)
        headerstr += "# this line only there to enable syntax highlighting in this file\n"
        headerstr += self.cmtln(100) + "\n#" + self.blank(2)
        headerstr += "EMERGE Config file - Enable/Disable compile-time options as needed"
        headerstr += self.blank(30) + "#\n" + self.cmtln(100) + "\n#"
        return headerstr

    def load_config(self, filepath):
        """Import an emerge config file and save the dictionary to an attribute.

        Parameters
        ----------
        filepath : string
            Path to the emerge config file that should be loaded.

        """
        self._blocks = {}
        with open(filepath) as fp:
            line = fp.readline().strip("\n")

            while line:
                if line.startswith("# -"):
                    line = fp.readline().strip("\n")
                    blkkey = line.strip("#").strip()
                    self._blocks[blkkey] = {}
                    line = fp.readline().strip("\n")
                if not (line.startswith("# ") or line.startswith("#!") or len(set(line)) == 1):
                    name = line.split("#")
                    if name[0] == "":
                        name = name[1].strip()
                        enable = False
                    else:
                        name = name[0].strip()
                        enable = True
                    name = name.split("=")
                    self._blocks[blkkey][name[0]] = {}
                    if len(name) == 1:
                        value = None
                    else:
                        try:
                            value = np.int(name[1])
                        except:  # noqa E722
                            value = np.float(name[1])

                    self._blocks[blkkey][name[0]]["value"] = value
                    self._blocks[blkkey][name[0]]["enable"] = enable
                    self._blocks[blkkey][name[0]]["description"] = line.split("#")[-1].strip()

                line = fp.readline().strip("\n")

    def load_compiled_config(self, filepath):
        """Import emerge `compiled_config.txt`.

        Parameters
        ----------
        filepath : string
            File path to `compiled_config.txt`

        """
        if hasattr(self, "_cmpl_opt"):
            del self._cmpl_opt
        self._compiled_path = filepath

        fp = open(filepath)
        line = fp.readline().strip("\n")
        while line.startswith("#"):
            if not line.startswith("#compiled"):
                self.build = {}  # read in the current build info version, branch, git-hash
                for b in line.strip("#").split(" - "):
                    key, value = b.split(": ")
                    self.build[key] = value
            line = fp.readline().strip("\n")
        self._cmpl_opt = {}
        while line:
            line = line.split("=")
            opt = line[0]
            self._cmpl_opt[opt] = {}
            self._cmpl_opt[opt]["enable"] = True

            if len(line) == 1:
                self._cmpl_opt[opt]["value"] = None
            else:
                try:
                    self._cmpl_opt[opt]["value"] = np.int(line[1])
                except:  # noqa E722
                    self._cmpl_opt[opt]["value"] = np.float(line[1])
                self._cmpl_opt[opt]["value"]
            line = fp.readline().strip("\n")

    def update_from_compiled_config(self, filepath=None):
        """Update current configuration using compiled_config.txt.

        Parameters
        ----------
        filepath : string, optional
            File path to `compiled_config.txt`, default None

        """
        if filepath is not None:
            self.load_compiled_config(filepath)
        self.set_all(enable=False)
        for opt in self._cmpl_opt:
            enable = self._cmpl_opt[opt]["enable"]
            value = self._cmpl_opt[opt]["value"]
            try:
                self.optupdate(option=opt, enable=enable, value=value)
            except:  # noqa E722
                self.optadd(block="OTHER OPTIONS", option=opt, enable=enable, value=value)

    def reset(self):
        """Reset config class to initial loaded state."""
        del self._blocks
        self._blocks = copy.deepcopy(self._bkp)

    @classmethod
    def from_compiled(cls, template_path, compiled_path):
        """Initialize config class set from compiled config.

        Parameters
        ----------
        template_path : string
            File path the a configuration file to use as a format template.
        compiled_path : string
            File path to the `compiled_config.txt` to be initiated from.

        Returns
        -------
        config object
            Emerge configuration file object

        """
        cfg = cls(template_path)
        cfg.update_from_compiled_config(compiled_path)
        setattr(cfg, "_template_path", template_path)
        setattr(cfg, "_compiled_path", compiled_path)
        return cfg

    def optwrite(self, block, option):
        """Create a formated string for printing a compile option to file.

        Parameters
        ----------
        block : type
            The section of the configuration file that the option will be added to.
        option : type
            Configuration file option name.

        Returns
        -------
        string
            A formated string for printing a compile option to single file line

        """
        if self._blocks[block][option]["enable"]:
            enable = ""
        else:
            enable = "#"
        if self._blocks[block][option]["value"] is None:
            optstr = option
        else:
            optstr = option + "={}".format(self._blocks[block][option]["value"])

        space = self.blank(self._indent - len(enable + optstr))
        description = "# " + self._blocks[block][option]["description"]

        return enable + optstr + space + description

    def optadd(self, block, option, enable=False, value=None, description=" "):
        """Add a compile option to the configuration.

        Parameters
        ----------
        block : type
            The section of the configuration file that the option will be added to.
        option : type
            Configuration file option name.
        enable : string
            Whether an option should be enabled in config file, or commented out.
            (the default is False).
        value : int, float
            If the compile option accepts an assignable value set it (the default is None).
        description : string
            A description of the of the compile option. (the default is ' ').

        """
        if block not in self._blocks.keys():
            self._blocks[block] = {}

        for k in self._blocks:
            if option in self._blocks[k].keys():
                raise KeyError("Option " + option + " already exists. Use `.optupdate()` method.")

        self._blocks[block][option] = {}
        self._blocks[block][option]["enable"] = enable
        self._blocks[block][option]["value"] = value
        self._blocks[block][option]["description"] = description

    def optupdate(self, option, **kwargs):
        """Update a configuration file option.

        Parameters
        ----------
        option : str
            Configuration file option name
        **kwargs : dict
            A dictionary of key value pairs to update the option with.

        """
        for k in self._blocks:
            if option in self._blocks[k].keys():
                for v in kwargs:
                    if v in self._blocks[k][option].keys():
                        self._blocks[k][option][v] = kwargs[v]
                    else:
                        raise KeyError("`{}`".format(v) + " is not a valid option key")

    def set_all(self, block=None, enable=True):
        """Enable or disable all config options in block or file.

        Parameters
        ----------
        block : string, optional
            The option block for the configuration (the default is None).
        enable : bool, optional
            Whether to enable or disable all options (the default is True).

        """
        if block is not None:
            block = np.atleast_1d(block)
            for blk in block:
                for opt in self._blocks[blk]:
                    self._blocks[blk][opt]["enable"] = enable
        else:
            for blk in self._blocks:
                for opt in self._blocks[blk]:
                    self._blocks[blk][opt]["enable"] = enable

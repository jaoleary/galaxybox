"""Classes for handling Emerge data."""
import numpy as np
import os
import copy
from astropy import cosmology as apcos
from . import emerge as em
from ..helper_functions.functions import *
from ..io import emerge_io as em_io
from matplotlib.pyplot import savefig
import warnings
import glob
from types import SimpleNamespace

__author__ = ("Joseph O'Leary",)


class Universe:
    """Universe is a class for managing output data of Emerge.

    This class references the respective parameter file directly to ensure all
    data is pulled from the same directory and that a consistent set of
    parameters are used in analysis.

    """

    def __init__(self, param_path, sim_type="EMERGE", include=None):
        """Initialize a Universe object

        Parameters
        ----------
        param_path : str
            Path to a `.param` file.
        sim_type : str, optional
            The type of simulation data for the universe, by default 'EMERGE'
        include : str list of str, optional
            A list of data classes that should be initialized in the universe, by default None

        """

        self.param_path = os.path.abspath(param_path)
        self.sim_type = sim_type.upper()

        # set the sim type io routines i really really hate how this is done
        if self.sim_type == "EMERGE":
            params = em_io.read_parameter_file(self.param_path)
        else:
            raise NotImplementedError("Currently only Emerge data is supported")

        if self.sim_type == "EMERGE":
            self.params = em.params(self.param_path)
            self.emerge_dir = os.path.abspath(
                self.param_path.split("parameterfiles")[0]
            )
            self.out_dir = os.path.join(
                self.emerge_dir, "output", self.params.get_param("ModelName")
            )
            self.TreefileName = self.params.get_param("TreefileName")
            tree_dir, tree_base = os.path.split(self.TreefileName)

            # check if this is a directory already
            if os.path.isdir(tree_dir):
                pass
            else:
                cwd = os.getcwd()
                # temprorarily set the working directory to the emerge directory
                os.chdir(self.emerge_dir)
                if os.path.isabs(self.TreefileName):
                    # assume it should have been relative since it failed the `isdir` check
                    self.TreefileName = "." + self.TreefileName
                self.TreefileName = os.path.abspath(self.TreefileName)
                os.chdir(cwd)

            self.TreefileName = os.path.abspath(self.TreefileName)
            self.num_procs = self.params.get_param("NumFilesInParallel")

            template_config_path = os.path.abspath(
                self.emerge_dir + "/Template-Config.sh"
            )
            if os.path.isfile(template_config_path):
                if os.path.isfile(
                    os.path.abspath(self.out_dir + "/compile_options.txt")
                ):
                    self.config = em.config.from_compiled(
                        template_config_path,
                        os.path.abspath(self.out_dir + "/compile_options.txt"),
                    )
                else:
                    self.config = em.config(template_config_path)

        else:
            raise NotImplementedError("Currently only Emerge data is supported")

        if include is not None:
            include = list(np.atleast_1d(include))
            self.__include = [self.alias(key) for key in include]
            self.__add()

        else:
            self.fig_dir = self.out_dir

    @property
    def cosmology(self):
        return self.params.cosmology

    @classmethod
    def new(cls, emerge_dir, model_name, include=None):
        """Set up a new emerge universe by generating new parameter file, config and output dir.

        Parameters
        ----------
        emerge_dir : string
            Path to the local emerge repository.
        model_name : string
            Model name for the new emerge universe
        include : string, list, optional
            Description of parameter `include` (the default is None).

        Returns
        -------
        Universe
            An instance of an emerge object

        """
        emerge_dir = os.path.abspath(emerge_dir)
        if os.path.isdir(os.path.join(emerge_dir, "output/" + model_name)):
            raise FileExistsError("A model with this name already exists")
        else:
            os.mkdir(os.path.join(emerge_dir, "output/" + model_name))

        param_path = os.path.join(emerge_dir, "parameterfiles/emerge.param")
        params_temp = em.params(param_path)
        params_temp.optupdate("ModelName", value=model_name, force=True)
        param_path = os.path.join(emerge_dir, "parameterfiles/" + model_name + ".param")
        params_temp.write(param_path)

        u = cls(param_path=param_path, sim_type="emerge")

        if include is not None:
            include = list(np.atleast_1d(include))
            include = [idx.upper() for idx in include]
            u.__setattr__("_Universe__include", include)
        return u

    @classmethod
    def clone(
        cls,
        universe,
        model_name,
        clone_current=False,
        clone_include=False,
        include=None,
    ):
        """Clone an existing emerge Universe.

        Parameters
        ----------
        universe : string, object
            The parameter file of the unierse to be cloned (string), or an instance of the Universe object to be cloned.
        model_name : type
            Name for the new universe.
        clone_current : boolean
            If true the `current` state of a universe object will be used. If false the initial state of the object will be cloned (the default is False).
        clone_include : boolean
            If true the initial include argument for the universe will also be cloned (the default is False).
        include : string, list
            A string or list of strings specifying which data should be loaded after running the model (the default is None).

        Returns
        -------
        Universe
            An instance of an emerge object

        """
        if isinstance(universe, str):
            param_path = os.path.abspath(universe)
            universe = cls(param_path=param_path, sim_type="emerge")
        else:
            param_path = universe.param_path

        emerge_dir = universe.emerge_dir

        if os.path.isdir(os.path.join(emerge_dir, "output/" + model_name)):
            raise FileExistsError("A model with this name already exists")
        else:
            os.mkdir(os.path.join(emerge_dir, "output/" + model_name))

        params_temp = copy.deepcopy(universe.params)
        config_temp = copy.deepcopy(universe.config)

        if not clone_current:
            params_temp.reset()
            config_temp.reset()

        if clone_include:
            include = copy.deepcopy(universe._Universe__include)

        params_temp.optupdate("ModelName", value=model_name, force=True)
        param_path = os.path.join(emerge_dir, "parameterfiles/" + model_name + ".param")
        params_temp.write(param_path)
        config_path = os.path.join(
            emerge_dir, "output/" + model_name + "/compile_options.txt"
        )
        config_temp.write_compiled(config_path)

        # create a universe...
        u = cls(param_path=param_path, sim_type="emerge")

        if include is not None:
            include = list(np.atleast_1d(include))
            include = [idx.upper() for idx in include]
            u.__setattr__("_Universe__include", include)
        return u

    def alias(self, key):
        """Return proper coloumn key for input alias key.

        Parameters
        ----------
        key : str
            A string alias for a galaxy tree column

        Returns
        -------
        str
            The proper column name for the input alias
        """

        col_alias = {}
        # add other aliases.
        col_alias["statistics"] = ["statistics", "stats"]
        col_alias["galaxy_catalog"] = ["galaxy_catalog", "galcat"]
        col_alias["galaxy_trees"] = [
            "galaxy_trees",
            "gtree",
            "gtrees",
            "galtree",
            "trees",
        ]
        col_alias["survey"] = ["survey"]
        col_alias["halo_trees"] = ["halo_trees", "htree", "htrees"]
        col_alias["galaxy_mergers"] = ["galaxy_mergers", "gmergers", "mergers"]
        col_alias["halo_mergers"] = ["halo_mergers", "hmergers"]
        col_alias["fig_dir"] = ["fig_dir", "figdir"]
        col_alias["fitting_files"] = ["mcmc", "fits", "fitting", "fit"]

        for k in col_alias.keys():
            if key.lower() in col_alias[k]:
                return k
        raise KeyError("`{}` has no known alias.".format(key))

    def __add(self):
        """Load data that was provide to `include`."""
        if "fig_dir" in self.__include:
            self.add_figdir(os.path.join(self.out_dir, "figures/"))
        else:
            self.fig_dir = self.out_dir

        if "statistics" in self.__include:
            self.add_statistics()

        if "galaxy_trees" in self.__include:
            self.add_galaxy_trees()

        if "survey" in self.__include:
            self.add_survey()

        if ("galaxy_catalog" in self.__include) and (
            "galaxy_trees" not in self.__include
        ):
            self.add_galaxy_catalog()

        if "halo_trees" in self.__include:
            self.add_halo_trees()

        if "galaxy_mergers" in self.__include:
            self.add_galaxy_mergers()

        if "halo_mergers" in self.__include:
            self.add_halo_mergers()

        if "fitting_files" in self.__include:
            self.add_fits()

    def add_figdir(self, directory_path):
        """Create a directory for saving figures."""
        directory = os.path.abspath(directory_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.fig_dir = directory

    def add_statistics(self):
        """Add the statistsics class as a Universe attribute."""
        if self.sim_type == "EMERGE":
            print("Loading statistics:")
            self.statistics = em.statistics.from_universe(self)
        else:
            print(
                "The statistics class is not available for simulation type "
                + self.sim_type
                + "."
            )

    def add_galaxy_trees(self):
        """Add the forest class as a Universe attribute."""
        if self.sim_type == "EMERGE":
            self.galaxy = em.galaxy_trees.from_universe(self)
        else:
            print(
                "The galaxy_trees class is not available for simulation type "
                + self.sim_type
                + "."
            )

    def add_galaxy_mergers(self, source=None, save=True, **kwargs):
        """Add the `mergers` class as a Universe attribute.

        This function initialises `mergers` class from a specified source.
        As merger lists are a derived dataset, if not existing `mergers.h5` file
        can be located one will be created from an existing `forest`.

        Parameters
        ----------
        source : type
            Description of parameter `source` (the default is None).
        save : type
            Description of parameter `save` (the default is True).


        Returns
        -------
        type
            Description of returned object.

        """

        if self.sim_type == "EMERGE":
            if source is None:
                try:
                    add = {
                        "out_dir": self.out_dir,
                        "fig_dir": self.fig_dir,
                        "ModelName": self.ModelName,
                        "BoxSize": self.BoxSize,
                        "UnitTime_in_yr": self.UnitTime_in_yr,
                        "OutputMassThreshold": self.OutputMassThreshold,
                        "cosmology": self.cosmology,
                    }
                    mergers = em.galaxy_mergers.from_file(
                        os.path.join(self.out_dir, "mergers.h5"), add_attrs=add
                    )
                except:
                    mergers = em.galaxy_mergers.from_galaxy_trees(
                        self.galaxy, save=save, **kwargs
                    )

            elif source == "trees":
                mergers = em.galaxy_mergers.from_galaxy_trees(
                    self.galaxy, save=save, **kwargs
                )
            else:
                add = {
                    "out_dir": self.out_dir,
                    "fig_dir": self.fig_dir,
                    "ModelName": self.ModelName,
                    "BoxSize": self.BoxSize,
                    "UnitTime_in_yr": self.UnitTime_in_yr,
                    "OutputMassThreshold": self.OutputMassThreshold,
                    "cosmology": self.cosmology,
                }
                mergers = em.galaxy_mergers.from_file(source, add_attrs=add)

            if hasattr(self, "galaxy"):
                self.galaxy.mergers = mergers
            else:
                self.galaxy_mergers = mergers
        else:
            print(
                "The merger class is not available for simulation type "
                + self.sim_type
                + "."
            )

    def add_halo_trees(self, include="halos"):
        """Add the halos class as a Universe attribute."""
        if self.sim_type == "EMERGE":
            self.halo = em.halo_trees.from_universe(self, include=include)
        else:
            print(
                "The halo trees class is not available for simulation type "
                + self.sim_type
                + "."
            )

    def add_halo_mergers(self, source=None, save=True):
        """Add the `mergers` class as a Universe attribute.

        This function initialises `mergers` class from a specified source.
        As merger lists are a derived dataset, if not existing `mergers.h5` file
        can be located one will be created from an existing `forest`.

        Parameters
        ----------
        source : type
            Description of parameter `source` (the default is None).
        save : type
            Description of parameter `save` (the default is True).


        Returns
        -------
        type
            Description of returned object.

        """
        if self.sim_type == "EMERGE":
            if source is None:
                try:
                    add = {
                        "out_dir": self.out_dir,
                        "fig_dir": self.fig_dir,
                        "ModelName": self.ModelName,
                        "BoxSize": self.BoxSize,
                        "UnitTime_in_yr": self.UnitTime_in_yr,
                        "cosmology": self.cosmology,
                    }
                    mergers = em.halo_mergers.from_file(
                        os.path.join(self.out_dir, "halo_mergers.h5"), add_attrs=add
                    )
                except:
                    mergers = em.halo_mergers.from_halo_trees(self.halo, save=save)

            elif source == "trees":
                mergers = em.halo_mergers.from_halo_trees(self.halo, save=save)
            else:
                add = {
                    "out_dir": self.out_dir,
                    "fig_dir": self.fig_dir,
                    "ModelName": self.ModelName,
                    "BoxSize": self.BoxSize,
                    "UnitTime_in_yr": self.UnitTime_in_yr,
                    "cosmology": self.cosmology,
                }
                mergers = em.halo_mergers.from_file(source, add_attrs=add)

            if hasattr(self, "halo"):
                self.halo.mergers = mergers
            else:
                self.halo_mergers = mergers
        else:
            print(
                "The halo merger class is not available for simulation type "
                + self.sim_type
                + "."
            )

    def add_survey(self):
        """Add a galaxy survey to the universe."""
        add = {
            "out_dir": self.out_dir,
            "fig_dir": self.fig_dir,
            "ModelName": self.ModelName,
            "BoxSize": self.BoxSize,
            "UnitTime_in_yr": self.UnitTime_in_yr,
            "cosmology": self.cosmology,
        }

        survey = em.survey(
            file_path=os.path.join(self.out_dir, "survey.h5"), add_attrs=add
        )
        if hasattr(self, "galaxy"):
            self.galaxy.survey = survey
            if "galaxy_trees" in self.__include:
                # if trees are available. link them.
                self.galaxy.survey.link_trees(self.galaxy)
        else:
            self.galaxy_survey = survey

    def add_fits(self):
        fit = {}
        files = []
        for name in glob.glob(self.out_dir + "/pt*"):
            files.append(name)
        if len(files) > 0:
            fit["pt"] = em.parallel_tempering(files)

        files = []
        for name in glob.glob(self.out_dir + "/mcmc*"):
            files.append(name)
        if len(files) > 0:
            fit["mcmc"] = em.mcmc(files)

        self.fit = SimpleNamespace(**fit)

    def flush(self):
        """Delete all data structures in Universe except for config, and params."""
        if self.sim_type == "EMERGE":
            print("Flusing data structures...")
            if hasattr(self, "statistics"):
                del self.statistics
            if hasattr(self, "galaxy"):
                del self.galaxy
            if hasattr(self, "galaxy_mergers"):
                del self.galaxy_mergers
            if hasattr(self, "halo"):
                del self.halo
            if hasattr(self, "halo_mergers"):
                del self.halo_mergers
            if hasattr(self, "galaxy_survey"):
                del self.galaxy_survey
            if hasattr(self, "fit"):
                del self.fit

    def reload(self):
        """Flush then reload data structures in Universe.

        Only data that was originally in the `include` argument are reloaded.
        """
        self.flush()
        self.__add()

    def make(self, clean=False, **kwargs):
        """Execute `make` command in Emerge directory.

        Parameters
        ----------
        clean : boolean, optional
            Whether to run `make` or `meak clean` (the default is False).

        """
        if self.sim_type == "EMERGE":
            if clean:
                cmd(["make", "clean"], path=self.emerge_dir, **kwargs)
            else:
                cmd("make", path=self.emerge_dir, **kwargs)
        else:
            raise TypeError(
                "The `make()` method is not available for simulation type "
                + self.sim_type
                + "."
            )

    def run(self, **kwargs):
        """Execute `emerge` using loaded config and parameterfile."""
        if self.sim_type == "EMERGE":
            cmd(
                [
                    "mpiexec",
                    "-np",
                    "{:d}".format(self.num_procs),
                    "./emerge",
                    self.param_path,
                ],
                path=self.emerge_dir,
                **kwargs
            )
        else:
            raise TypeError(
                "The `run()` method is not available for simulation type "
                + self.sim_type
                + "."
            )

    def update(
        self,
        params=True,
        config=True,
        flush=False,
        clean=False,
        make=False,
        run=False,
        reload=False,
    ):
        """Update params/config then execute and reload universe.

        Parameters
        ----------
        params : bool
            Update emerge parameter file with current configuration (the default is True).
        config : bool
            Update emerge `Config.sh` with current config (the default is True).
        flush : bool
            Clear loaded data structures (the default is False).
        clean : bool
            Run the `clean` command in emerge directory (the default is False).
        make : bool
            Run the `make` command in emerge directory (the default is False).
        run : bool
            Execute emerge with current parameterfile and config (the default is False).
        reload : bool
            Reload data from initial `include` argument (the default is False).

        """
        if params:
            self.params.write(self.param_path)
        if config:
            self.config.write(self.emerge_dir + "/Config.sh")
        if flush:
            self.flush()
        if clean:
            self.make(clean=True)
        if make:
            self.make()
        if run:
            self.run()
        if reload:
            self.reload()

    def savefig(self, fname, **kwargs):
        """Save figure in directory set to Universe class.

        This function is a wrapper to `matplotlib.pyplot.savefig()` refer to their
        documentation for further details on usage.

        Parameters
        ----------
        fname : str
            The output figure name
        """

        if hasattr(self.config, "build"):
            if "metadata" in kwargs.keys():
                kwargs["metadata"] = {**kwargs["metadata"], **self.config.build}
            else:
                kwargs["metadata"] = self.config.build.copy()

            # this will catch the user warning for `Unknown infodict keyword` so we can attach build info
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                savefig(os.path.join(self.fig_dir, fname), **kwargs)
        else:
            savefig(os.path.join(self.fig_dir, fname), **kwargs)

# galaxybox

A collection of mostly personal tools and scripts that I use for my work.

## Installation

Clone this repository by executing

```bash
git clone https://github.com/jaoleary/galaxybox.git
```

move into the repsitory directy with:

```bash
cd galaxybox
```

Then install by executing:

```bash
python setup.py install
```

## Basic usage

These tools are primarly intended for use with [Emerge](http://www.usm.uni-muenchen.de/emerge/) data. Accordingly, to make the most use of these tools it is best if you install and run the model yourself, as much of the functionally comes with the expectiation that files follow the Emerge directory structure.

I primarily keep my data/results structured through use of the `Universe` class, which takes an Emerge parameter file as an input. In this example I am loading a model I have named `P200`. Using these structures makes it easy to load multiple models into the same python instance for comparison, without mixing up their parameters, cosmology or output files.

```python
from galaxybox import sim_managers as gb

P200 = gb.Universe(param_path = '../Programs/emerge/parameterfiles/P200.param', sim_type = 'emerge')
```

The basic import allows the user to inspect/edit the parameter file, as well as the configuration file used to run a given model. these structures can be accessed with:

```python
P200.params.current()
P200.config.current()
```

Output data from emerge can be loaded into the object with the 'add' methods. For instance the **statsitics.h5** file can be imported by executing:

```python
P200.add_statistics()
```

This file contains the best fit to observables produced by emerge when using the specified parameter file. these can readily be plotted with the plot method. For example, the galaxy stellar mass function can be produce with:

```python
P200.statistics.plot('smf')
```

Other emerge data can also be added into the struct. Currently galaxy merger trees have the best support as that is what I work with most.

```python
P200.galaxy_trees()
```

Similar to the statistics file this will attach a new class to the universe object. This imports **all** available galaxy merger trees corresponding to this parameter file and drops them into a single monolithic pandas dataframe. For small simulations this is fine, though you might run into memory issues for larger simulation volumes. The basic tree dataframe can be accessed by calling:

```python
P200.galaxy.trees
```

If however you want to know the growth history of an individual galaxy and you know the ID of that galaxy you can access a single tree with:

```python
P200.galaxy.tree(igal=`some_ID_number`)
```

Without any additional information this method just executes a straight recursive tree walk to get the growth of `igal`. This can be quite slow, if you ran emerge with reindexing enabled, then you could also first sort the trees for faster retreival. This is accomplished easily with:

```python
P200.galaxy.sort_index()
```

The tree can then be accessed in the same way as before.

If you want to access galaxy lists for a single redshift(snapshot) this can be done using the `list()` method. However, this method sacrifices clarify for convenience so it doesnt accept an explicit list of arguments. The basic appraoch to is to accept a min/max argument for any of the columns appearing in the `P200.galaxy.trees` dataframe. For instance a galaxy catalog at z=1 for a specific mass range could be obtained from:

```python
P200.galaxy.list(redshift=1, min_Stellar_mass=10, max_Stellar_mass=11)
```

The columns of the galaxy trees files are meant to be descriptive, not convenient. So for my own sanity aliasing is available for most columns. This same galaxy catalog could be acquired with:

```python
P200.galaxy.list(z=1, min_mstar=10, max_mstar=11)
```

You can check if an alias is valid using the `alias()` method.

The `list()` method also lets you select galaxies based on derived properties such as color. You could for instance select all passive galaxies in the trees by running:

```python
P200.galaxy.list(redshift='all', color='red')
```

This is just a snippet of what you can do with with these tools. Most of the code should be commented, but if there is a question regarding how something works, or if youd like to see something implemented just contact me. Further examples of how I use these tools can be found in the jupyter-notebooks for my papers.

## Notebooks

Paper: [EMERGE: Empirical predictions of galaxy merger rates since z~6](https://ui.adsabs.harvard.edu/abs/2020arXiv200102687O/abstract)\
Notebook: [2001.02687](https://github.com/jaoleary/2001.02687)

Paper: [EMERGE: Constraining merging probabilities and timescales of close galaxy pairs](https://ui.adsabs.harvard.edu/abs/2020arXiv201105341O/abstract)\
Notebook: [2011.05341](https://github.com/jaoleary/2011.05341)

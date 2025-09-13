# GalaxyBox

[![PyPI version](https://badge.fury.io/py/galaxybox.svg)](https://badge.fury.io/py/galaxybox)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/jaoleary/galaxybox)](LICENSE)
[![Tests](https://github.com/jaoleary/galaxybox/workflows/galaxybox-build-validation/badge.svg)](https://github.com/jaoleary/galaxybox/actions)


## Features

- **Galaxy Merger Trees**: Efficient tools for processing and analyzing galaxy merger trees
- **Data Management**: Streamlined data loading and manipulation for large simulation datasets
- **Visualization**: Specialized plotting tools for galaxy evolution and merger tree visualization
- **Mock Observables**: Generate synthetic observations and lightcone catalogs
- **EMERGE Integration**:Ssupport for [EMERGE](https://github.com/bmoster/emerge) model data

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager. Install GalaxyBox with:

```bash
uv add galaxybox
```

Or in a new project:

```bash
uv init my-project
cd my-project
uv add galaxybox
```

### From PyPI

```bash
pip install galaxybox
```

### From Source

```bash
git clone https://github.com/jaoleary/galaxybox.git
cd galaxybox
uv pip install .
```

### Development Installation

For development and contributing, we use uv for dependency management:

```bash
git clone https://github.com/jaoleary/galaxybox.git
cd galaxybox
uv sync --group dev
```

## Quick Start

```python
import galaxybox as gb
from galaxybox.modules.trees.emerge import EmergeGalaxyTrees

# Load galaxy merger trees
trees = EmergeGalaxyTrees("/path/to/tree/data")

# Get galaxy catalog at z=0
galaxies = trees.list(z=0, min_mstar=9.0)

# Extract individual merger tree
tree = trees.tree(galaxy_id=12345)
```

## Publications

GalaxyBox has been used in the following:

- **[EMERGE: Empirical predictions of galaxy merger rates since z~6](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3215O/abstract)**
  *Monthly Notices of the Royal Astronomical Society, 2021*

- **[EMERGE: Constraining merging probabilities and timescales of close galaxy pairs](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5646O/abstract)**
  *Monthly Notices of the Royal Astronomical Society, 2020*

- **[Predictions on the stellar-to-halo mass relation in the dwarf regime using the empirical model for galaxy formation EMERGE](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520..897O/abstract)**
  *Monthly Notices of the Royal Astronomical Society, 2023*

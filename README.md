# `victor` - Python code for cross-correlation function modelling and fitting

## Introduction

**`victor`** is a Python package for modelling and likelihood analysis of the cross-correlation function of density regions (e.g., voids or density-split centres) with galaxies. It can be used for posterior sampling using MCMC through an interface with [`cobaya`](https://cobaya.readthedocs.io/en/latest/).

## Installation

### Pre-requisites

The only pre-requisites are *Python* (version>=3.7.4) and the Python package manager *pip* (version>=20.0).

### Installing with pip

To install `victor`, do
```
python -m pip install git+https://github.com/seshnadathur/victor.git
```
This will install the package with the minimal required packages (`numpy`, `scipy`, `matplotlib`, `astropy`, `h5py` and `PyYAML`). Some features of the code require additional packages:
   - [`camb`](https://camb.readthedocs.io/en/latest/) (for more accurate calculation of the matter power spectrum in certain models)
   - [`cobaya`](https://cobaya.readthedocs.io/en/latest/) (for posterior sampling via MCMC)
   - [`GetDist`](https://getdist.readthedocs.io/en/latest/) (analysis/visualisation of the results of posterior sampling)

To install with these extra dependencies, do
```
python -m pip install git+https://github.com/seshnadathur/victor.git#egg=victor[all]
```
Here the `[all]` will install all extra dependencies (`camb`, `cobaya` and `GetDist`); replacing this with `[mcmc]` will install only `cobaya` and `GetDist` for MCMC analyses, and replacing it with `[camb]` will install only `camb`.

*Note*: To enable MPI parallelisation with `cobaya` (highly recommended), you may wish to separately install the package `mpi4py` (or use the corresponding module on your cluster). To do this, follow the instructions [here](https://cobaya.readthedocs.io/en/latest/installation.html) before installing.

### Installing in development mode

If you want edit the code and to collaborate on further development of `victor` the best way to do this is using `git`. Fork the [`victor` repo](https://github.com/seshnadathur/victor) on GitHub then clone your fork using
```
git clone https://YOUR_USERNAME@github.com/YOUR_USERNAME/victor.git
```
and then install in editable mode using
```
python -m pip install -e .[all]
```
where the `.[full]` at the end is optional and will install all the extra dependencies as above. (See (here)[https://stackoverflow.com/questions/30239152/specify-extras-require-with-pip-install-e] if you use `zsh` instead of `bash`.)

Alternatively, simply do
```
python -m pip install -e git+https://github.com/seshnadathur/victor.git#egg=victor[all]
```
(but this means you won't be able to share your changes via pull requests).

## Documentation:

Full documentation (including API) is a work in progress and will be updated soon. In the meantime, look at [victor_usage_demo.ipynb](notebooks/victor_usage_demo.ipynb) for worked examples of typical use cases, and [model_options_demo.ipynb](notebooks/model_options_demo.ipynb) for a fairly extensive summary of various model evaluation options.

## License

**`victor`** is free software distributed under a GNU GPLv3 license. For details see the [LICENSE](LICENSE).

# `Victor`

Python code for modelling and likelihood analysis of void-galaxy cross-correlation data

Posterior sampling is performed using an interface to the [`cobaya`](https://cobaya.readthedocs.io/en/latest/) Monte
Carlo framework code.

### Requirements
   - python version >= 3.7
   - ```numpy```
   - ```scipy```
   - [`pyyaml`](https://pypi.org/project/PyYAML/)
   - [`camb`](https://camb.readthedocs.io/en/latest/) 
   - [`cobaya`](https://cobaya.readthedocs.io/en/latest/) (for posterior sampling)
   - (optional) [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/) 
   
Additionally, to analyse chains produced using `cobaya` and to run parts of the example notebook included here you will 
need [`GetDist`](https://getdist.readthedocs.io/en/latest/). 

### Usage:

The main functionality of `Victor` is provided to the user via the [`VoidGalaxyCCF`](models/void_galaxy_ccf.py) class. 
To initialise an instance of this class you must pass two Python dictionaries, `paths` containing a list of paths to 
necessary data files, and `settings` containing options to be used in the model calculations (see below for details). 
This allows for modular usage of the code.

In order to sample the posterior we use the MCMC capabilities of `cobaya`, via the 
[`VoidGalaxyLikelihood`](likelihoods/VoidGalaxyLikelihood.py) external likelihood class. To use `cobaya`, additional 
inputs must be provided, specifying the likelihood options to be used, the sampler and its options, the parameters to be 
sampled over, priors, output folder etc. in addition to `paths` and `settings` required for `VoidGalaxyCCF` as above. 

The inputs required for `cobaya` can be provided directly via a Python interpreter (or Jupyter notebook), or can be 
read from input text files in YAML format (see the 
[`cobaya` documentation](https://cobaya.readthedocs.io/en/latest/example.html)). Example `cobaya` input files are 
provided in the [cobaya_inputs/]() folder; edit these according to your requirements. To run chains sampling the 
posterior then simply use `cobaya-run` as described in the `cobaya` instructions:

1. To run a single chain, do  

    ```$ cobaya-run <path/to/input/yaml/file>```
    
2. Or if you have installed `mpi4py`, then to run several chains in parallel and obtain faster convergence, do 

    ```$ mpirun -n [n_processes] cobaya-run <path/to/input/yaml/file>```
  
   (On some HPC systems you might want to replace `mpirun` with `srun`.)
   
That's it! After a few seconds, chain files and other associated outputs will start to appear in the output directory 
you specified. 

### Inputs and model choices:

A full description of all input options and specifications will be provided soon, along with an example Jupyter 
notebook. A brief summary is provided for now.

#### Settings:

Some key elements of the `settings` dictionary can be set as follows:

- `model`: specify the choice of model used of calculating the void-galaxy correlation
    - `dispersion`: use the generalised dispersion model of Nadathur *et al.* 2019 
    ([arXiv:1904.01030](https://arxiv.org/abs/1904.01030)) and Nadathur *et al.* 2020 
    ([arXiv:2008.06060](https://arxiv.org/abs/2008.06060))
    - `streaming`: use the Gaussian streaming formulation of Cai *et al.* 2016 
    ([arXiv:1603.05184](https://arxiv.org/abs/1603.05184))
    - `Kaiser`: the simpler Kaiser model used in several papers, with the additional option:
        - `approx_Kaiser`: `False` (default) uses the exact form of the Kaiser model correct to all orders, as given 
        in Nadathur *et al.* 2020 ([arXiv:2008.06060](https://arxiv.org/abs/2008.06060)); if `True`, uses the less 
        accurate truncated expansion in Cai *et al.* 2016 ([arXiv:1603.05184](https://arxiv.org/abs/1603.05184)) and 
        several subsequent papers
     
    `dispersion` and `streaming` models will give very similar results to each other (see Paillas *et al.* 2021, 
[arXiv:2101.05184](https://arxiv.org/abs/2101.09854))

- `do_coord_shift`: if `True` (default), all model calculations correctly account for the shift from real-space to 
redshift-space coordinates in the argument of redshift-space void-galaxy correlation as described by Nadathur & Percival 
2019 ([arxiv:1712.07575](https://arxiv.org/abs/1712.07575)); if `False`, these corrections are ignored, as in the 
expressions 
presented by Cai *et al.* 2016 ([arXiv:1603.05184](https://arxiv.org/abs/1603.05184)) and several other papers
 
- `delta_profile`: sets the choice of how to determine the void-matter profile used the model calculation, with the 
following options: 
    -  `use_linear_bias`: assumes the void-matter monopole is simply related to the void-galaxy monopole by the constant 
    linear galaxy bias – this requires input parameter `beta` (=f/b) instead of `fsigma8`
    - `use_template`: performs a template fit using a template void-matter monopole function read from file (which must 
    be provided in `paths`); you must also separately specify `template_sigma8` in `settings` this is the value of 
    sigma_8(z) in the N-body simulation and for the redshift z at which the template was constructed
    - `use_excursion_model`: uses the excursion set model approach of Massara & Sheth 2018 
    ([arXiv:1811.03132](https://arxiv.org/abs/1811.03132)) – this requires a different set of input parameters, see the 
    notes in [likelihoods/VoidGalaxyLikelihood.yaml]()  

- `fit_to_data`: `False` if only model calculations are required and not fits to data. If `True` then the code expects 
paths to data files containing the redshift space void-galaxy multipole measurements and a covariance matrix. Note that 
even if this is `False`, a path to a file containing the real space void-galaxy multipoles (containing at least the 
monopole) *must* be provided to enable model calculations

- `data_uses_reconstruction`: set to `True` if RSD removal via reconstruction was performed prior to void-finding as 
advocated by Nadathur, Carter & Percival 2019
([arXiv:1805.09349](https://arxiv.org/abs/1805.09349)). Reconstruction depends on the input value of `beta` (=f/b)), so 
the code then assumes that the multipole data vectors have been evaluated on a grid of `beta` values, and requires that 
`paths` contain a path to a data file containing the grid values. Interpolation is used to obtain the data vectors at 
intermediate points.

- `fixed_covmat`: if `True` (default) then the covariance matrix is fixed independent of the value of beta in 
reconstruction prior to void-finding; if `False` then the covariance matrix should also be provided on a grid of `beta` 
and will be interpolated at intermediate points

- `likelihood_type`: specifies the form of the likelihood to use to account for propagation of uncertainties in the 
estimation of the covariance matrix, with the following options:
    - `Sellentin`: use the prescription of Sellentin & Heavens 2016 applying the Wishart distribution for a covariance 
    matrix estimated from a finite number of mocks; this further requires specification of the number of mocks used in 
    this covariance matrix estimation (see the example input files)
    - `Hartlap`: apply the common Hartlap correction instead; as above it requires specification of the number of mocks 
    used to estimate the covariance matrix
    - anything else: assumes the covariance is exact so applies no correction and uses a Gaussian likelihood
    
Additional options present are not described here (yet) but hopefully are self-explanatory from the example input files 
provided.

#### Input data files

Depending on the settings chosen, a number of input data files may be required by the code and paths to these must be 
specified in the `paths` dictionary. 

- `realspace_multipole_file`: File containing the real-space void-galaxy monopole (higher multipoles can be included 
but are not required). This file *must* be present, as the real-space monopole is required for any model calculations. 
If `settings[data_uses_reconstruction]` is `False` then the multipoles in this file should be an array of length 
`N = number_of_multipoles * number_of_bins`; if `True` then it should be a 2D array of shape `M x N` where `M` is the 
number of bins in the grid of beta

- `redshiftspace_multipole_file`: Required if `settings[fit_to_data]` is `True`. The shape of the array of multipoles in 
the file depends on `settings[data_uses_reconstruction]` as above

- `covariance_matrix_file`: Required if `settings[fit_to_data]` is `True`. If `settings[data_uses_reconstruction]` is 
`True` *and* If `settings[fixed_covmat]` is False, then `beta`-dependence is assumed and the covariance matrix should be 
provided on a grid, as for the multipoles, i.e. as an array of shape `M x N x N`

- `multipole_beta_grid_file`: Required if `settings[data_uses_reconstruction]` is `True`; should then contain the grid 
of `beta` values on which the multipole data vectors were evaluated

- `delta_template_file`: Required if `settings[use_template]` is `True`; contains template void-matter correlation 
monopole (i.e., void matter density profile) to use in the template fit

- `velocity_dispersion_template_file`: optional, used if either the `dispersion` or `streaming` models are chosen. If 
not provided, the code will simply assume a constant velocity dispersion function.


Note that most data files must contain Python dictionaries and all need to be in `numpy` pickle format with `.npy` 
file-endings (`h5py` functionality may be added later). The formats of the data required in these files can be checked 
by comparing to the data provided in the [BOSS_DR12_CMASS_data/]() and [eBOSS_DR16_LRG_data/]() folders.

### SDSS Data:

Processed data files for void-galaxy measurements from the SDSS BOSS Data Release 12 CMASS sample described in Nadathur 
*et al.* 2019 ([arXiv:1904.01030](https://arxiv.org/abs/1904.01030)) are provided in the folder
[BOSS_DR12_CMASS_data/](). The equivalent data from the eBOSS Data Release 16 luminous red galaxy sample, described in 
Nadathur *et al.* 2020 ([arXiv:2008.06060](https://arxiv.org/abs/2008.06060)), are provided in the folder 
[eBOSS_DR16_LRG_data/]()). 

### Why is the code called `Victor`?
Why not?

(Originally `Victor` was an acronym for VoId-galaxy CorrelaTion cosmolOgy fitteR – generated using
[```acronym```](https://github.com/bacook17/acronym), of course. But now it is just `Victor`.)

# Victor

General Python code for likelihood analysis of void-galaxy cross-correlation data

### Requirements
   - python version 3.7 or higher 
   - ```numpy``` 
   - ```scipy``` 
   - [```emcee```](https://emcee.readthedocs.io/en/stable/) (for MCMC sampling)
   - [```getdist```](https://getdist.readthedocs.io/en/latest/) (for MCMC sampling)

### Quickstart usage:
1. Create an input parameter file (see the example provided in [parameters/generic_likelihood_params.py]()) 
to point to the data files you want to fit and set modelling options (note the special format required for 
data files!)
2. To run an MCMC posterior sampler using ```emcee```, do 
```python run_sampler.py --param_file <path/to/input/parameter/file>``` 
 
For subsequent runs if you wish to create additional chains, use the ```-c <N>``` argument to specify integer 
argument N to be appended to the end of subsequent chains (if option ```-c``` is not specified, it defaults 
to 1).

### Modular usage:
The code can also be used in modular fashion, e.g. for plotting theory models or data multipoles, calculating 
chi-squared values, etc. To do this, do

```from posterior_samplers.void_galaxy_posterior import VoidGalaxyPosterior```

and initialise an instance of class ```VoidGalaxyPosterior``` by passing it the input options from your 
parameter file (see the simple code in [run_sampler.py](run_sampler.py) for an example of how to do this, I 
will eventually include a Jupyter notebook with demos). 

### Multiprocessing:
[run_sampler.py](run_sampler.py) uses the ```multiprocessing``` package to automatically use all available
CPU cores to run the MCMC sampling. 

### Theoretical models implemented:
Three different theory models are coded, and the choice of model is specified in the input parameter file. 
Note that the three models produce **very** different predictions: their relative merits and comparisons to 
simulation results are described in [arXiv:1712.07575](https://arxiv.org/abs/1712.07575).

**Model 1:** 
The dispersion model of Nadathur *et al.* 2019 ([arXiv:1904.01030](https://arxiv.org/abs/1904.01030)) that 
correctly includes all linear-order terms. Setting the velocity dispersion to a constant and very small value 
(~10 km/s) is sufficient to recover the basic (Kaiser) model of Nadathur & Percival
([arXiv:1712.07575](https://arxiv.org/abs/1712.07575)).

**Model 2:**
The "quasi-linear" model of Cai *et al.* 2016 ([arXiv:1603.05184](https://arxiv.org/abs/1603.05184)). This is
based on applying the Gaussian Streaming Model for the galaxy autocorrelation in redshift space to the 
void-galaxy cross-correlation by analogy.

**Model 3:**
The linear "multipole ratio" model of Cai *et al.* (2016).

Additional options for models 1 and 2 (e.g., whether to assume linear galaxy bias within voids, and the form 
of the velocity dispersion function) can also be specified in the input parameter file. 

### BOSS DR12 CMASS data:
The file [parameters/boss_cmass_params.py]() provides the appropriate input options to reproduce the fits to 
the void-galaxy measurements in the BOSS DR12 CMASS sample described in Nadathur *et al.* 2019 
(https://arxiv.org/abs/1904.01030). The required data files are provided in the folder 
[BOSS_DR12_CMASS_data/]().

### Why is the code called Victor?
Why not? 

(Originally Victor was an acronym for VoId-galaxy CorrelaTion cosmolOgy fitteR – generated using 
[```acronym```](https://github.com/bacook17/acronym), of course. But now it is just Victor.)
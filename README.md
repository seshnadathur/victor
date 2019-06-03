# void-galaxy-cosmo-fitter
General Python code for likelihood analysis and MCMC posterior sampling of measured 
void-galaxy cross-correlation data post-reconstruction, including data for BOSS 
DR12 CMASS.

##### The code is modular and can be used in different ways: 
 - Classes defined in posterior_samplers/ provide likelihood modules for different 
 data samples, currently limited to BOSS DR12 data and simulations. These can be 
 imported and analysed individually: see the provided notebook 
 **void-cosmo-fitter.ipynb** for an example that reproduces the main analysis and 
 figures from *Beyond BAO: improving cosmological constraints from BOSS with 
 measurement of the void-galaxy cross-correlation*, **S. Nadathur** et al., 2019 
 (https://arxiv.org/abs/1904.01030) for BOSS DR12 CMASS data, for which data and 
 MCMC chains are provided
 - To rerun your own MCMC chains for BOSS DR12 data, use **run_mcmc.py**. To see 
 usage options, do 
 
    > python run_mcmc.py --help
 
 - **run_mcmc.py** can be easily edited to incorporate additional datasets and/or
 new likelihood modules
 
##### Requirements:

void-cosmo-fitter.ipynb requires:
 - scipy
 - matplotlib
 - GetDist (https://getdist.readthedocs.io/en/latest/)
 - astropy (https://www.astropy.org/) (optional)
 - corner (https://github.com/dfm/corner.py) (optional)
 
run_mcmc.py requires:
 - scipy
 - emcee v2.2.1 (https://emcee.readthedocs.io/en/v2.2.1/user/install/) (we use the 
 older version of emcee for now to maintain compatibility with Python2.7, this 
 will be updated in the future)
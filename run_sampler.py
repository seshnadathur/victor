import os
import sys
import argparse
import numpy as np
import emcee
import importlib.util
from multiprocessing import Pool, cpu_count
from posterior_samplers.void_galaxy_posterior import VoidGalaxyPosterior
from utilities.utilities import UtilMethods


parser = argparse.ArgumentParser(description='Run an MCMC sampling of the posterior for model fits '
                                             'to the void-galaxy cross-correlation function')
parser.add_argument('-p', '--param_file', type=str, help='filename for parameter specification file')
parser.add_argument('-c', dest='chain', default=1, type=int, help='chain number (default =1)')
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("name", args.param_file)
params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(params)

# create output directory if it doesn't exist
if not os.access(params.output_folder, os.F_OK):
    os.makedirs(params.output_folder)

# initialize the fitter
void_gal_fitter = VoidGalaxyPosterior(params)


# define a global log posterior function to be called by the MCMC sampler
def lnpost_global(theta):
    return void_gal_fitter.lnpost(theta)


nsteps, nwalkers, ndim, burnin = 1000, 20, int(void_gal_fitter.ndim), 50
if ndim == 4:
    # the case for Models 1 or 2 when not assuming linear bias
    start_values = [0.5, 1.2, 390, 1]
    scales = [0.05, 0.05, 10, 0.03]
    names = ['fs8', 'bs8', 'sigmav', 'epsilon']
    labels = [r'f\sigma_8', r'b\sigma_8', r'\sigma_{v_{||}}', r'\epsilon']
    ranges = {'fs8': [0.05, 1.5], 'bs8': [0.1, 2], 'sigmav': [250, 500], 'epsilon': [0.8, 1.2]}
elif ndim == 3:
    # the case for Models 1 or 2 when assuming linear bias
    start_values = [0.4, 390, 1]
    scales = [0.05, 50, 0.03]
    names = ['beta', 'sigmav', 'epsilon']
    labels = [r'\beta', r'\sigma_{v_{||}}', r'\epsilon']
    ranges = {'beta': void_gal_fitter.beta_prior_range, 'sigmav': [250, 500], 'epsilon': [0.8, 1.2]}
if ndim == 1:
    # the case for Model 3
    start_values = [0.4]
    scales = [0.05]
    names = ['beta']
    labels = [r'\beta']
    ranges = {'beta': void_gal_fitter.beta_prior_range}
rootname = os.path.join(params.output_folder, params.root)

start = np.asarray([start_values + np.random.randn(ndim) * scales * 1e-2 for i in range(nwalkers)])

os.environ["OMP_NUM_THREADS"] = "1"

ncpu = cpu_count()
print("Using {0} CPUs".format(ncpu))

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_global, pool=pool)
    state = sampler.run_mcmc(start, burnin, progress=True, thin_by=5)
    print("Done the burn-in, starting production run")
    sys.stdout.flush()
    sampler.reset()
    # the backend function in emcee gives lots of problems, so save to file by hand
    icount = 0
    continue_run = True
    while continue_run:
        state = sampler.run_mcmc(state, nsteps, progress=True, thin_by=5)
        part_chain = sampler.flatchain
        lnprob = sampler.get_log_prob().flatten()
        output = np.ones((part_chain.shape[0], part_chain.shape[1] + 2))
        output[:, 1] = lnprob
        output[:, 2:] = part_chain
        if icount == 0:
            full_chain = output
        else:
            full_chain = np.load(rootname + '_%d.npy' % args.chain)
            full_chain = np.vstack([full_chain, output])
        np.save(rootname + '_%d.npy' % args.chain, full_chain)
        icount += 1
        ntotal = icount * nsteps

        # now estimate the autocorrelation time
        full_chain = full_chain[:, 2:].reshape((int(full_chain.shape[0] / nwalkers), int(nwalkers), int(ndim)))
        print("After %d steps, autocorrelation times for parameters:" % ntotal)
        acor = np.empty(ndim)
        for i in range(acor.shape[0]):
            acor[i] = UtilMethods.autocorrelation(full_chain[:, :, i].T)
            print("\t %s: %0.3f" % (names[i], acor[i]))
        sys.stdout.flush()
        if nwalkers * ntotal > 200 * np.max(acor):
            # chain is long enough for convergence in each parameter
            print('Chain converged, stopping')
            continue_run = False
        else:
            print('Chain not yet converged, continuing')
        sys.stdout.flush()

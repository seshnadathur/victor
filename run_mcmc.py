from __future__ import print_function
import argparse
import sys
from multiprocessing import Pool
import emcee
import numpy as np
from posterior_samplers.CMASS_DR12 import CMASSPosterior
from utilities.utilities import UtilMethods
import os

parser = argparse.ArgumentParser(description='Run an MCMC sampling of the posterior for model fits '
                                             'to the void-galaxy cross-correlation function')

parser.add_argument('directory', help='path to directory with data files')
parser.add_argument('-a', '--action', default=1, type=int,
                    help='action=1 to estimate maximum likelihood point, =2 for MCMC sampling using emcee (default =1)')
parser.add_argument('-c', dest='chain', default=1, type=int,
                    help='chain number suffix to assign to MCMC chain output file (default =1)')
parser.add_argument('-t', '--threads', default=1, type=int,
                    help='number of threads available for parallelization (default =1)')
parser.add_argument('-C', '--covariance', default=0, type=int,
                    help='=0 for cosmology-dependent covariance, =1 for fixed covariance (default =0)')
parser.add_argument('-r', '--rmax', default=120., type=float,
                    help='maximum scale to use for fits in Mpc/h (default =120.0)')
parser.add_argument('-R', '--R_smooth', default=10.0, type=float,
                    help='smoothing scale for BOSS DR12 reconstruction, in Mpc/h; either 10 or 15 (default =10)')
parser.add_argument('-l', '--likelihood', default='SH',
                    help='likelihood calculation method; SH = Sellentin and Heavens, ' +
                         'H = Hartlap correction, G = standard Gaussian (default =SH)')
args = parser.parse_args()

# add trailing / to directory if not present
if not(args.directory[-1] == '/'):
    args.directory = args.directory + '/'

print('\nRunning on folder %s with the following options:' % args.directory)
print('\taction: %d' % args.action)
if args.action == 1:
    print('\tchain number: %d' % args.chain)
print('\tnumber of threads: %d' % args.threads)
print('\tuse cosmology-dependent covariance?: %s' % (not(bool(args.covariance))))
print('\trmax: %0.1f' % args.rmax)
print('\tR_smooth: %0.1f' % args.R_smooth)
print('\tlikelihood method: %s ' % args.likelihood)

if not(args.R_smooth==10.0):
    if args.R_smooth==15.0:
        print('Warning: using smoothing scale of 15 Mpc/h requires some approximation in the modelling')
    else:
        sys.exit('Unsupported option for smoothing scale, aborting.')

vg_fit = CMASSPosterior(args.directory, smooth=args.R_smooth, rmax=args.rmax, chain_number=args.chain,
                        like=args.likelihood, fixed_covmat=args.covariance, mono_from_mocks=True)


# define a global log posterior function to allow for emcee parallelization
def lnpost_global(theta):
    """
      Log posterior function: combines log likelihood and log prior

      :param theta: numpy array, vector position in parameter space
      :return:      log posterior
      """

    lp = vg_fit.lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnlkl = vg_fit.lnlike(theta)
    return lp + lnlkl


if args.action == 1:
    ml = vg_fit.find_max_like()
elif args.action == 2:
    # --- set options for using emcee --- #
    nwalkers = args.threads * 2      # number of walkers to use
    nstep = 500         # number of steps after which to check autocorrelation and write to file
    nburnin = 100       # number of burn-in steps to discard
    stop_factor = 0.02  # stop condition: integrated autocorrelation time / total chain length
    pos0 = [vg_fit.start_params + 1e-2 * np.random.randn(vg_fit.ndim) * vg_fit.scales for i in range(nwalkers)]

    # # apparently this is necessary to stop some problems with using Pool (?)
    # os.environ["OMP_NUM_THREADS"] = "1"

    # simplest parallelization option
    pool = Pool(args.threads)

    # initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, vg_fit.ndim, lnpost_global, pool=pool)

    # run the chains for nburnin burn-in steps (discarded)
    print("Running burn-in phase for %d steps" % nburnin)
    sys.stdout.flush()
    pos, prob, state = sampler.run_mcmc(pos0, nburnin)

    # reset the sampler state
    print("Done the burn-in, starting production run")
    sys.stdout.flush()
    sampler.reset()

    # now do the final production run
    icount = 0
    continue_run = True
    while continue_run:

        for istep, result in enumerate(sampler.sample(pos, iterations=nstep, storechain=False)):
            position = result[0]
            lnp = result[1]
            alpha_par = position[:, 3] * (position[:, 4] ** (-2. / 3))
            alpha_perp = position[:, 4] * alpha_par
            beta = position[:, 0] / position[:, 1]

            chain = np.zeros((position.shape[0], vg_fit.ndim + 1))
            chain[:, 0] = -1. * lnp
            chain[:, 1:] = position

            # at the end of nstep steps, write to file
            if icount == 0 and istep == 0:
                full_chain = chain
            else:
                full_chain = np.load(vg_fit.data_directory + 'MCMC_chains/' + vg_fit.mc_output_file, allow_pickle=True)
                full_chain = np.vstack([full_chain, chain])
            np.save(vg_fit.data_directory + 'MCMC_chains/' + vg_fit.mc_output_file, full_chain)

        icount += 1
        ntotal = icount * nstep

        # now calculate the autocorrelation time for chains produced so far, and decide whether to continue running
        # now calculate the integrated autocorrelation time for params and decide whether to proceed
        full_chain = np.load(vg_fit.data_directory + 'MCMC_chains/' + vg_fit.mc_output_file, allow_pickle=True)
        full_chain = full_chain[:, 1:].reshape((full_chain.shape[0] / nwalkers, nwalkers,
                                                full_chain.shape[1] - 1))
        print("After %d steps, autocorrelation times for parameters:" % ntotal)
        acor = np.empty(vg_fit.ndim)
        for i in range(acor.shape[0]):
            acor[i] = UtilMethods.autocorrelation(full_chain[:, :, i].T)
            print("\t %s: %0.3f" % (vg_fit.paramnames[i], acor[i]))
        sys.stdout.flush()
        if vg_fit.stop_factor * np.max(acor) < ntotal:
            # chain is long enough for convergence in each parameter
            print('Chain length exceeds %d x the maximum autocorrelation length for sampled params, so stopping'
                  % vg_fit.stop_factor)
            continue_run = False
        else:
            print('Chain length = %d, worst autocorrelation length = %d; continuing' % (ntotal, np.max(acor)))
        sys.stdout.flush()

    pool.terminate()
    print("Done production run")
else:
    print('Action option %d not understood, stopping' % args.action)

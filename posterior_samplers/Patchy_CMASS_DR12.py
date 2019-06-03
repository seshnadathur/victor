from __future__ import print_function
import sys
import os
import numpy as np
import random as rd
from utilities.utilities import UtilMethods
from scipy.integrate import quad
from scipy.optimize import basinhopping
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline


class PatchyCMASSPosterior:
    """
    Class to sample from the posterior for joint RSD-AP fits to BOSS DR12 CMASS void-galaxy correlation data
    """

    def __init__(self, data_directory, mock_index=0, rmax=120., chain_number=1, like='SH',
                 mono_from_mocks=True, fixed_covmat=False, quad_only=False, mono_only=False):
        """
        Initialize instance and read in necessary data from file

        :param data_directory:  path to folder with data files
        :param mock_index:      integer, optional
                                index of the Patchy mock sample to analyse (range 1-1000); mock_index=0 refers to the
                                mean of the 1000 mocks, with suitably rescaled covariance
        :param rmax:            float, optional
                                maximum void-galaxy separation to use if < 120 Mpc/h
        :param chain_number:    integer, optional
                                chain number
        :param like:            string, optional
                                'SH' = Sellentin & Heavens likelihood, 'H' = Hartlap correction, anything else = basic
                                Gaussian likelihood (default = 'SH')
        :param mono_from_mocks: bool, optional
                                whether to take reconstructed real-space monopole as average over mocks, or from data
                                default=True, i.e. use average over mocks
        :param fixed_covmat:    bool, optional
                                whether covariance matrix is fixed as beta varies, default=False
        :param quad_only:       bool, optional
                                whether to fit only to quadrupole, default=False
        :param mono_only:       bool, optional
                                whether to fit only to monopole, default=False
        """

        self.data_directory = data_directory
        self.mock_index = mock_index

        self.num_r_bins = 30    # fixed to the number of bins used in measurement of xi(r, mu) in the data
        self.num_mu_bins = 80   # fixed to the number of bins used in measurement of xi(r, mu) in the data
        mu_edges = np.linspace(0, 1., self.num_mu_bins + 1)
        self.mu_mid = 0.5 * (mu_edges[1:] + mu_edges[:-1])  # mid-point mu values for each bin

        # fiducial cosmology
        omega_m = 0.308
        eff_z = 0.57       # effective redshift of CMASS sample
        eofz = np.sqrt((omega_m * (1 + eff_z) ** 3 + 1 - omega_m))
        self.iaH = (1 + eff_z) / (100. * eofz)    # this sets the conversion from velocity dispersion to distance scales
        self.mock_b = 2.10      # nominal bias value for PATCHY mock galaxies (from Alam et al. 2017)

        # BigMD normalizing factor to scale the sigma8 value relative to that used in calibrating BigMD simulation
        growth = 0.76343  # growth factor at redshift 0.52 for the BigMD simulation
        self.s8norm = 0.8228 * growth  # sigma_8 value at redshift 0.52 for BigMD simulation

        # read the calibrated void matter density profile data from file and interpolate
        delta_fit_data = np.loadtxt(self.data_directory + 'CMASS_combined_Zobov_pVoids_Rcut_delta_profile.txt')
        self.r_for_delta = delta_fit_data[:, 0]   # separation bin mid-point values
        self.delta_r = InterpolatedUnivariateSpline(self.r_for_delta, delta_fit_data[:, 1], k=1, ext=3)

        # calculate cumulative density profile
        integral = np.zeros_like(self.r_for_delta)
        for i in range(len(integral)):
            integral[i] = quad(lambda x: self.delta_r(x) * x ** 2, 0, self.r_for_delta[i], full_output=1)[0]
        self.Delta_r = InterpolatedUnivariateSpline(self.r_for_delta, 3 * integral / self.r_for_delta ** 3, k=1, ext=3)

        # read the calibrated velocity dispersion profile data from file, normalize and interpolate
        sv_data = np.loadtxt(self.data_directory + 'CMASS_combined_Zobov_pVoids_Rcut_los_dispersion_profile.txt')
        # minimal filtering out of noise in measurement
        sv = savgol_filter(sv_data[:, 1] / sv_data[-1, 1], 3, 1)
        self.r_for_sv = sv_data[:, 0]
        self.sv_norm_func = InterpolatedUnivariateSpline(self.r_for_sv, sv, k=3, ext=3)

        # read the data covariance matrix computed from mocks
        self.fixed_covmat = fixed_covmat
        if self.fixed_covmat:
            self.covmat = np.load(self.data_directory + 'CMASS_combined_Zobov_pVoids_Rcut_x_sGals_covmat.npy',
                                  allow_pickle=True)
            self.nmocks = 2048  # sorry, hard-coded
        else:
            self.covmat = np.load(self.data_directory + 'CMASS_combined_Zobov_pVoids_Rcut_x_sGals_grid_covmat.npy',
                                  allow_pickle=True)
            self.nmocks = 1000  # that's just how it is

        # read in the multipole data on the grid
        self.f_grid, self.r_for_xi, avg_multipoles = np.load(self.data_directory +
                                                             'Patchy_CMASS_mean_combined_Zobov_pVoids_Rcut_' +
                                                             'multipoles_Rs10.0.npy', allow_pickle=True)
        self.s_multipoles = avg_multipoles[0]
        self.p_multipoles = avg_multipoles[1]
        if mock_index > 0:
            # this line will only work on Sciama because there is just too much data here (17G) for me to copy it
            # to any other location
            f_grid, r_for_xi, multipoles = np.load('/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_CCFs/' +
                                                   'multipoles_grid/Patchy_CMASS_DR12_i%04d_' % mock_index +
                                                   'combined_Zobov_pVoids_Rcut_multipoles_Rs10.0.npy'
                                                   )
            self.s_multipoles = multipoles[0]
            if not mono_from_mocks:
                self.p_multipoles = multipoles[1]
        else:
            # rescale covmat to account for smaller error in the mean of 1000 mocks
            self.covmat = self.covmat / 1000.

        # truncate r extent if necessary
        if self.r_for_xi[-1] > rmax:
            print('Using data multipoles out to maximum separation s=%0.1f Mpc/h only' % rmax)
            self.truncate = True
            self.max_r_index = np.where(self.r_for_xi > rmax)[0][0]
            if self.fixed_covmat:
                covmat = UtilMethods.truncate_covmat(self.covmat, self.max_r_index)
            else:
                covmat = UtilMethods.truncate_grid_covmat(self.covmat, self.max_r_index)
            self.covmat = covmat
            self.r_for_xi = self.r_for_xi[:self.max_r_index]
        else:
            self.truncate = False

        self.r_length = len(self.r_for_xi)     # the actual number of r bins to be used in comparison to data

        # invert the data covariance matrix
        if fixed_covmat:
            self.icov = np.linalg.inv(self.covmat)
            self.iqcov = np.linalg.inv(self.covmat[self.r_length:, self.r_length:])
            self.imcov = np.linalg.inv(self.covmat[:self.r_length, :self.r_length])
        else:
            self.icov = np.empty_like(self.covmat)
            for i, f in enumerate(self.f_grid):
                self.icov[i] = np.linalg.inv(self.covmat[i])
            self.iqcov = self.icov[:, self.r_length:, self.r_length:]
            self.imcov = self.icov[:, :self.r_length, :self.r_length]

        # name the output files: ridiculous number of options for historical reasons only
        self.quad_only = quad_only
        self.mono_only = mono_only
        if like == 'SH':
            self.like = 0
            likestring = 'SH_'
        elif like == 'H':
            self.like = 1
            likestring = 'Hartlap_'
            self.hartlap_alpha = (self.nmocks - 2 * self.r_length - 2.) / (self.nmocks - 1)
        else:
            self.like = 2
            likestring = ''     # basic Gaussian likelihood
        if mono_from_mocks:
            monostring = 'mockmono_'
        else:
            monostring = 'datamono_'
        if self.quad_only:
            multstring = 'r%0.1f_Qonly' % rmax
        elif self.mono_only:
            multstring = 'r%0.1f_Monly' % rmax
        else:
            multstring = 'r%0.1f' % rmax
        if self.fixed_covmat:
            covstring = 'fixedcov_'
        else:
            covstring = 'varycov_'
        # output chain file name
        if self.mock_index == 0:
            self.mc_output_file = 'Patchy_CMASS_mean_emcee%d_%s%s%s%s.npy' % (chain_number, likestring, monostring,
                                                                              covstring, multstring)
            self.ml_output_file = 'Patchy_CMASS_mean_ML_%s%s%s%s.txt' % (likestring, monostring, covstring, multstring)
        else:
            self.mc_output_file = 'Patchy_CMASS_i%04d_emcee%d_%s%s%s%s.npy' % (mock_index, chain_number, likestring,
                                                                               monostring, covstring, multstring)
            self.ml_output_file = 'Patchy_CMASS_i%04d_ML_%s%s%s%s.txt' % (mock_index, likestring, monostring,
                                                                          covstring, multstring)

        # check MCMC output directory exists and if not, create it
        if not os.access(self.data_directory + 'MCMC_chains/', os.F_OK):
            os.makedirs(self.data_directory + 'MCMC_chains/')

        # NOTE: for the MCMC we stick with the formulation with 5 free parameters (fs8, bs8, sigma_v, alpha, epsilon)
        # rather than 4 (fs8, bs8, sigma_v, epsilon) even though the data places no constraints on alpha, because it
        # helps get a nice representative spread of D_A and H values for pretty final posterior plots

        # --- set options for emcee sampling --- #
        # reasonable guess of maximum likelihood point
        self.start_params = np.array([0.5, 1.35, 390, 1.0, 1.0])
        # length of parameter vector (here we keep redundant parameter alpha for historical reasons)
        self.ndim = 5
        self.paramnames = ['fs8', 'bs8', 'sigma_v', 'alpha', 'epsilon']
        self.scales = [1, 1, 100, 1, 1]
        self.stop_factor = 50  # stop sampling when total chain length = (stop factor) x (integrated autocorr. time)
        # -------------------------------------- #

        # --- set options for maximum likelihood estimation --- #
        # tuned proposal covariance (based on knowledge from previous emcee run! Note alpha omitted) #
        self.prop_cov = np.array([[2.55508577e-03, 6.70245862e-03, 1.29768058e+00, -1.83918326e-04],
                                  [6.70245862e-03, 1.89098488e-02, 3.09356864e+00, -5.69374357e-04],
                                  [1.29768058e+00, 3.09356864e+00, 1.97912689e+03, -5.25130398e-02],
                                  [-1.83918326e-04, -5.69374357e-04, -5.25130398e-02, 1.11706355e-04]])
        # eigenvalues and eigenvectors of this proposal, used for taking steps in parameter space
        self.eigval, self.eigvec = np.linalg.eig(np.linalg.inv(self.prop_cov))
        self.reduced_ndim = len(self.eigval)    # Note alpha omitted
        # number of iterations to run basinhopping algorithm for
        self.niter = 10     # relatively small number fine for our well-behaved likelihood
        # ------------------------------------------------------------- #

    def find_max_like(self):
        """
        Method to estimate the maximum likelihood point in parameter space and write to file
        :return:  array_like, dimensions (self.reduced_ndim)
        """

        print("Running maximum likelihood estimator")
        sys.stdout.flush()
        # output destination
        if not os.access(self.data_directory + 'maxLikelihood/', os.F_OK):
            os.makedirs(self.data_directory + 'maxLikelihood/')
        # we fix alpha = 1.0 by hand because it does not affect the posterior at all
        start = np.array([self.start_params[0], self.start_params[1], self.start_params[2], self.start_params[4]])
        result = basinhopping(self.minus_lnpost, start,
                              minimizer_kwargs={"method": "BFGS", "options": {"maxiter": 200}, "tol": 1e-1},
                              niter=self.niter, take_step=self.takestep, disp=True)
        print(result.message)

        # calculate derived parameter values
        beta = result.x[0] / result.x[1]
        apar = 1.0 * (result.x[3]**(-2/3.))
        aperp = result.x[3] * apar

        # write result to file
        output_params = np.hstack([result.fun, result.x[0], result.x[1], result.x[2], 1.0,
                                   result.x[3], beta, aperp, apar]).reshape((1, 9))
        np.savetxt(self.data_directory + 'maxLikelihood/' + self.ml_output_file, output_params,
                   header='estimated ML point\nmax(-logpost) fsigma8 bsigma8 sigma_v alpha=1.0 eps beta a_perp a_par',
                   fmt='%0.4f')

        return output_params[1:6]

    def theory_multipoles(self, fs8, bs8, sigma_v, alpha_perp, alpha_par, s, mu, rescale_all=True):
        """
        Method to calculate the theoretical multipoles (monopole and quadrupole) of the redshift-space void-galaxy
        cross-correlation function for given model parameters

        :param fs8: growth rate times sigma_8 at the sample redshift
        :param bs8: galaxy bias times sigma_8 at the sample redshift
        :param sigma_v: asymptotic amplitude of velocity dispersion at large s
        :param alpha_perp: perpendicular AP parameter
        :param alpha_par: parallel AP parameter
        :param s: array_like, dimensions (N_s)
                  void-galaxy separation values at which multipole output is required
        :param mu: array_like
                   values of 0 < mu < 1 used for interpolation and integration of multipole moments
        :param rescale_all: bool, optional
                            historical settings parameter, =True to ensure correct rescaling of all input monopole
                            functions specified in the fiducial cosmology (default True)
        :return: array_like, dimensions (2*N_s)
                 theory multipoles: first N_s entries contain monopole at separations s, last N_s entries the quadrupole
        """

        multipoles = np.zeros(2 * len(s))
        xi_model = np.zeros(len(mu))
        true_mu = np.zeros(len(mu))
        scaled_fs8 = fs8 / self.s8norm
        beta = fs8 / bs8

        # interpolate on multipole grid to get the reconstructed real-space multipoles
        # NOTE: interpolation grid was constructed over parameters (f, b), but the result only depends on beta = f/b;
        # we therefore determine f from beta using fixed b=2.10
        data_p_multipoles = np.zeros(2 * self.num_r_bins)
        try:
            for i in range(len(data_p_multipoles)):
                interpolater = InterpolatedUnivariateSpline(self.f_grid, self.p_multipoles[:, i], k=3, ext=2)
                data_p_multipoles[i] = interpolater(beta * self.mock_b)
        except ValueError:
            # if the interpolation fails, return all theory multipoles as zero
            return np.zeros(len(s))
        # build an interpolating function for the real-space monopole at any separation
        xi_r_func = InterpolatedUnivariateSpline(self.r_for_xi, data_p_multipoles[:self.num_r_bins], k=3)

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 101)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_par) * np.sqrt(1. + (1. - mus ** 2) *
                                                                       (alpha_perp ** 2 / alpha_par ** 2 - 1)), mus)
        if rescaled_r[-1] < r[-1]:
            # hack: extend the range at least as far to avoid uncontrolled extrapolation later
            numpts = 0
            x = np.hstack([rescaled_r, np.linspace(rescaled_r[-1], r[-1], numpts + 1)[1:]])
            y1 = np.hstack([xi_r_func(r), xi_r_func(r[-1]) * np.ones(numpts)])
            y2 = np.hstack([self.delta_r(r), self.delta_r(r[-1] * np.ones(numpts))])
            y3 = np.hstack([self.Delta_r(r), self.Delta_r(r[-1]) * np.ones(numpts)])
            y4 = np.hstack([self.sv_norm_func(r), self.sv_norm_func(r[-1]) * np.ones(numpts)])
        else:
            x = rescaled_r
            y1 = xi_r_func(r)
            y2 = self.delta_r(r)
            y3 = self.Delta_r(r)
            y4 = self.sv_norm_func(r)
        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3)
        if rescale_all:
            # default: rescale all real-space monopoles calibrated to the fiducial cosmology, and correct amplitude
            # of the dispersion function
            rescaled_delta_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=3)
            rescaled_Delta_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=3)
            rescaled_sv_norm_func = InterpolatedUnivariateSpline(x, y4, k=3, ext=3)
            sigma_v = alpha_par * sigma_v
        else:
            # alternative for historical purposes: leave matter density and velocity dispersion profiles fixed
            rescaled_delta_r = InterpolatedUnivariateSpline(r, self.delta_r(r), k=3, ext=3)
            rescaled_Delta_r = InterpolatedUnivariateSpline(r, self.Delta_r(r), k=3, ext=3)
            rescaled_sv_norm_func = InterpolatedUnivariateSpline(r, self.sv_norm_func(r), k=3, ext=3)

        # calculate the redshift-space multipoles
        for i in range(len(s)):
            for j in range(len(mu)):

                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_par
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar + true_s * scaled_fs8 * rescaled_Delta_r(true_s) * true_mu[j] / 3.
                sy_central = sigma_v * rescaled_sv_norm_func(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-3 * sy_central, 3 * sy_central, 100)

                rpar = true_spar + true_s * scaled_fs8 * rescaled_Delta_r(true_s) * true_mu[j] / 3. - y
                rr = np.sqrt(true_sperp ** 2 + rpar ** 2)
                sy = sigma_v * rescaled_sv_norm_func(rr) * self.iaH

                integrand = (1 + rescaled_xi_r(rr)) * \
                            (1 + (scaled_fs8 * rescaled_Delta_r(rr) / 3. - y * true_mu[j] / rr) * (1 - true_mu[j]**2) +
                             scaled_fs8 * (rescaled_delta_r(rr) - 2 * rescaled_Delta_r(rr) / 3.) * true_mu[j]**2)
                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
                xi_model[j] = np.trapz(integrand, y) - 1

            # interpolating function for xi as a function of mu at each s
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)], xi_model[np.argsort(true_mu)], k=3)
            # monopole values
            multipoles[i] = quad(lambda xx: mufunc(xx), 0, 1, full_output=1)[0]
            # quadrupole values
            multipoles[i + len(s)] = quad(lambda xx: mufunc(xx) * 5 * (3 * xx ** 2 - 1) / 2., 0, 1, full_output=1)[0]

        return multipoles

    def lnlike(self, theta):
        """
        Log likelihood function

        :param theta:   array_like, dimensions (self.ndim)
                        vector position in parameter space
        :return: lnlkl: log likelihood value
        """

        fs8, bs8, sigmav, alpha, epsilon = theta
        beta = fs8 / bs8
        apar = alpha * epsilon**(-2./3)
        aperp = epsilon * apar

        # interpolate on multipole grids to get the redshift-space data multipoles
        data_s_multipoles = np.zeros(2 * self.num_r_bins)
        try:
            for i in range(len(data_s_multipoles)):
                interpolater = InterpolatedUnivariateSpline(self.f_grid, self.s_multipoles[:, i], k=3, ext=2)
                data_s_multipoles[i] = interpolater(beta * self.mock_b)
        except ValueError:
            # if the interpolation fails, return a negative infinite log likelihood
            return -np.inf

        # based on recon monopole, calculate the theory redshift-space multipoles on the fine grid
        theory = self.theory_multipoles(fs8, bs8, sigmav, aperp, apar, self.r_for_xi, self.mu_mid)
        if np.all(theory == 0):
            # this indicates an error (most likely out-of-bounds value of beta) at this point
            return -np.inf

        # resize theory and data vectors if necessary
        if self.truncate:
            theory = UtilMethods.resize_vector(theory, self.max_r_index)
            data_s_multipoles = UtilMethods.resize_vector(data_s_multipoles, self.max_r_index)

        # calculate the chi-squared value
        if self.fixed_covmat:
            if self.quad_only:
                chisq = np.dot(np.dot(theory[self.r_length:] - data_s_multipoles[self.r_length:], self.iqcov),
                               theory[self.r_length:] - data_s_multipoles[self.r_length:])
            elif self.mono_only:
                chisq = np.dot(np.dot(theory[:self.r_length] - data_s_multipoles[:self.r_length], self.imcov),
                               theory[:self.r_length] - data_s_multipoles[:self.r_length])
            else:
                chisq = np.dot(np.dot(theory - data_s_multipoles, self.icov), theory - data_s_multipoles)
        else:
            # interpolate to get the covariance matrix at this value of beta
            if self.quad_only:
                iq = np.empty_like(self.iqcov[0])
                for i in range(self.iqcov.shape[1]):
                    for j in range(self.iqcov.shape[2]):
                        interpolater = InterpolatedUnivariateSpline(self.f_grid, self.iqcov[:, i, j], k=3, ext=2)
                        iq[i, j] = interpolater(self.mock_b * beta)
                chisq = np.dot(np.dot(theory[self.r_length:] - data_s_multipoles[self.r_length:], iq),
                               theory[self.r_length:] - data_s_multipoles[self.r_length:])
            elif self.mono_only:
                im = np.empty_like(self.imcov[0])
                for i in range(self.imcov.shape[1]):
                    for j in range(self.imcov.shape[2]):
                        interpolater = InterpolatedUnivariateSpline(self.f_grid, self.imcov[:, i, j], k=3, ext=2)
                        im[i, j] = interpolater(self.mock_b * beta)
                chisq = np.dot(np.dot(theory[:self.r_length] - data_s_multipoles[:self.r_length], im),
                               theory[:self.r_length] - data_s_multipoles[:self.r_length])
            else:
                ic = np.empty_like(self.icov[0])
                for i in range(self.icov.shape[1]):
                    for j in range(self.icov.shape[2]):
                        interpolater = InterpolatedUnivariateSpline(self.f_grid, self.icov[:, i, j], k=3, ext=2)
                        ic[i, j] = interpolater(self.mock_b * beta)
                chisq = np.dot(np.dot(theory - data_s_multipoles, ic), theory - data_s_multipoles)

        if self.like == 0:
            # calculate log-likelihood using Sellentin & Heavens prescription
            lnlkl = - self.nmocks * np.log(1 + chisq / (self.nmocks - 1)) / 2.
        elif self.like == 1:
            # calculate log-likelihood with Hartlap factor added
            lnlkl = - 0.5 * self.hartlap_alpha * chisq
        else:
            # log-likelihood without propagating covariance matrix uncertainty
            lnlkl = - 0.5 * chisq

        return lnlkl

    def lnprior(self, theta):
        """
        Log prior function, to ensure proper priors

        :param theta: array_like, dimensions (self.ndim)
                      vector position in parameter space
        :return: log prior value
        """

        fs8, bs8, sigmav, alpha, epsilon = theta
        beta = fs8 / bs8

        # for now apply uninformative priors with a very (even unphysically) wide range; but, must also correctly
        # account for limited grid range in beta to avoid extrapolation errors
        if 0.1 < fs8 < 0.8 and 0.45 < bs8 < 1.9 and 0.163 < beta < 0.57 and 250 < sigmav < 500 \
                and 0.8 < alpha < 1.2 and 0.8 < epsilon < 1.2:
            return 0.0

        return -np.inf

    def lnpost(self, theta):
        """
        Log posterior function: combines log likelihood and log prior

        :param theta: array_like, dimensions (self.ndim)
                      vector position in parameter space
        :return: log posterior value
        """

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnlkl = self.lnlike(theta)
        return lp + lnlkl

    def minus_lnpost(self, trunc_theta):
        """
        Minus the log posterior, for maximum likelihood estimation via minimization

        :param trunc_theta: array_like, dimensions (self.reduced_ndim)
                            vector position in (fs8, bs8, sv, epsilon) parameter space, excluding alpha parameter
        :return: minus the log posterior
        """

        # NOTES: 1. Despite the name, we want to maximize the log posterior, not the log likelihood
        # 2. alpha is not a meaningful variable and was kept only for historical reasons, so dropped here

        return -1. * self.lnpost(np.array([trunc_theta[0], trunc_theta[1], trunc_theta[2], 1.0, trunc_theta[3]]))

    def takestep(self, params_vector):
        """
        Propose a step in (fs8, bs8, sv, epsilon) parameter space from the guessed starting point

        :param params_vector: array_like, dimensions (self.reduced_ndim)
                              vector position in (fs8, bs8, sv, epsilon) parameter space, excluding alpha parameter
        :return: array_like, new position in parameter space
        """

        steps = np.zeros_like(params_vector)
        for i, value in enumerate(params_vector):
            steps[i] = np.sqrt(1. / self.eigval[i] / self.reduced_ndim) * rd.gauss(0, 1)

        new_params_vector = params_vector + np.dot(self.eigvec, steps)

        return new_params_vector

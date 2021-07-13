import os
import sys
import functools
import numpy as np
import scipy.interpolate as si
from scipy.integrate import quad, simps
from scipy.signal import savgol_filter
from scipy.stats import norm
from models import ExcursionSetProfile
from models import EisensteinHu
from tools import multipoles, cosmology, utilities

_spline = si.InterpolatedUnivariateSpline

@functools.lru_cache(maxsize=10000)
def get_excursion_set_model(h, om, omb, mnu, ns, omk, z, use_EH, acc):
    return ExcursionSetProfile(h, om, omb, z=z, mnu=mnu, ns=ns, omega_k=omk,
                               use_eisenstein_hu=use_EH, camb_accuracy=acc)

class VoidGalaxyCCF:
    """
    Class to compute model predictions for the void-galaxy cross-correlation function and to evaluate the likelihood for fits to data
    """

    def __init__(self, paths, settings):
        """
        Initialize instance

        :param settings: Python dict containing input settings and settings
        """

        # effective redshift at which to calculate
        self.effective_z = settings['effective_redshift']

        # set the fiducial cosmology
        self.fiducial_omega_m = settings['fiducial_omega_m']
        cosmo = cosmology.Cosmology(omega_m=settings['fiducial_omega_m'], omega_l=settings['fiducial_omega_l'])
        self.iaH = (1 + self.effective_z) / cosmo.get_ez(self.effective_z)

        # was reconstruction used or not?
        self.use_recon = settings.get('data_uses_reconstruction', True)
        # If reconstruction was used, the redshift-space data vector depends on beta=f/b. Normally, so does the
        # real-space input. We assume they have been measured at several different values of beta and read in the
        # grid of beta values separately. The covariance matrix can also depend on beta, and can be provided on a beta
        # grid (which can be a different grid if desired). Override options to neglect the beta dependence of the
        # real-space input and the covariance matrix are provided to enable specific testing scenarios. If no
        # reconstruction was used, there is no beta dependence of any of these quantities.
        if self.use_recon:
            self.fixed_covmat = settings.get('fixed_covmat', True)
            self.fixed_real_input = settings.get('fixed_real_input', False)
            filename = paths['multipole_beta_grid_file']
            if filename.endswith('.npy'):
                self.beta_grid = np.load(filename, allow_pickle=True)
            else:
                self.beta_grid = np.loadtxt(filename)
            filename = paths.get('covmat_beta_grid_file', paths['multipole_beta_grid_file'])
            if filename.endswith('.npy'):
                self.covmat_beta_grid = np.load(filename, allow_pickle=True)
            else:
                self.covmat_beta_grid = np.loadtxt(filename)

        # ---- load the real-space input from file ----- #
        real_multipole_data = np.load(paths['realspace_multipole_file'], allow_pickle=True).item()
        # failsafe check for how the dict keys are named in this file
        got_it = False
        for poss in ['rvals', 'r', 'r_for_xi']:
            if poss in real_multipole_data:
                self.r_for_xi = real_multipole_data[poss]
                self.nrbins = len(self.r_for_xi)
                got_it = True
                pass
        if not got_it: raise ValueError('Could not find distance info in file %s. Aborting' % paths['realspace_multipole_file'])
        got_it = False
        for poss in ['multipoles', 'monopole']:
            if poss in real_multipole_data:
                self.multipoles_real = real_multipole_data[poss]
                got_it = True
                pass
        if not got_it: raise ValueError('Could not find x-corr multipole info in file %s. Aborting' % paths['realspace_multipole_file'])
        if self.use_recon and not self.fixed_real_input:
            # check the data loaded makes sense
            if not self.multipoles_real.shape[0] == len(self.beta_grid):
                sys.exit('Real-space input is not fixed and its shape does not match length of beta grid provided. Aborting')
            if not self.multipoles_real.shape[1] % self.nrbins == 0:
                sys.exit('Binning mismatch in real space input data. Aborting')
            # determine how many multipoles we are using
            self.real_multipole_number = int(self.multipoles_real.shape[1] / self.nrbins)
        else:
            # check the loaded data makes sense
            if not len(self.multipoles_real) % self.nrbins == 0:
                sys.exit('Binning mismatch in real space multipole data. Aborting')
            # determine how many multipoles we are using
            self.real_multipole_number = int(len(self.multipoles_real) / self.nrbins)

        # check whether a fit to data is desired or not
        self.fit_to_data = settings.get('fit_to_data', True)
        # If fit_to_data is False, redshift-space data vector and covariance matrix information is not loaded, and
        # need not be provided. This allows the user to simply use this class to calculate theory model predictions
        # even in the absence of data (which can be costly to obtain). NOTE: however, a real-space void-galaxy
        # correlation MUST be provided (as we can't calculate the theory without this)

        if self.fit_to_data:
            # ---- load the redshift-space data vector from file ---- #
            if not os.access(paths['redshiftspace_multipole_file'], os.F_OK):
                raise ValueError(f"File {paths['redshiftspace_multipole_file']} required for data fit, but not found.")
            red_multipole_data = np.load(paths['redshiftspace_multipole_file'], allow_pickle=True).item()
            # failsafe check for how the dict keys are named in this file
            got_it = False
            for poss in ['s', 's_for_xi', 'svals', 'rvals', 'r', 'r_for_xi']:
                if poss in red_multipole_data:
                    self.s_for_xi = red_multipole_data[poss]
                    self.nsbins = len(self.s_for_xi)
                    got_it = True
                    pass
            if not got_it: raise ValueError(f"Could not find distance info in file {paths['redshiftspace_multipole_file']}. Aborting")
            if 'multipoles' in red_multipole_data:
                self.multipoles_redshift = red_multipole_data['multipoles']
            else:
                raise ValueError(f"Could not find x-corr multipole info in file {paths['redshiftspace_multipole_file']}. Aborting")
            if self.use_recon:
                # check the data loaded makes sense
                if not self.multipoles_redshift.shape[0] == len(self.beta_grid):
                    sys.exit('use_recon=True but redshift-space data does not match length of beta grid provided. Aborting')
                if not self.multipoles_redshift.shape[1] % self.nsbins == 0:
                    sys.exit('Binning mismatch in redshift space multipole data. Aborting')
                # determine how many multipoles we are using
                self.multipole_number = int(self.multipoles_redshift.shape[1] / self.nsbins)
            else:
                # check the loaded data makes sense
                if not len(self.multipoles_redshift) % self.nsbins == 0:
                    sys.exit('Binning mismatch in redshift space multipole data. Aborting')
                # determine how many multipoles we are using
                self.multipole_number = int(len(self.multipoles_redshift) / self.nsbins)

            # ---- load the covariance matrix from file ---- #
            if not os.access(paths['covariance_matrix_file'], os.F_OK):
                raise ValueError(f"File {paths['covariance_matrix_file']} required for data fit, but not found.")
            self.covmat = np.load(paths['covariance_matrix_file'])
            num_entries = int(self.multipole_number * self.nsbins)
            if self.use_recon and not self.fixed_covmat:
                assert self.covmat.shape == ((len(self.covmat_beta_grid), num_entries, num_entries)), 'Unexpected shape of covariance matrix'
            else:
                assert self.covmat.shape == ((num_entries, num_entries)), 'Unexpected shape of covariance matrix'
            self.icovmat = np.linalg.inv(self.covmat)
        else:
            print('fit_to_data is False, so not loading redshift-space data or covariance matrix')
            # set the following to avoid errors in subsequent calls
            self.s_for_xi = self.r_for_xi
            self.multipole_number = self.real_multipole_number
            self.nsbins = self.nrbins

        # ---- check if a template file is provided for delta, load info if it is ---- #
        if 'delta_template_file' in paths:
            delta_template_data = np.load(paths['delta_template_file'], allow_pickle=True).item()
            got_it = False
            for poss in ['rvals', 'r', 'r_for_delta', 'r_for_xi']:
                if poss in delta_template_data:
                    r_for_delta = delta_template_data[poss]
                    got_it = True
                    pass
            if not got_it: raise ValueError('Could not find distance info in file %s. Check dict keys' % paths['delta_template_file'])
            got_it = False
            for poss in ['delta', 'monopole']:
                if poss in delta_template_data:
                    delta_vals = delta_template_data[poss]
                    got_it = True
                    pass
            if not got_it: raise ValueError('Could not find delta template info in file %s. Check dict keys' % paths['delta_template_file'])
            # build an interpolating function for later use
            self.delta_r = _spline(r_for_delta, delta_vals, ext=3)
            # integrate this for the cumulative density profile
            integral = np.zeros_like(r_for_delta)
            for i in range(len(integral)):
                # since we only do this once, it doesn't matter if we use the slower quad instead of trapz
                integral[i] = quad(lambda x: 3 * self.delta_r(x) * x**2 / r_for_delta[i]**3, 0, r_for_delta[i], full_output=1)[0]
            self.int_delta_r = _spline(r_for_delta, integral, ext=3)
        elif settings['delta_profile'] == 'use_template':
            raise ValueError('Delta profile option use_template selected but no delta template file is provided.')

        # ---- check if a template file is provided for the velocity dispersion profile, load info if it is ---- #
        if 'velocity_dispersion_template_file' in paths:
            sv_template_data = np.load(paths['velocity_dispersion_template_file'], allow_pickle=True).item()
            got_it = False
            for poss in ['rvals', 'r', 'r_for_sv', 'r_for_xi']:
                if poss in sv_template_data:
                    r_for_sv = sv_template_data[poss]
                    got_it = True
                    pass
            if not got_it: raise ValueError('Could not find distance info in file %s. Check dict keys' % paths['velocity_dispersion_template_file'])
            got_it = False
            for poss in ['sigma_v', 'sigma_v_los', 'dispersion', 'sigma']:
                if poss in sv_template_data:
                    sv_vals = sv_template_data[poss]
                    got_it = True
                    pass
            if not got_it:
                raise ValueError('Could not find velocity dispersion template info in file %s. Check dict keys' % paths['velocity_dispersion_template_file'])
            # build an interpolating function for later use
            normed_sv = savgol_filter(sv_vals / sv_vals[-1], 3, 1)  # slightly smooth as this data is usually noisy
            self.sv_norm_func = _spline(r_for_sv, normed_sv, ext=3)
        else:
            # default to a constant function (which may not be used, depending on model settings)
            self.sv_norm_func = _spline(self.r_for_xi, np.ones_like(self.r_for_xi), ext=3)

    def get_interpolated_multipoles(self, beta, redshift=True):
        """
        Return the interpolated multipoles at the specified value of beta=f/b
        """

        if redshift:
            return si.PchipInterpolator(self.beta_grid, self.multipoles_redshift, axis=0)(beta)
        else:
            return si.PchipInterpolator(self.beta_grid, self.multipoles_real, axis=0)(beta)

    def get_interpolated_covmat(self, beta):
        """
        Return the interpolated covariance matrix at the specified value of beta=f/b
        """

        # if the requested beta value falls outside the provided grid, return the covariance matrix fixed to
        # the value at the extreme edge of the grid (for a grid that is wide wrt the posterior in beta, this
        # will not affect science results but prevents the code from crashing if accidentally hitting a boundary)
        if beta < self.covmat_beta_grid[0]:
            return self.covmat[0]
        if beta > self.covmat_beta_grid[-1]:
            return self.covmat[-1]

        # else find bounding beta values in the grid
        lind = np.where(self.covmat_beta_grid < beta)[0][-1]
        hind = np.where(self.covmat_beta_grid >= beta)[0][0]
        # return the simple linear interpolation of the covmat measured at these two beta values
        # we use the arithmetic interpolation because this seems to perform better than geometric
        # interpolation, and is simpler and faster to implement
        t = (beta - self.covmat_beta_grid[lind]) / (self.covmat_beta_grid[hind] - self.covmat_beta_grid[lind])
        return (1 - t) * self.covmat[lind] + t * self.covmat[hind]

    def get_interpolated_precision(self, beta):
        """
        Return the interpolated inverse covariance (precision) matrix at the specified value of beta=f/b
        """

        # interpolation proceeds exactly as for the covariance matrix case above
        if beta < self.covmat_beta_grid[0]:
            return self.icovmat[0]
        if beta > self.covmat_beta_grid[-1]:
            return self.icovmat[-1]
        lind = np.where(self.covmat_beta_grid < beta)[0][-1]
        hind = np.where(self.covmat_beta_grid >= beta)[0][0]
        t = (beta - self.covmat_beta_grid[lind]) / (self.covmat_beta_grid[hind] - self.covmat_beta_grid[lind])
        return (1 - t) * self.icovmat[lind] + t * self.icovmat[hind]

    def correlation_matrix(self, beta):
        """
        Utility to compute the normalised correlation matrix at specified value of beta (for visualisation)
        """

        covmat = self.get_interpolated_covmat(beta)
        corrmat = np.zeros_like(covmat)
        diagonals = np.sqrt(np.diag(covmat))
        for i in range(corrmat.shape[0]):
            for j in range(corrmat.shape[1]):
                if not (diagonals[i] * diagonals[j] == 0):
                    corrmat[i, j] = covmat[i, j] / (diagonals[i] * diagonals[j])

        return corrmat

    def delta_fn(self, r, params, settings):
        """
        Void delta(r) profile, i.e. void-matter cross-correlation monopole
        """

        if settings['delta_profile'] == 'use_linear_bias':
            # in this case delta(r) is just the (real-space) xi(r) divided by linear bias
            # also in this case beta has to be a sampled parameter
            if 'beta' in params:
                beta = params['beta']
            else:
                raise ValueError('If use_linear_bias is True, beta must be a sampled parameter')
            if self.use_recon:
                real_multipoles = self.get_interpolated_multipoles(beta, redshift=False)
            else:
                real_multipoles = self.multipoles_real
            xir = _spline(self.r_for_xi, real_multipoles[:self.nrbins], ext=3)  # the real-space monopole
            return (xir(r) / params.get('bias', 2.0))
        elif settings['delta_profile'] == 'use_template':
            return self.delta_r(r)
        elif settings['delta_profile'] == 'use_excursion_model':
            # check that required minimum parameters are provided
            for chk in ['b10', 'b01', 'Rp', 'Rx']:
                if not chk in params: # parameter is missing
                    raise ValueError('Parameter %s required for delta model calculation but is not provided' % chk)
            om = params.get('Omega_m', self.fiducial_omega_m)
            h = params.get('H0', 67.5) / 100
            s80 = params.get('sigma_8_0', 0.81)
            omb = params.get('Omega_b', 0.048)
            deltac = params.get('delta_c', 1.686)
            ns = params.get('ns', 0.96)
            mnu = params.get('mnu', 0.06)
            omk = params.get('Omega_k', 0)
            use_EH = settings.get('use_eisenstein_hu', False)
            acc = settings.get('camb_accuracy', 1.0)
            esp = get_excursion_set_model(h, om, omb, mnu, ns, omk, self.effective_z, use_EH, acc)
            esp.set_normalisation(s80, z=0)
            # get the value of sigma8(z) to return as derived parameter
            self.s8z = esp.s8z_fiducial * np.sqrt(esp.normalisation)
            x = np.linspace(0.1, np.max(r))
            delta = esp.delta(x, params.get('b10'), params.get('b01'), params.get('Rp'), params.get('Rx'), self.effective_z, deltac=deltac)
            return delta(r)
        else:
            raise ValueError('Unrecognised choice of option delta_profile')


    def integrated_delta_fn(self, r, params, settings):
        """
        Void Delta(r) profile, i.e. integral of void-matter cross-correlation monopole or density contrast within sphere
        Also referred to as delta(<r) in some papers
        """

        if settings['delta_profile'] == 'use_linear_bias':
            # in this case delta(r) is just the (real-space) xi(r) divided by linear bias
            # also in this case beta has to be a sampled parameter
            if 'beta' in params:
                beta = params['beta']
            else:
                raise ValueError('If use_linear_bias is True, beta must be a sampled parameter')
            if self.use_recon:
                real_multipoles = self.get_interpolated_multipoles(beta, redshift=False)
            else:
                real_multipoles = self.multipoles_real
            xir = _spline(self.r_for_xi, real_multipoles[:self.nrbins], ext=3)  # the real-space monopole
            integral = np.zeros_like(r)
            npts = 100  # this is plenty to ensure the accuracy of trapz integration
            for i in range(len(r)):
                rarr = np.linspace(0, r[i], npts)
                integrand = xir(rarr) * rarr**2
                integral[i] = np.trapz(integrand, rarr)
                # trapz integration is much faster than quad
                # integral[i] = quad(lambda x: xir(x) * x**2, 0, r[i], full_output=1)[0]
            interpfn = _spline(r, (3 * integral / r**3) / params.get('bias', 2.0), ext=3)
            return interpfn(r)
        elif settings['delta_profile'] == 'use_template':
            return self.int_delta_r(r)
        elif settings['delta_profile'] == 'use_excursion_model':
            # check that required minimum parameters are provided
            for chk in ['b10', 'b01', 'Rp', 'Rx']:
                if not chk in params: # parameter is missing
                    raise ValueError('Parameter %s required for delta model calculation but is not provided' % chk)
            om = params.get('Omega_m', self.fiducial_omega_m)
            h = params.get('H0', 67.5) / 100
            s80 = params.get('sigma_8_0', 0.81)
            omb = params.get('Omega_b', 0.048)
            deltac = params.get('delta_c', 1.686)
            ns = params.get('ns', 0.96)
            mnu = params.get('mnu', 0.06)
            omk = params.get('Omega_k', 0)
            use_EH = settings.get('use_eisenstein_hu', False)
            acc = settings.get('camb_accuracy', 1.0)
            esp = get_excursion_set_model(h, om, omb, mnu, ns, omk, self.effective_z, use_EH, acc)
            esp.set_normalisation(s80, z=0)
            # get the value of sigma8(z) to return as derived parameter
            self.s8z = esp.s8z_fiducial * np.sqrt(esp.normalisation)
            x = np.linspace(0.1, np.max(r))
            rql, rqe, model = esp.eulerian_model_profiles(x, self.effective_z, params.get('b10'), params.get('b01'),
                                                          params.get('Rp'), params.get('Rx'), deltac=deltac)
            int_delta = _spline(rqe, model, ext=3)
            return int_delta(r)
        else:
            raise ValueError('Unrecognised choice of option delta_profile')


    def Delta_evolution(self, esp, b10, b01, Rp, Rx, deltac=1.686, pairwise=False):
        """
        Calculates an expression for (1/f)(dDelta/dlna) = 3(1+delta)v_r / (faHr) in the excursion set model
        """

        x = np.linspace(0.1, np.max(self.r_for_xi))
        RqE, dSph = esp.eulerian_1halo(x, self.effective_z, b10, b01, Rp, Rx, deltac)
        RqE = RqE[0]; dSph = dSph[0]
        # check for NaNs in RqE and remove if necessary
        valid = np.logical_not(np.isnan(RqE))
        RqE = RqE[valid]; dSph = dSph[valid]
        dSph = _spline(RqE, dSph)
        dSph_deriv = _spline(RqE, np.gradient(dSph(RqE), RqE))

        delta2 = np.zeros_like(RqE)
        for i, re in enumerate(RqE):
            delta2[i] = esp.growth_factor(self.effective_z) * esp.eulerian_2halo(re, Rp, Rx)

        DeltaL = esp.lagrangian_profile(x, b10, b01, Rp, Rx)

        if pairwise:
            # return expression for pairwise void-galaxy motion, ie including contribution for void motion
            # we never really want this option for void-galaxy ccf studies; included for completeness only
            # (deltac * (1 + dSph(RqE) + RqE * dSph_deriv(RqE) / 3) * ((1 + dSph(RqE))**(1 / deltac) - 1) + 2 * esp.growth_factor(z) * delta2) / (1 + delta(RqE))
            term = deltac * (1 + dSph(RqE) + RqE * dSph_deriv(RqE) / 3) * ((1 + dSph(RqE))**(1 / deltac) - 1) + 2 * esp.growth_factor(self.effective_z) * delta2
        else:
            # return expression for galaxy velocities around voids, not including void motion term
            # for consistency of RSD modelling, we always want to use this option
            term = deltac * (1 + dSph(RqE) + RqE * dSph_deriv(RqE) / 3) * ((1 + dSph(RqE))**(1 / deltac) - 1) + esp.growth_factor(self.effective_z) * delta2

        interpfn = _spline(RqE, term)
        return interpfn


    def velocity_terms(self, r, rescaled_r, params, settings):
        """
        Calculates the radial component v_r(r) of the galaxy velocities around voids, and its derivative with respect to r,
        for the given model choices
        """

        # first we get Delta(r) and delta(r)
        int_dr = self.integrated_delta_fn(self.r_for_xi, params, settings)
        if settings['delta_profile'] in ['use_linear_bias', 'use_template']:
            # rescale this Delta(r) function to account for AP dilation
            int_delta = _spline(rescaled_r, int_dr, ext=3)
            # also get the delta(r) function and rescale that as well
            dr = self.delta_fn(self.r_for_xi, params, settings)
            delta = _spline(rescaled_r, dr, ext=3)
        elif settings['delta_profile'] == 'use_excursion_model':
            # get the delta(r) by numerical differentiation of Delta(r) (faster than full model call)
            dr = int_dr + self.r_for_xi * np.gradient(int_dr, self.r_for_xi) / 3
            # and don't rescale either function
            int_delta = _spline(self.r_for_xi, int_dr, ext=3)
            delta = _spline(self.r_for_xi, dr, ext=3)
        else:
            raise ValueError('Unrecognised choice of option delta_profile')

        # calculate a multiplicative growth factor for later use, value depending on model choices
        if settings['delta_profile'] == 'use_linear_bias':
            # we are assuming f*Delta(r) = beta*xibar(r), and int_delta was calculated above as
            # xibar(r)/b, so we need to multiply by beta*b
            growth_term = params.get('beta') * params.get('bias', 2.0)
        elif settings['delta_profile'] == 'use_template':
            # int_delta was calculated above using the template, and needs to be scaled by a factor
            # of sigma8(z) / sigma8_template(z_template)
            if not 'template_sigma8' in settings:
                raise ValueError('template_sigma8 must be provided in settings to use delta template')
            growth_term = params['fsigma8'] / settings['template_sigma8']
        elif settings['delta_profile'] == 'use_excursion_model':
            # no special correction needed
            growth_term = params['f']

        # now calculate v_r depending on the model choices
        model = settings.get('velocity_model', 'linear')
        empirical_corr = settings.get('empirical_velocity_correction', False)
        Avel = params.get('Avel', 0)
        r_for_int = np.linspace(0.1, np.max(self.r_for_xi), 100)  # to get finer grid on which to calculate numerical derivatives
        if model == 'linear':
            if not empirical_corr:
                # simplest linearised form of continuity equation, v_r = -faHrDelta(r)/3
                v_r = -growth_term * r * int_delta(r) / (3 * self.iaH)
                # in this case we can calculate the derivative using simple analytic expression
                v_r_prime = -growth_term * (delta(r) - 2 * int_delta(r) / 3) / self.iaH
            else:
                # apply empirical correction factor of (1 + A * delta(r)) to above
                v_r = -growth_term * r * int_delta(r) * (1 + Avel * delta(r)) / (3 * self.iaH)
                # calculate the derivative numerically
                vr_int = -growth_term * r_for_int * int_delta(r_for_int) * (1 + Avel * delta(r_for_int)) / (3 * self.iaH)
                vrp = _spline(r_for_int, np.gradient(vr_int, r_for_int), ext=3)
                v_r_prime = vrp(r)
        elif model == 'part_linear':
            # NOTE/TODO: here there is an inconsistency combined with use_linear_bias or use_template options which needs fixing
            # for use_linear_bias, we have calculated delta(r) = xi(r) / b for fixed b=2 by default
            # for use_template, we have delta_template(r), which needs to be multiplied by sigma_8(z) / template_sigma8
            # but we don't currently sample over b or sigma_8(z) for either of these cases, so the normalisations are then not correct
            # if using the empirical correction this probably cancels out ... but a proper fix is needed!
            if not empirical_corr:
                # hybrid form of continuity equation; v_r = -aHr(dDelta/dlna)/(3(1+delta(r))) as for full
                # non-linear form, but assumes (dDelta/dlna) = fDelta(r) as for linear evolution behaviour
                v_r = -growth_term * r * int_delta(r) / (3 * self.iaH * (1 + delta(r)))
                # numerical derivative
                vr_int = -growth_term * r_for_int * int_delta(r_for_int) / (3 * self.iaH * (1 + delta(r_for_int)))
                vrp = _spline(r_for_int, np.gradient(vr_int, r_for_int), ext=3)
                v_r_prime = vrp(r)
            else:
                # hybrid form as above but including (1 + A*delta(r)) empirical correction factor
                v_r = -growth_term * r * int_delta(r) * (1 + Avel * delta(r)) / (3 * self.iaH * (1 + delta(r)))
                # numerical derivative
                vr_int = -growth_term * r_for_int * int_delta(r_for_int) * (1 + Avel * delta(r_for_int))/ (3 * self.iaH * (1 + delta(r_for_int)))
                vrp = _spline(r_for_int, np.gradient(vr_int, r_for_int), ext=3)
                v_r_prime = vrp(r)
        elif model == 'non_linear':
            # first step is to calculate non-linear expression for (dDelta/dlna)
            # this can only be done in the excursion set model
            if not settings['delta_profile'] == 'use_excursion_model':
                err = 'Incompatible options: non-linear velocity model can only be used with excursion model profile'
                raise ValueError(err)
            # cset up instance of the excursion set model class
            om = params.get('Omega_m', self.fiducial_omega_m)
            h = params.get('H0', 67.5) / 100
            s80 = params.get('sigma_8_0', 0.81)
            omb = params.get('Omega_b', 0.048)
            deltac = params.get('delta_c', 1.686)
            ns = params.get('ns', 0.96)
            mnu = params.get('mnu', 0.06)
            omk = params.get('Omega_k', 0)
            use_EH = settings.get('use_eisenstein_hu', False)
            acc = settings.get('camb_accuracy', 1.0)
            esp = get_excursion_set_model(h, om, omb, mnu, ns, omk, self.effective_z, use_EH, acc)
            esp.set_normalisation(s80, z=0)
            # get the function for (dDelta/dlna)
            nonlin_evolution = self.Delta_evolution(esp, params.get('b10'), params.get('b01'), params.get('Rp'),
                                                    params.get('Rx'), deltac=deltac, pairwise=False)
            if not empirical_corr:
                # fully consistent non-linear form of continuity equation, v_r = -aHr(dDelta/dlna)/(3(1+delta(r)))
                v_r = -params.get('f') * r * nonlin_evolution(r) / (3 * self.iaH * (1 + delta(r)))
                # and again we evaluate the derivative numerically
                vr_int = -params.get('f') * r_for_int * nonlin_evolution(r_for_int) / (3 * self.iaH * (1 + delta(r_for_int)))
                vrp = _spline(r_for_int, np.gradient(vr_int, r_for_int), ext=3)
                v_r_prime = vrp(r)
            else:
                # non-linear expression as above, but including (1 + A*delta(r)) empirical correction factor
                v_r = -params.get('f') * r * nonlin_evolution(r) * (1 + Avel * delta(r))/ (3 * self.iaH * (1 + delta(r)))
                # and again we evaluate the derivative numerically
                vr_int = -params.get('f') * r_for_int * nonlin_evolution(r_for_int) * (1 + Avel * delta(r_for_int)) / (3 * self.iaH * (1 + delta(r_for_int)))
                vrp = _spline(r_for_int, np.gradient(vr_int, r_for_int), ext=3)
                v_r_prime = vrp(r)

        return v_r, v_r_prime

    def theory_xi(self, s, mu, params, settings):
        """
        Calculates the model prediction for the anisotropic redshift-space cross-correlation xi(s, mu)
        """

        beta = params.get('beta')
        epsilon = params.get('epsilon', params['aperp'] / params['apar'])

        if self.use_recon:
            real_multipoles = self.get_interpolated_multipoles(beta, redshift=False)
        else:
            real_multipoles = self.multipoles_real

        # apply Alcock-Paczynski correction by rescaling the radial coordinate
        mu_vals = np.linspace(1e-8, 1)
        mu_integral = np.trapz(params.get('apar', 1.0) * np.sqrt(1 + (1 - mu_vals**2) * (epsilon**2 - 1)), mu_vals)
        ref_r = self.r_for_xi
        rescaled_r = ref_r * mu_integral  # this is now r in the true cosmology

        # rescale the input real-space xi(r) function accordingly
        rescaled_xi_r = _spline(rescaled_r, real_multipoles[:self.nrbins], ext=3)
        # build interpolating function for the velocity velocity terms
        x = self.r_for_xi #np.linspace(0.1, 1.1 * np.max(self.r_for_xi))
        v_r, v_r_prime = self.velocity_terms(x, rescaled_r, params, settings)
        v_r_interp = _spline(x, v_r, ext=3)
        v_r_p_interp  = _spline(x, v_r_prime, ext=3)

        if settings['model'] in ['dispersion', 'streaming']:
            # so far only coded a template approach to the dispersion profile sigma_v(r) so this is always rescaled
            rescaled_sv_norm_func = _spline(rescaled_r, self.sv_norm_func(ref_r), ext=3)
            sv_grad = _spline(rescaled_r, np.gradient(rescaled_sv_norm_func(rescaled_r), rescaled_r), ext=3)
            sigma_v = params.get('sigma_v', 380) * params.get('apar', 1.0)

            y = np.linspace(-5, 5) * 500 * self.iaH  # use 5x and 500 to make sure range is wide enough in all cases
            if np.ndim(s) == 2 and np.ndim(mu) == 2:
                if not s.shape == mu.shape:
                    raise ValueError('If arguments s and mu are 2D arrays they must have the same shape')
                S, Mu, Y = np.meshgrid(s[0], mu[:, 0], y)
            elif np.ndim(s) == 1 and np.ndim(mu) == 1:
                S, Mu, Y = np.meshgrid(s, mu, y)
            else:
                raise ValueError('Arguments s and mu must have the same dimensions')
        else:
            if np.ndim(s) == 2 and np.ndim(mu) == 2:
                if not s.shape == mu.shape:
                    raise ValueError('If arguments s and mu are 2D arrays they must have the same shape')
                S, Mu = s, mu
            elif np.ndim(s) == 1 and np.ndim(mu) == 1:
                S, Mu = np.meshgrid(s, mu)
            else:
                raise ValueError('Arguments s and mu must have the same dimensions')

        # apply AP corrections to get redshift-space coordinates (true_s, true_mu) in the true
        # cosmology from (s, mu) in fiducial cosmology
        true_mu = Mu
        true_mu[Mu>0] = 1 / np.sqrt(1 + epsilon**2 * (1 / Mu[Mu>0]**2 - 1))
        true_sperp = S * np.sqrt(1 - true_mu**2) * params.get('aperp', 1.0)
        true_spar = S * true_mu * params.get('apar', 1.0)
        true_s = np.sqrt(true_spar**2 + true_sperp**2)

        # now implement coordinate transformation from redshift-space to real-space coordinates
        if not settings.get('do_coord_shift', True):
            # some papers do not account for the coordinate shift correctly and thus calculate
            # the RSD model as xi(r, mu_r) instead of xi(s, mu) – this option allows to reproduce
            # the results in those papers; however remember this is actually not correct!
            r_par = true_spar
        else:  # default case, using iterative solver to do the transformation
            # M is an optional nuisance parameter introduced by Hamaus et al 2020
            M = params.get('M', 1.0) if settings['model'] == 'Kaiser' else 1.0
            r_par = true_spar / (1 + M * self.iaH * v_r_interp(true_s) / true_s)
            for i in range(settings.get('niter', 5)):
                r = np.sqrt(true_sperp**2 + r_par**2)
                r_par = true_spar / (1 + M * self.iaH * v_r_interp(r) / r)
        # r_par is now the real-space l-o-s pair separation calculated using the mean coherent outflow velocity
        r = np.sqrt(true_sperp**2 + r_par**2)

        # recalculate the real-space mu from redshift-space value
        true_mu_r = r_par / r

        if settings['model'] == 'dispersion':  # implement the velocity dispersion model of Nadathur & Percival 2019

            # convert variable of integration from v to y=v/aH
            sigma_y = sigma_v * rescaled_sv_norm_func(r) * self.iaH
            y = sigma_y * Y
            # Note: we integrate over the range [-X, X] where X = settings['yrange'] * sigma_y

            # rpar is the real-space l-o-s coordinate calculated using actual outflow velocity at
            # each value of the integration variable y=v/aH
            rpar = r_par - y
            r = np.sqrt(true_sperp**2 + rpar**2)

            sy = sigma_v * rescaled_sv_norm_func(r) * self.iaH  # recalculate dispersion at each point
            dy = sigma_v * sv_grad(r) * self.iaH  # calculate derivative of the velocity dispersion term

            # vr_term corresponds to v_r/raH where v_r is radial outflow plus component of random dispersion velocity
            # vr_prime_term corresponds to 1/aH times derivative wrt r of v_r
            vr_term = self.iaH * v_r_interp(r) / r + y * true_mu_r / r
            vr_prime_term = self.iaH * v_r_p_interp(r) + (dy / r) * true_mu_r
            integrand = (1 + rescaled_xi_r(r)) * (1 + vr_term + (vr_prime_term - vr_term) * true_mu_r**2)**(-1) * \
                        np.exp(-0.5 * (y / sy)**2) / (sy * np.sqrt(2 * np.pi))
            # the resulting model prediction for xi(s, mu) from integration
            xi_smu = np.trapz(integrand, x=y, axis=2) - 1

        elif settings['model'] == 'streaming':  # implement a Gaussian streaming model

            # in this case we recalculate r_par as the coherent mean value is not required
            r_par = true_spar - Y * sigma_v * rescaled_sv_norm_func(r) * self.iaH
            # iterate a few times to ensure convergence
            for i in range(settings.get('niter', 5)):
                r = np.sqrt(true_sperp**2 + r_par**2)
                r_par = true_spar - Y * sigma_v * rescaled_sv_norm_func(r) * self.iaH
            r = np.sqrt(true_sperp**2 + r_par**2)
            true_mu_r = r_par / r
            sv = sigma_v * rescaled_sv_norm_func(r)

            v = Y * sv
            vel_pdf = norm.pdf(v, loc=v_r_interp(r) * true_mu_r, scale=sv)

            integrand = (1 + rescaled_xi_r(r)) * vel_pdf
            xi_smu = simps(integrand, x=v, axis=2) - 1

        elif settings['model'] == 'Kaiser':  # implement simple Kaiser RSD model
            # check if additional nuisance parameters are included or not as per Hamaus et al 2020
            M = params.get('M', 1.0)
            Q = params.get('Q', 1.0)

            if settings.get('approx_Kaiser', False):
                # this option uses a (poor) approximation to the true expression based on a
                # series expansion truncated at linear order
                xi_smu = M * (rescaled_xi_r(r) - self.iaH * v_r_interp(r) / r - \
                              Q * true_mu_r**2 * self.iaH * (v_r_p_interp(r) - v_r_interp(r) / r))
            else:
                # directly evaluate full expression without approximation
                xi_smu = (1 + M * rescaled_xi_r(r)) / (1 + self.iaH * v_r_interp(r) / r + \
                                                       Q * true_mu_r**2 * self.iaH * (v_r_p_interp(r) - v_r_interp(r) / r)) - 1

        return xi_smu


    def theory_multipoles(self, s, params, settings):
        """
        Calculate Legendre multipole compression of the model xi(s, mu)

        Returns monopole, quadrupole and hexadecapole moments
        """

        mu = np.linspace(0, 1)
        S, Mu = np.meshgrid(s, mu)

        xi_smu = self.theory_xi(S, Mu, params, settings)
        xi_model = si.interp2d(s, mu, xi_smu, kind='cubic')
        monopole, quadrupole, hexadecapole = multipoles.multipoles_singleshot(xi_model, s)
        # monopole = multipoles.multipoles(xi_model, s, ell=0)
        # quadrupole = multipoles.multipoles(xi_model, s, ell=2)
        # hexadecapole = multipoles.multipoles(xi_model, s, ell=4)

        return monopole, quadrupole, hexadecapole


    def chi_squared_multipoles(self, params, settings):
        """
        Return the chi-square for a given point in parameter space

        Assumes comparison of theory and data vector takes place via compression to Legendre multipoles

        Additionally returns the covariance matrix used for this calculation (to save repeated calculation
        in the log likelihood step)
        """

        if self.fit_to_data:
            # get the theory
            monopole, quadrupole, hexadecapole = self.theory_multipoles(self.s_for_xi, params, settings)
            if self.multipole_number == 1:
                theoryvec = monopole
            elif self.multipole_number == 2:
                theoryvec = np.hstack([monopole, quadrupole])
            else:  # hard coded maximum of l=0,2,4
                theoryvec = np.hstack([monopole, quadrupole, hexadecapole])

            # now get the data vector and inverse covariance matrix
            if 'beta' in params:
                beta = params['beta']
            else:
                raise ValueError('Missing necessary input parameter beta')
            if self.use_recon:
                datavec = self.get_interpolated_multipoles(beta, redshift=True)
            else:
                datavec = self.multipoles_redshift
            if self.use_recon and not self.fixed_covmat:
                cov = self.get_interpolated_covmat(beta)
                icov = self.get_interpolated_precision(beta)
            else:
                cov = self.covmat
                icov = self.icovmat

            # calculate the chi square and covariance matrix
            return np.dot(np.dot(theoryvec - datavec, icov), theoryvec - datavec), cov
        else:
            print('fit_to_data is False, cannot calculate chi-squared?!')
            return 0, 0

    def lnlike_multipoles(self, params, settings):
        """
        Log likelihood function (for case of compression to Legendre multipoles)
        """

        if self.fit_to_data:
            # first just get the chi-square value
            chisq, cov = self.chi_squared_multipoles(params, settings)

            # now apply appropriate conversion to get the log likelihood from this
            like_factor = 0
            if not self.fixed_covmat:
                # covariance matrix itself varies with beta, so need to normalise for change
                determinant = np.linalg.slogdet(cov)
                if not determinant[0] == 1:
                    # something has gone dramatically wrong!
                    print('Likelihood evaluation failed, bad covariance matrix?')
                    return -np.inf, np.inf
                like_factor = -0.5 * determinant[1]
            if 'Sellentin' in settings['likelihood_type']:
                # use the approach of Sellentin & Heavens 2016 to correctly propagate the uncertainty in the
                # covariance matrix estimation to the likelihood
                nmocks = settings['likelihood_type']['Sellentin']['nmocks']
                lnlike = -nmocks * np.log(1 +  chisq / (nmocks - 1)) / 2 + like_factor
            elif 'Hartlap' in settings['likelihood_type']:
                # use the Hartlap correction factor to approximately account for uncertainty in estimation of
                # the covariance matrix
                p = self.multipole_number * self.nrbins  # number of bins in data vector
                nmocks = settings['likelihood_type']['Hartlap']['nmocks']
                a = (nmocks - p - 2) / (nmocks - 1)
                lnlike = -0.5 * chisq * a + like_factor
            else:
                # not applying any correction – if your covariance matrix is estimated this is wrong!
                lnlike = -0.5 * chisq + like_factor

            # add a sanity check to catch cases which fail due to unforeseen errors (e.g. -ve chisq, ...)
            if np.isnan(lnlike):
                print(f'Likelihood evaluation failed, returning NaN. Parameters at fail point: {params}')
                print(f'Chisq: {chisq}, like_factor: {like_factor}')
                lnlike = -np.inf, np.inf

        else:
            print('fit_to_data is False, cannot calculate lnlike?!')
            lnlike = 0

        return lnlike, chisq

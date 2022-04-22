import os
import sys
import functools
import copy
import numpy as np
import scipy.interpolate as si
from scipy.integrate import quad, simps
from scipy.stats import norm
from scipy.special import legendre
from . import utils
from .utils import InputError
from .excursion_set_profile import ExcursionSetProfile
from .cosmology import BackgroundCosmology

_spline = si.InterpolatedUnivariateSpline

@functools.lru_cache(maxsize=10000)
def get_excursion_set_model(h, om, omb, mnu, ns, omk, z, use_EH, acc):
    return ExcursionSetProfile(h, om, omb, z=z, mnu=mnu, ns=ns, omega_k=omk,
                               use_eisenstein_hu=use_EH, camb_accuracy=acc)

class CCFModel:
    """
    Class to perform CCF model calculations.

    The following methods take as input the cross-correlation between centres of density (most commonly voids,
    but could also be *density-split* centres) and galaxies or matter tracers in real space, and calculate the
    corresponding cross-correlation function in redshift space.
    """

    def __init__(self, model):
        """
        Initialize :class:`CCFModel`

        Parameters
        --------
        model : dict
            Dictionary of options for model evaluation
        """

        self.z_eff = model.get('z_eff', InputError('Effective redshift z_eff must be provided'))
        cosmo = BackgroundCosmology(model.get('cosmology'))
        self.iaH = (1 + self.z_eff) / (100 * cosmo.Ez(self.z_eff))

        self._load_realspace_ccf(model)
        self.matter_model = model['matter_ccf'].get('matter_model', 'linear_bias')
        if self.matter_model == 'template':
            self._set_matter_ccf_template(model)
        self._set_velocity_pdf(model)
        # set of model options: these settings provide the default to be used in automated evaluations
        # (eg when running chains), but can be overridden by the user passing an optional dict at the
        # time of model evaluation (eg when debugging or plotting)
        self.model = {'cosmology': model.get('cosmology'),
                      'rsd_model': model.get('rsd_model', 'streaming'),
                      'assume_isotropic_realspace': model['realspace_ccf'].get('assume_isotropic', 'True'),
                      'matter_model': self.matter_model,
                      'excursion_set_options': model['matter_ccf'].get('excursion_set_options', None),
                      'bias': model['matter_ccf'].get('bias', 1.9),
                      'mean_model': model['matter_ccf'].get('mean_model', 'linear'),
                      'pdf_form': model['velocity_pdf'].get('form', 'gaussian'),
                      'empirical_corr': model['velocity_pdf'].get('empirical_corr', False)
                     }

    def _load_realspace_ccf(self, model):
        """"""

        if model.get('reconstruction', True):
            self.fixed_real_input = False
            if isinstance(model['realspace_ccf']['beta_grid'], str):
                beta_file = os.path.join(model.get('dir',''), model['realspace_ccf']['beta_grid'])
                self.beta = utils.read_beta_file(beta_file)
            else:
                self.beta = np.array(model['realspace_ccf']['beta_grid'])
            if not np.all(self.beta[1:] - self.beta[:-1] > 0):
                raise InputError('Realspace beta grid must be stricly monotonically increasing')
        else:
            self.fixed_real_input = True

        data_file = os.path.join(model.get('dir', ''), model['realspace_ccf']['data_file'])
        format = model['realspace_ccf'].get('format', 'multipoles')
        if format == 'multipoles':
            self.r, self.poles_r, self.real_multipoles = utils.read_cf_multipoles_file(data_file)
        elif format == 'rmu':
            # NOTE: I can't imagine a situation where the realspace ccf would be provided as a fn of (r, mu)
            # but the user would still want to assume it to be isotropic, so have not coded for this
            self.r, mu, real_ccf = utils.read_cf_2d_file(data_file)
            self.poles_r = [0, 2, 4] # do not provide the functionality to deal with odd multipoles!
            if self.fixed_real_input:
                self.real_multipoles = utils.multipoles_from_fn(real_ccf, self.r, ell=self.poles_r)
            else:
                self.real_multipoles = {}
                for ell in self.poles_r:
                    self.real_multipoles[f'{ell}'] = np.zeros((len(self.beta), len(self.r)))
                for i in range(len(self.beta)):
                    tmp_mults = utils.multipoles_from_fn(real_ccf[i], self.r, ell=self.poles_r)
                    for ell in self.poles_r:
                        self.real_multipoles[f'{ell}'][i] = tmp_mults[f'{ell}']
        else:
            raise InputError(f"Unrecognised format '{format}' for realspace ccf, options are 'multipoles' or 'rmu'")

    def _set_matter_ccf_template(self, model):
        """"""
        matter_ccf = model['matter_ccf']
        template_file = os.path.join(model.get('dir', ''), matter_ccf.get('template_file', ''))
        r_for_delta, delta = utils.read_matter_ccf_template_file(template_file)
        integrated = matter_ccf.get('integrated', 'False')
        if not 'template_sigma8' in matter_ccf:
            raise InputError('When using template model for the matter ccf, template_sigma8 must be provided')
        self.template_sigma8 = matter_ccf['template_sigma8']

        # TODO: any additions to deal with a template that is not spherically symmetric should be added here
        if not len(r_for_delta) == len(delta):
             raise InputError('Binning mismatch in matter template (asymmetric templates not yet allowed)')

        r = np.linspace(r_for_delta.min(), r_for_delta.max())
        if integrated:
            self.integrated_delta = _spline(r_for_delta, delta, ext=3)
            derivative = np.gradient(self.integrated_delta(r), r)
            self.delta = _spline(r, self.integrated_delta(r) + r * derivative / 3, ext=3)
        else:
            self.delta = _spline(r_for_delta, delta, ext=3)
            integral = np.zeros_like(r)
            for i in range(len(integral)): # as we only need to do this once, use the slower quad routine
                integral[i] = quad(lambda x: 3 * self.delta(x) * x**2 / r[i]**3, 0, r[i], full_output=1)[0]
            self.integrated_delta = _spline(r, integral)

    def _set_velocity_pdf(self, model):
        """"""
        velocity_pdf = model['velocity_pdf']
        mean_model = velocity_pdf.get('mean_model', 'linear')
        self.has_velocity_template = False
        if mean_model == 'template':
            # read in a template: this option only used for specific testing
            template_file = os.path.join(model.get('dir', ''), velocity_pdf.get('mean_template_file', ''))
            r_for_v, vr = utils.read_velocity_template_file(template_file)
            self.radial_velocity = _spline(r_for_v, vr)
            print('Using template for mean velocity profile, overriding matter ccf options')
            self.has_velocity_template = True
        elif mean_model == 'nonlinear' and not self.matter_model == 'excursion_set':
            raise InputError('Cannot have nonlinear mean velocity model unless using excursion_set matter model')
        disp_model = velocity_pdf.get('dispersion_model', 'template')
        if disp_model == 'constant' or velocity_pdf.get('dispersion_template_file', '') == '':
            # default to a constant dispersion
            self.r_for_sv = self.r
            self.mu_for_sv = np.linspace(0, 1)
            self.sv_rmu = np.ones((len(self.mu_for_sv), len(self.r_for_sv)))
        elif disp_model == 'template':
            sv_template_file = os.path.join(model.get('dir', ''), velocity_pdf.get('dispersion_template_file', ''))
            self.r_for_sv, self.mu_for_sv, self.sv_rmu = utils.read_dispersion_template_file(sv_template_file)
        else:
            raise InputError(f"Bad choice '{disp_model}' for dispersion model, options are 'template' or 'constant'")

    def get_interpolated_real_multipoles(self, beta):
        """
        Interpolate the realspace multipoles over stored grid of beta=f/b and return their values at point

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        multipoles : array
            A two-dimensional array with each row corresponding to the interpolated value of a multipole
            The orders of the multipoles are those given by :attr:`poles_r`
        """

        if self.fixed_real_input:
            multipoles = np.empty((len(self.poles_r), len(self.r)))
            for i, ell in enumerate(self.poles_r):
                multipoles[i] = self.real_multipoles[f'{ell}']
            return np.atleast_2d(multipoles)
        else:
            multipoles = np.empty((len(self.poles_r), len(self.beta), len(self.r)))
            for i, ell in enumerate(self.poles_r):
                multipoles[i] = self.real_multipoles[f'{ell}']
            return np.atleast_2d(si.PchipInterpolator(self.beta, multipoles, axis=1)(beta))

    def delta_profiles(self, r, params, **kwargs):
        """
        Compute the matter cross-correlation monopole and its integral

        The matter ccf monopole is the profile of the matter density contrast at distance `r`, , i.e. the function
        :math:`\delta(r)`. Its integral gives the excess matter density within a sphere of radius `r`.
        Parameters
        ----------
        r : array
            Array of distances at which the monopole value is required

        params : dict
            Dictionary of cosmological parameters

        kwargs : dict
            Optional arguments to override :attr:`model`

        Returns
        -------
        delta : array
            The matter ccf monopole, evaluated at `r`

        integrated_delta : array
            The integrated profile, evaluated at `r`
        """

        model = copy.deepcopy(self.model)  # so that we never overwrite the default settings
        for key, value in kwargs.items():
            model[key] = value  # override defaults

        if model['matter_model'] == 'linear_bias':
            beta = params['beta']
            bias = params.get('bias', model['bias'])
            real_monopole = self.get_interpolated_real_multipoles(beta)[0]
            xir = _spline(self.r, real_monopole, ext=3)
            # perform the integral using trapz as faster than quad
            integral = np.zeros_like(r)
            for i in range(len(r)):
                rarr = np.linspace(0, r[i], 100)
                integral[i] = np.trapz(xir(rarr) * rarr**2, rarr)
            delta = xir(r) / bias
            integrated_delta = 3 * integral / (bias * r**3)
            return delta, integrated_delta
        elif model['matter_model'] == 'template':
            return self.delta(r), self.integrated_delta(r)
        elif model['matter_model'] == 'excursion_set':
            excursion_model = self.set_ESM_params(params, model)
            integrated_delta = excursion_model.model_enclosed_density_profile(r, self.z_eff, params['b10'],
                                                                              params['b01'], params['Rp'],
                                                                              params['Rx'],
                                                                              params.get('delta_c', 1.686))
            derivative = np.gradient(integrated_delta(r), r)
            delta = _spline(r, integrated_delta(r) + r * derivative / 3)
            return delta(r), integrated_delta(r)
        else:
            raise InputError(f"Invalid choice of matter_model {model['matter_model']}")

    def velocity_terms(self, r, params, **kwargs):
        """
        Calculate the mean radial velocity profile :math:`v_r(r)` and its derivative with respect to r

        Parameters
        ----------
        r : array
            The radial distances (in the fiducial cosmology) at which to calculate velocity terms

        params : dict
            Dictionary of parameter values for calculation

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        vr : array
            Radial velocity profile evaluated at `r`

        dvr : array
            Derivative of the radial velocity profile, also evaluated at `r`
        """

        model = copy.deepcopy(self.model)  # so that we never overwrite the default settings
        for key, value in kwargs.items():
            model[key] = value  # override defaults

        delta_r, integrated_delta_r = self.delta_profiles(r, params, **kwargs)
        delta = _spline(r, delta_r)
        int_delta = _spline(r, integrated_delta_r)

        # set the term proportional to growth rate: different in different models
        if model['matter_model'] == 'linear_bias':
            # here we want to multiply by growth rate f alone, since there is a 1/b factor already included
            # in delta and int_delta; we obtain this as beta*b (so that bias values cancel out)
            growth_term = params['beta'] * params.get('bias', model['bias'])
        if model['matter_model'] == 'template':
            # here we want to rescale the template by sigma8 / sigma8_template as well as multiply by f
            growth_term = params['fsigma8'] / self.template_sigma8
        if model['matter_model'] == 'excursion_set':
            # just multiply by f alone
            growth_term = params['f']

        # now the actual velocity profile calculation
        if model['mean_model'] == 'linear':
            # simplest linearised form of the continuity equation, potentially with an empirical correction
            if not model['empirical_corr']:
                vr = -growth_term * r * int_delta(r) / (3 * self.iaH)
                dvr = -growth_term * (delta(r) - 2 * int_delta(r) / 3) / self.iaH
            else:
                # add multiplicative empirical correction factor (1 + Av*delta(r))
                Av = params.get('Av', model.get('Av', 0)) # defaults to 0 unless sampled/set
                vr = -growth_term * r * int_delta(r) * (1 + Av * delta(r)) / (3 * self.iaH)
                # build a finer grid to better estimate derivative numerically
                rgrid = np.linspace(0.1, self.r.max(), 100)
                vr_grid = -growth_term * rgrid * int_delta(rgrid) * (1 + Av * delta(rgrid)) / (3 * self.iaH)
                dvr_interp = _spline(rgrid, np.gradient(vr_grid, rgrid), ext=3)
                dvr = dvr_interp(r)
        if model['mean_model'] == 'nonlinear':
            excursion_model = self.set_ESM_params(params, model)
            # model prediction for derivative of enclosed density wrt log scale factor
            logderiv_Delta = excursion_model.density_evolution(self.z_eff, param['b10'], params['b01'], params['Rp'],
                                                               params['Rx'], deltac=params.get('deltac', 1.686),
                                                               r_max=np.max(r))
            if not model['empirical_corr']:
                # fully non-linear continuity equation
                vr = -growth_term * r * logderiv_Delta(r) / (3 * self.iaH * (1 + delta(r)))
                # build a finer grid to better estimate derivative numerically
                rgrid = np.linspace(0.1, self.r.max(), 100)
                vr_grid = -growth_term * rgrid * logderiv_Delta(rgrid) / (3 * self.iaH * (1 + delta(rgrid)))
                dvr_interp = _spline(rgrid, np.gradient(vr_grid, rgrid), ext=3)
                dvr = dvr_interp(r)
            else:
                # add multiplicative empirical correction factor (1 + Av*delta(r))
                Av = params.get('Av', model.get('Av', 0)) # defaults to 0 unless sampled/set
                vr = -growth_term * r * logderiv_Delta(r) * (1 + Av * delta(r))/ (3 * self.iaH * (1 + delta(r)))
                # build a finer grid to better estimate derivative numerically
                rgrid = np.linspace(0.1, self.r.max(), 100)
                vr_grid = -growth_term * rgrid * logderiv_Delta(rgrid) / (3 * self.iaH * (1 + delta(rgrid)))
                dvr_interp = _spline(rgrid, np.gradient(vr_grid, rgrid), ext=3)
                dvr = dvr_interp(r)
        if model['mean_model'] == 'template':
            if not self.has_velocity_template:
                raise InputError('velocity_terms: Cannot use template option as no template has been supplied.')
            vr = self.radial_velocity(r)
            # build a finer grid to better estimate derivative numerically
            rgrid = np.linspace(0.1, self.r.max(), 100)
            dvr_interp = _spline(rgrid, np.gradient(self.radial_velocity(r_grid), rgrid), ext=3)
            dvr = dvr_interp(r)

        return vr, dvr

    def set_ESM_params(self, params, model=None):
        """
        Initialize an instance of :class:`ExcursionSetProfile` with appropriate parameters and settings

        Parameters
        ----------
        params : dict
            Dictionary of parameter values

        model : dict, optional
            Dictionary containing other defaults and options

        Returns
        -------
        excursion_model : instance of :class:`ExcursionSetProfile`
        """
        if model is None:
            model = self.model # revert to defaults already set

        # check that minimum reqd parameters are present
        for chk in ['b10', 'b01', 'Rp', 'Rx']:
            if not chk in params:
                raise InputError(f'set_ESM_params: Parameter {chk} is required for ESM calculation but not provided')

        # assign all other parameter values
        omm = params.get('Omega_m', model['cosmology'].get('Omega_m', 0.31))
        omk = params.get('Omega_k', model['cosmology'].get('Omega_k', 0))
        omb = params.get('Omega_b', model['cosmology'].get('Omega_b', 0.048))
        s80 = params.get('sigma_8_0', model['cosmology'].get('sigma_8', 0.81))
        h = params.get('H0', model['cosmology'].get('H0', 67.5)) / 100
        ns = params.get('ns', 0.96)
        mnu = params.get('mnu', 0.96)
        deltac = params.get('delta_c', 1.686)
        eisenstein_hu = model['excursion_set_options'].get('use_eisenstein_hu', False)
        accuracy = model['excursion_set_options'].get('camb_accuracy', 1)

        # Initialize
        excursion_model = get_excursion_set_model(h, omm, omb, mnu, ns, omk, self.z_eff, eisenstein_hu, camb_accuracy)
        excursion_model.set_normalisation(s80, z=0)
        # record the value of sigma_8 at redshift z
        self.s8z = excursion_model.s8z_fiducial * np.sqrt(excursion_model.normalisation)

        return excursion_model

    def theory_xi(self, s, mu, params, **kwargs):
        """
        Calculate the model prediction for the redshift-space ccf xi(s, mu)

        Input arguments `s` and `mu` can be 1D arrays or 2D coordinate arrays generated using `np.meshgrid()`
        Parameters
        ----------
        s : array
            Radial coordinate in redshift space. This is interpreted as the radial coordinate in the
            fiducial cosmology, and is corrected based on the AP parameters passed

        mu : array
            Cosine of the angle to the line of sight direction

        params : dict
            Dictionary containing parameter values at which to evaluate the model

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        xirmu : 2D array
            Model ccf. If `s` and `mu` are 1D arrays, `xirmu` has same shape as `np.meshgrid(s, mu)`.
            Otherwise it has the same shape as `s` and `mu`.
        """

        model = copy.deepcopy(self.model)  # so that we never overwrite the default settings
        for key, value in kwargs.items():
            model[key] = value  # override defaults

        # build grid of coordinates at which to evaluate the model
        x = np.linspace(-6, 6) if model['rsd_model'] in ['streaming', 'dispersion'] else 0
        if np.ndim(s) == 2 and np.ndim(mu) == 2:
            if not s.shape == mu.shape:
                raise InputError('theory_xi: If arguments s and mu are 2D arrays they must have same shape')
            S, Mu, X = np.meshgrid(s[0], mu[:, 0], x)
        elif np.ndim(s) == 1 and np.ndim(mu) == 1:
            S, Mu, X = np.meshgrid(s, mu, x)
        else:
            raise InputError('theory_xi: arguments s and mu have incompatible dimensions')

        # parameters that are always used for all models
        beta = params['beta']
        # following allows for differences in which combination of AP parameters are sampled
        epsilon = params.get('epsilon', params.get('aperp', 1) / params.get('apar', 1))
        apar = params.get('apar', params.get('alpha', 1) * params.get('epsilon', 1)**(-2/3))

        # --- rescale real-space functions to account for Alcock-Paczynski dilation --- #
        mu_vals = np.linspace(1e-8, 1)
        mu_integral = np.trapz(apar * np.sqrt(1 + (1 - mu_vals**2) * (epsilon**2 - 1)), mu_vals)
        reference_r = self.r
        rescaled_r = reference_r * mu_integral
        # real-space correlation
        ccf_mult = self.get_interpolated_real_multipoles(beta)
        real_multipoles  = {}
        for i, ell in enumerate(self.poles_r):
            real_multipoles[f'{ell}'] = _spline(rescaled_r, ccf_mult[i])
        # velocity terms
        vr, dvr = self.velocity_terms(reference_r, params, **kwargs)
        if model['matter_model'] == 'excursion_set':
            # this is a special case: as the model predicts the absolute scale of void-matter ccf we do not rescale
            # in principle, means using void size as standard ruler; in practice has very little information because
            # of the additional nuisance parameters (and because realspace multipoles of xi still get rescaled)
            vr_interp = _spline(reference_r, vr)
            dvr_interp = _spline(reference_r, dvr)
        else:
            # rescale as normal
            vr_interp = _spline(rescaled_r, vr)
            dvr_interp = _spline(rescaled_r, dvr)
        if model['rsd_model'] in ['streaming', 'dispersion']:
            # we will rescale the actual dispersion function later, but scale the amplitude with apar here
            sigma_v = params.get('sigma_v', 380) * apar

        # apply AP corrections to shift input coordinates in the fiducial cosmology to those in true cosmology
        mu_s = Mu
        mu_s[Mu>0] = 1 / np.sqrt(1 + epsilon**2 * (1 / Mu[Mu>0]**2 - 1))
        s_perp = S * np.sqrt(1 - mu_s**2) * params.get('aperp', 1.0)
        s_par = S * mu_s * params.get('apar', 1.0)
        s = np.sqrt(s_par**2 + s_perp**2) # note this is no longer same as the input!

        if model['rsd_model'] in ['streaming', 'dispersion']:

            v_par = X * sigma_v # range of integration large enough to converge to integral over (-infty, infty)
            if model['rsd_model'] == 'streaming':
                r_par = s_par - v_par * self.iaH
                r = np.sqrt(s_perp**2 + r_par**2)
                mu_r = r_par / r
                # now scale the dispersion function for AP dilation and then evaluate
                sv_spl = si.RectBivariateSpline(self.r_for_sv * mu_integral, self.mu_for_sv, self.sv_rmu.T)
                sv = sigma_v * sv_spl.ev(r, mu_r)
                vel_pdf = norm.pdf(v_par, loc=vr_interp(r) * mu_r, scale=sv)
                jacobian = 1 # no change in variables
            else:
                # start by iteratively solving for the mean real-space coordinate
                r_par = s_par / (1 + self.iaH * vr_interp(s) / s)
                for i in range(model.get('niter', 5)):
                    r = np.sqrt(s_perp**2 + r_par**2)
                    r_par = s_par / (1 + self.iaH * vr_interp(r) / r)
                r_par -= v_par * self.iaH
                r = np.sqrt(s_perp**2 + r_par**2)
                mu_r = r_par / r
                # now scale the dispersion function for AP dilation and then evaluate
                sv_spl = si.RectBivariateSpline(self.r_for_sv * mu_integral, self.mu_for_sv, self.sv_rmu.T)
                sv = sigma_v * sv_spl.ev(r, mu_r)
                vel_pdf = norm.pdf(v_par, loc=0, scale=sv)
                # as we've changed variables account for this in the Jacobian
                jacobian = 1 / (1 + vr_interp(r)*self.iaH/r + self.iaH * mu_r**2 * (dvr_interp(r) - vr_interp(r)/r))

            # build the real-space ccf at each point
            if model['assume_isotropic_realspace']:
                xi_rmu =  real_multipoles['0'](r) * legendre(0)(mu_r)
            else:
                xi_rmu = np.zeros_like(r)
                for ell in self.poles_r:
                    xi_rmu = xi_rmu + real_multipoles[f'{ell}'](r) * legendre(ell)(mu_r)

            # integrate for model prediction
            xi_smu = simps((1 + xi_rmu) * jacobian * vel_pdf, x=v_par, axis=2) - 1

        elif model['rsd_model'] == 'Kaiser':

            # get possible additional nuisance parameters if included, as per Hamaus et al 2020
            M = params.get('M', 1.0)
            Q = params.get('Q', 1.0)

            if model.get('Kaiser_coord_shift', True):
                # solve iteratively for mean real-space coordinate
                r_par = s_par / (1 + M * self.iaH * vr_interp(s) / s)
                for i in range(model.get('niter', 5)):
                    r = np.sqrt(s_perp**2 + r_par**2)
                    r_par = s_par / (1 + M * self.iaH * vr_interp(r) / r)
            else:
                # NOTE: this is incorrect! it is included only to allow users to reproduce results in some
                # previous papers which do not include the coordinate shift!
                r_par = s_par
            r = np.sqrt(s_perp**2 + r_par**2)
            mu_r = r_par / r

            if model.get('Kaiser_approximation', False):
                # approximate Jacobian by a series expansion truncated at linear order in velocity terms
                jacobian = 1 - vr_interp(r)*self.iaH/r - Q*mu_r**2*self.iaH*(dvr_interp(r) - vr_interp(r)/r)
            else:
                # use the full expression without approximation
                jacobian = 1 / (1 + vr_interp(r)*self.iaH/r + Q*mu_r**2*self.iaH*(dvr_interp(r) - vr_interp(r)/r))

            # build the real-space ccf at each point
            if model['assume_isotropic_realspace']:
                xi_rmu =  real_multipoles['0'](r) * legendre(0)(mu_r)
            else:
                xi_rmu = np.zeros_like(r)
                for ell in self.poles_r:
                    xi_rmu = xi_rmu + real_multipoles[f'{ell}'](r) * legendre(ell)(mu_r)

            # obtain the model without integration (ie assuming the pdf is a delta function)
            xi_smu = (1 + M * xi_rmu) * jacobian - 1
            # drop the unnecessary dimension
            xi_smu = xi_smu[:, :, 0]

        else:
            raise InputError(f"theory_xi: Unrecognised choice of model {model['rsd_model']}")

        return xi_smu

    def theory_multipoles(self, s, params, poles=[0, 2], **kwargs):
        """
        Calculate the model prediction for specified Legendre multipoles of the redshift-space ccf

        Parameters
        ----------
        s : array
            Radial coordinate in redshift space at which multipoles are desired

        params : dict
            Dictionary containing parameter values at which to evaluate the model

        poles : tuple of ints, default=[0, 2]
            Legendre multipoles to return

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        multipoles : dict
            Dictionary of multipoles, with keys being the poles requested (e.g. '0', '2') and items
            being arrays of same length as `s`
        """

        poles = np.atleast_1d(poles)
        if np.any(poles % 2):
            even = False
            mu = np.linspace(-1, 1, 100)
        else:
            even = True
            mu = np.linspace(0, 1, 100)
        xi_smu = self.theory_xi(*np.meshgrid(s, mu), params, **kwargs)
        xi_model = si.interp2d(s, mu, xi_smu, kind='cubic')
        multipoles = utils.multipoles_from_fn(xi_model, s, poles, even=even)

        return multipoles

    def theory_multipole_vector(self, s, params, poles=[0, 2], **kwargs):
        """
        Calculate a theory vector composed of all desired multipoles, for easy comparison with an observed datavector
        and chi-square calculation

        Parameters
        ----------
        s : array
            Radial distance coordinate values at which to evaluate theory multipoles

        params : dict
            Dictionary of parameter values at which to evaluate the model

        poles : list, default=[0, 2]
            Legendre multipoles to include in the theory vector

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        theory_vector : array
            Array of length `len(s) * len(poles)` containing the model multipoles
        """

        multipoles = self.theory_multipoles(s, params, poles, **kwargs)
        poles = np.atleast_1d(poles)
        theory_vector = np.zeros(len(poles) * len(s))
        for i, ell in enumerate(poles):
            theory_vector[i*len(s) : (i+1)*len(s)] = multipoles[f'{ell}']

        return theory_vector

    def theory_xi_2D(self, params, rmax=85, **kwargs):
        r"""
        Compute the theory model prediction as a function of distances perpendicular and parallel to the line of sight

        Parameters
        ----------
        params : dict
            Dictionary containing parameter values at which to evaluate the model

        rmax : float, default=85
            Maximum distance (in each direction) out to which model should be calculated

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        xi_model : instance of :class:`scipy.interpolate.interp2d`
            Representing the redshift-space model prediction :math:`\xi^s(s_\perp, s_{||})`
        """

        sperp = np.linspace(0.01, rmax)
        spar = np.linspace(-rmax, rmax)  # allow for potential odd functions
        def smu(sigma, pi):
            s = np.sqrt(sigma**2 + pi**2)
            return s, pi/s
        s, mu = distance_modulus(*np.meshgrid(sperp, spar))
        xi_smu = self.theory_xi(s, mu, params, **kwargs)
        xi_model = si.interp2d(sperp, spar, xi_smu)
        return xi_model

    def xi_2D_from_multipoles(self, multipoles, poles, rmax=85):
        r"""
        Use input Legendre multipoles to generate a 2D ccf as a function of distances perpendicular and parallel to the
        line of sight

        Parameters
        ----------
        multipoles : dict
            Dictionary of callable multipole functions accepting distance modulus argument, with keys being the Legendre order

        poles : list
            The Legendre multipole orders, should match the keys in `multipoles`

        rmax : float, default=85
            Maximum distance (in each direction) out to which to interpolate

        Returns
        -------
        xi_2D : instance of :class:`scipy.interpolate.interp2d`
            Representing the 2D ccf :math:`\xi(r_\perp, r_{||})`
        """

        sperp = np.linspace(0.01, rmax)
        spar = np.linspace(-rmax, rmax)  # allow for potential odd functions
        def smu(sigma, pi):
            s = np.sqrt(sigma**2 + pi**2)
            return s, pi/s
        s, mu = distance_modulus(*np.meshgrid(sperp, spar))

        xi_2D_grid = np.zeros_like(s)
        for ell in poles:
            xi_2D_grid = xi_2D_grid + multipoles[f'{ell}'](s) * legendre(ell)(mu)
        xi_2D = si.interp2d(sperp, spar, xi_2D_grid)
        return xi_2D

class CCFFit(CCFModel):
    """
    Class to perform fits to measured cross-correlation function data
    """

    def __init__(self, model, data):
        """
        Initialise :class:`CCFFit`.

        Parameters
        ----------
        model : dict
            Dictionary of input options for calculation of the theory model

        data : dict
            Dictionary of input options and file paths for the ccf data
        """

        super(CCFFit, self).__init__(model)
        self._load_redshiftspace_ccf(data)
        self._load_covariance_matrix(data)
        self.fit_options = {'beta_interpolation': data.get('beta_interpolation', 'datavector'),
                            'likelihood': data.get('likelihood', {'form': 'Gaussian'})}

    def _load_redshiftspace_ccf(self, data):
        """"""

        if data.get('reconstruction', True):
            self.fixed_data = False
            beta_grid = data['redshift_space_ccf'].get('beta_grid', None)
            if beta_grid is None:
                self.beta_ccf = self.beta
            elif isinstance(beta_grid, str):
                beta_file = os.path.join(data.get('dir',''), beta_grid)
                self.beta_ccf = self._load_beta(beta_file)
            else:
                self.beta_ccf = np.array(beta_grid)
            if not np.all(self.beta_ccf[1:] - self.beta_ccf[:-1] > 0):
                raise InputError('Redshift-space beta grid must be strictly monotonically increasing')
        else:
            self.fixed_data = True

        data_file = os.path.join(data.get('dir', ''), data['redshift_space_ccf']['data_file'])
        self.format = data['redshift_space_ccf'].get('format', 'multipoles')
        if self.format == 'multipoles':
            self.s, self.poles_s, self.redshift_multipoles = utils.read_cf_multipoles_file(data_file)
        elif self.format == 'rmu' or self.format == 'sigmapi':
            raise InputError("Do you REALLY want to use 2D ccf data? I haven't coded this yet")
        else:
            raise InputError("Unrecognised format for redshift-space ccf")

    def _load_covariance_matrix(self, data):
        """"""

        if data.get('reconstruction', True):
            self.fixed_covmat = data['covariance_matrix'].get('fixed_beta', True)
            if not self.fixed_covmat:
                beta_grid = data['covariance_matrix'].get('beta_grid', None)
                if beta_grid is None:
                    self.beta_covmat = self.beta_ccf
                elif isinstance(beta_grid, str):
                    beta_file = os.path.join(data.get('dir', ''), beta_grid)
                    self.beta_covmat = self._load_beta(beta_file)
                else:
                    self.beta_covmat = np.array(beta_grid)
                if not np.all(self.beta_covmat[1:] - self.beta_covmat[:-1] > 0):
                    raise InputError('Covariance matrix beta grid must be stricly monotonically increasing')
        else:
            self.fixed_covmat = True

        covmat_file = os.path.join(data.get('dir', ''), data['covariance_matrix']['data_file'])
        extensions = {'npy': ['.npy'],
                      'hdf5': ['.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5', '.h5py']}
        for file_format, exts in extensions.items():
            if any(covmat_file.endswith(ext) for ext in exts):
                if file_format == 'npy':
                    covmat = np.load(covmat_file, allow_pickle=True)
                    success = True
                    try:
                        for poss in ['covmat', 'covariance', 'cov']:
                            if poss in covmat.item():
                                covmat = covmat.item()[poss]
                    except:
                        pass
                elif file_format == 'hdf5':
                    import h5py
                    with h5py.File(covmat_file, 'r') as file:
                        for poss in ['covmat', 'covariance', 'cov']:
                            if poss in file.keys():
                                covmat = file[poss][:]
                                success = True
        if not success:
            try:
                covmat = np.genfromtxt(covmat_file)
                success = True
            except Exception:
                pass
        if not success:
            raise InputError(f'Failed to read covariance matrix from {covmat_file} - check file')

        # check the size of the covariance matrix is right
        # NOTE: the data vector we use is composed of stacking the observed multipoles, so this sets the shape
        # of the covariance matrix
        if self.fixed_covmat:
            if not (covmat.shape == ((len(self.s) * len(self.poles_s), len(self.s) * len(self.poles_s)))):
                raise InputError('Unexpected shape of (fixed) covariance matrix')
        else:
            if not (covmat.shape == ((len(self.beta_covmat), len(self.s) * len(self.poles_s), len(self.s) * len(self.poles_s)))):
                raise InputError('Unexpected shape of (beta-varying) covariance matrix')

        self.covmat = covmat
        self.icov = np.linalg.inv(self.covmat)

    def get_interpolated_redshift_multipoles(self, beta):
        """
        Interpolate the redshift-space multipoles over stored grid of beta=f/b and return their values at point

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        multipoles : array
            A two-dimensional array with each row corresponding to the interpolated value of a multipole
            The orders of the multipoles are those given by :attr:`poles_s`
        """

        if self.fixed_data:
            multipoles = np.empty((len(self.poles_s), len(self.s)))
            for i, ell in enumerate(self.poles_s):
                multipoles[i] = self.redshift_multipoles[f'{ell}']
            return np.atleast_2d(multipoles)
        else:
            multipoles = np.empty((len(self.poles_s), len(self.beta_ccf), len(self.s)))
            for i, ell in enumerate(self.poles_s):
                multipoles[i] = self.redshift_multipoles[f'{ell}']
            return np.atleast_2d(si.PchipInterpolator(self.beta_ccf, multipoles, axis=1)(beta))

    def get_interpolated_covariance(self, beta):
        """
        Interpolate the covariance matrix over stored grid of beta=f/b and return its value at point

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        covmat : array
            A 2D array containing the covariance matrix
        """

        if self.fixed_covmat:
            return self.covmat
        else:
            # if requested beta is outside the provided grid, return the boundary value covmat
            # for a grid of beta that is wide relative to the posterior on beta this does not affect science
            # results, but it does prevent the code from crashing when accidentally hitting a boundary
            if beta < self.beta_covmat.min(): return self.covmat[0]
            if beta > self.beta_covmat.max(): return self.covmat[-1]

            if beta in self.beta_covmat:
                return self.covmat[np.where(self.beta_covmat==beta)[0][0]]

            # if nothing else, linearly inteprolate between two bracketing entries
            lowind = np.where(self.beta_covmat < beta)[0][-1]
            highind = np.where(self.beta_covmat >= beta)[0][-1]
            t = (beta - self.beta_covmat[lowind]) / (self.beta_covmat[highind] - self.beta_covmat[lowind])
            return (1 - t) * self.covmat[lowind] + t * self.covmat[highind]

    def get_interpolated_precision(self, beta):
        """
        Interpolate the precision (=inverse covariance) matrix over stored grid of beta=f/b and return its value at point

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        icovmat : array
            A 2D array containing the inverse covariance or precision matrix
        """

        if self.fixed_covmat:
            return self.icov
        else:
            if beta < self.beta_covmat.min(): return self.icov[0]
            if beta > self.beta_covmat.max(): return self.icov[-1]

            if beta in self.beta_covmat:
                return self.icov[np.where(self.beta_covmat==beta)[0][0]]

            # if nothing else, linearly inteprolate between two bracketing entries
            lowind = np.where(self.beta_covmat < beta)[0][-1]
            highind = np.where(self.beta_covmat >= beta)[0][-1]
            t = (beta - self.beta_covmat[lowind]) / (self.beta_covmat[highind] - self.beta_covmat[lowind])
            return (1 - t) * self.icov[lowind] + t * self.icov[highind]

    def correlation_matrix(self, beta):
        """
        Compute the normalised correlation matrix at input value of beta

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        corrmat : array
            A 2D array containing the correlation matrix
        """

        covmat = self.get_interpolated_covmat(beta)
        corrmat = np.zeros_like(covmat)
        diagonals = np.sqrt(np.diag(covmat))
        for i in range(corrmat.shape[0]):
            for j in range(corrmat.shape[1]):
                if not (diagonals[i] * diagonals[j] == 0):
                    corrmat[i, j] = covmat[i, j] / (diagonals[i] * diagonals[j])
        return corrmat

    def diagonal_errors(self, beta):
        """
        Compute approximate errors in each bin of the multipole data vector from diagonal entries of the covariance

        Parameters
        ----------
        beta : float
            Coordinate at which to evaluate the interpolation

        Returns
        -------
        errors : array
            A 2D array of shape (M, N) where M is the length of :attr:`poles_s` and N is the length of :attr:`s`
            That is, each row contains the diagonal errors in a given multipole
        """

        covmat = self.get_interpolated_covariance(beta)
        diagonals = np.sqrt(np.diag(covmat))
        return diagonals.reshape((len(self.poles_s), len(self.s)))

    def multipole_datavector(self, beta=0.4):
        """
        Create the data vector composed of the stack of all redshift-space multipoles

        Parameters
        ----------
        beta : float, default=0.4
            Reconstruction parameter at which to evaluate the redshift-space multipoles. If no reconstruction is used and
            redshift-space ccf does not depend on beta, this parameter is ignored

        Returns
        -------
        data_vector : array
            1D array containing the redshift-space multipoles
        """

        multipoles = self.get_interpolated_redshift_multipoles(beta)
        return multipoles.reshape(len(self.poles_s) * len(self.s))

    def chi_squared(self, params, **kwargs):
        """
        Compute the chi-squared value for fit of theory model to the data vector at a point in parameter space

        Since this calculation depends on a covariance matrix which may itself vary with the parameter beta, the method
        returns the covariance matrix used for evaluation as well: this can be used in normalizing the likelihood

        Parameters
        ----------
        params : dict
            Dictionary containing parameter values at which to evaluate the model

        kwargs : dict
            Optional settings to override those in :attr:`model`

        Returns
        -------
        chisq : float
            The chi-squared value

        covmat : array
            The 2D covariance matrix used for this evaluation
        """

        theory_vector = self.theory_multipole_vector(self.s, params, self.poles_s, **kwargs)
        data_vector = self.multipole_datavector(params['beta'])
        cov = self.get_interpolated_covariance(params['beta'])
        icov = self.get_interpolated_precision(params['beta'])

        return np.dot(np.dot(theory_vector - data_vector, icov), theory_vector - data_vector), cov

    def log_likelihood(self, params, **kwargs):
        """
        Compute the log likelihood for the theory model at a point in parameter space

        Parameters
        ----------
        params : dict
            Dictionary of parameter values at which to evaluate the model

        kwargs : dict
            Optional dictionary of settings to override those in :attr:`model` (used for theory evaluation) and in
            :attr:`fit_options` (to determine the likelihood evaluation)

        Returns
        -------
        lnlike : float
            Log likelihood

        chisq : float
            The chi-square value (which may be different to -2*lnlike depending on the likelihood form)
        """

        # override defaults with settings provided
        fit_options = copy.deepcopy(self.fit_options)
        for key in kwargs:
            fit_options[key] = kwargs[key]

        if fit_options['beta_interpolation'] =='likelihood' and not self.fixed_data:
            # evaluate the chi-square at the two beta values in the grid bracketing the input beta and
            # interpolate between them rather than interpolating the theory/datavectors themselves
            # NOTE: this option will also avoid interpolating the covariance matrix, IF the covariance is
            # measured on the same grid as the data multipoles (why wouldn't it be?) or if it is fixed (duh!)
            beta = copy.deepcopy(params['beta'])
            lowind = np.where(self.beta_ccf < beta)[0][-1]
            highind = np.where(self.beta_ccf >= beta)[0][0]
            t = (beta - self.beta_ccf[lowind]) / (self.beta_ccf[highind] - self.beta_ccf[lowind])
            print(t, self.beta_ccf[lowind], self.beta_ccf[highind])
            params['beta'] = self.beta_ccf[lowind]
            low_chisq, low_cov = self.chi_squared(params, **kwargs)
            params['beta'] = self.beta_ccf[highind]
            high_chisq, high_cov = self.chi_squared(params, **kwargs)
            params['beta'] = beta # return to original value!

            if not self.fixed_covmat:
                # need to normalize for changing covariance matrix in likelihood calculation
                det = np.linalg.slogdet(low_cov)
                if not det[0] == 1:
                    print(f'Singular covariance matrix at beta={self.beta_ccf[lowind]}, likelihood failed')
                    # NOTE: If it fails at either point we treat it as failing at both, I *think* this is best
                    return -np.inf, np.inf
                low_like_factor = -0.5 * det[1]
                det = np.linalg.slogdet(high_cov)
                if not det[0] == 1:
                    print(f'Singular covariance matrix at beta={self.beta_ccf[highind]}, likelihood failed')
                    # NOTE: If it fails at either point we treat it as failing at both, I *think* this is best
                    return -np.inf, np.inf
                high_like_factor = -0.5 * det[1]
            else:
                low_like_factor, high_like_factor = 0, 0

            if fit_options['likelihood']['form'].lower() == 'sellentin':
                nmocks = fit_options['likelihood'].get('nmocks', 1) # duty on user to provide sensible number
                lnlike_low = -nmocks * np.log(1 + low_chisq / (nmocks -1)) / 2 + low_like_factor
                lnlike_high = -nmocks * np.log(1 + high_chisq / (nmocks -1)) / 2 + high_like_factor
            elif fit_options['likelihood']['form'].lower() == 'hartlap':
                nmocks = fit_options['likelihood'].get('nmocks', 1)
                p = len(self.s) * len(self.poles_s)
                a = (nmocks - p - 2) / (nmocks - 1)
                lnlike_low = -0.5 * low_chisq * a + low_like_factor
                lnlike_high = -0.5 * high_chisq * a + high_like_factor
            elif fit_options['likelihood']['form'].lower() == 'percival':
                nmocks = fit_options['likelihood'].get('nmocks', 1)
                nparams = fit_options['likelihood']['nparams'] # we want it to fail if this is not provided
                ndata = len(self.s) * len(self.poles_s)
                B = (nmocks - ndata - 2) / ((nmocks - ndata - 1) * (nmocks - ndata - 4))
                m = nparams + 2 + (nmocks - 1 + B * (ndata - nparams)) / (1 + B * (ndata - nparams))
                lnlike_low = -m * np.log(1 + low_chisq / (nmocks -1)) / 2 + low_like_factor
                lnlike_high = -m * np.log(1 + high_chisq / (nmocks -1)) / 2 + high_like_factor
            elif fit_options['likelihood']['form'].lower() == 'gaussian':
                lnlike_low = -0.5 * low_chisq + low_like_factor
                lnlike_high = -0.5 * high_chisq + high_like_factor
            else:
                raise InputError('Unrecognised likelihood form')
            lnlike = (1 - t) * lnlike_low + t * lnlike_high
            chisq =  (1 - t) * low_chisq + t * high_chisq
        else:
            # otherwise, interpolate the multipole data vector and covariance matrix themselves (automatically
            # handles the case where these don't vary with beta too)
            chisq, cov = self.chi_squared(params, **kwargs)
            if not self.fixed_covmat:
                # need to normalize for changing covariance matrix in likelihood calculation
                det = np.linalg.slogdet(cov)
                if not det[0] == 1:
                    print(f"Singular covariance matrix at beta={params['beta']}, likelihood failed")
                    return -np.inf, np.inf
                like_factor = -0.5 * det[1]
            else:
                like_factor = 0

            if fit_options['likelihood']['form'].lower() == 'sellentin':
                nmocks = fit_options['likelihood'].get('nmocks', 1) # duty on user to provide sensible number
                lnlike = -nmocks * np.log(1 + chisq / (nmocks -1)) / 2 + like_factor
            elif fit_options['likelihood']['form'].lower() == 'hartlap':
                nmocks = fit_options['likelihood'].get('nmocks', 1)
                p = len(self.s) * len(self.poles_s)
                a = (nmocks - p - 2) / (nmocks - 1)
                lnlike = -0.5 * chisq * a + like_factor
            elif fit_options['likelihood']['form'].lower() == 'percival':
                nmocks = fit_options['likelihood'].get('nmocks', 1)
                nparams = fit_options['likelihood']['nparams'] # we want it to fail if this is not provided
                ndata = len(self.s) * len(self.poles_s)
                B = (nmocks - ndata - 2) / ((nmocks - ndata - 1) * (nmocks - ndata - 4))
                m = nparams + 2 + (nmocks - 1 + B * (ndata - nparams)) / (1 + B * (ndata - nparams))
                lnlike = -m * np.log(1 + chisq / (nmocks -1)) / 2 + like_factor
            elif fit_options['likelihood']['form'].lower() == 'gaussian':
                lnlike = -0.5 * chisq + like_factor
            else:
                raise InputError('Unrecognised likelihood form')

        # add a sanity check to check cases which fail due to unforeseen errors (such as -ve chisq from a
        # bad covariance matrix or similar)
        if np.isnan(lnlike):
            print(f'Likelihood evaluation failed, returning NaN. Parameters at fail point: {params}')
            print(f'Chisq: {chisq}, like_factor: {like_factor}')
            lnlike = -np.inf
            chisq = np.inf

        return lnlike, chisq

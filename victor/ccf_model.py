import os
import sys
import functools
import copy
import h5py
import numpy as np
import scipy.interpolate as si
from scipy.integrate import quad, simps
from scipy.stats import norm
from scipy.special import legendre
from . import utils
from .utils import InputError
from .excursion_set_profile import ExcursionSetProfile
from .cosmology import BackgroundCosmology
import matplotlib.pyplot as plt

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

        self.z_eff = model['z_eff']
        cosmo = BackgroundCosmology(model.get('cosmology'))
        self.iaH = (1 + self.z_eff) / (100 * cosmo.Ez(self.z_eff))

        # check the input model data file exists
        base_dir = model.get('dir', '')
        input_fn = os.path.join(base_dir, model['input_model_data_file'])
        if not os.path.isfile(input_fn):
            raise InputError(f'File {input_fn} containing input model data not found')

        # get the format of input model data file
        self.extensions = {'npy': ['.npy'],
                           'hdf5': ['.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5', '.h5py']
                          }
        for file_format, exts in self.extensions.items():
            if any(input_fn.endswith(ext) for ext in exts):
                break

        # get the input data from file
        if file_format == 'npy':
            input_data = np.load(input_fn, allow_pickle=True).item()
        elif file_format == 'hdf5':
            with h5py.File(input_fn, 'r') as f:
                input_data = {}
                for key in list(f.keys()):
                    input_data[key] = f[key][:]

        self._load_realspace_ccf(model['realspace_ccf'], input_data)
        self.matter_model = model['matter_ccf'].get('model', 'linear_bias')
        self.realspace_ccf_from_data = model['realspace_ccf'].get('from_data', False)
        if self.matter_model == 'linear_bias' and not self.realspace_ccf_from_data:
            # when supplying a template real-space ccf we still assume the same scaling as for the template model
            self.template_sigma8 = model['matter_ccf'].get('template_sigma8', None)
            if not self.template_sigma8:
                raise InputError('When using linear bias for the matter ccf and the real-space ccf is from a template, template_sigma8 must be provided')
        if self.matter_model == 'template':
            self._set_matter_ccf_template(model['matter_ccf'], input_data)
        self._set_velocity_pdf(model['velocity_pdf'], input_data)
        del input_data
        # set of model options: these settings provide the default to be used in automated evaluations
        # (eg when running chains), but can be overridden by the user passing an optional dict at the
        # time of model evaluation (eg when debugging or plotting)
        self.model = {'rsd_model': model.get('rsd_model', 'streaming'),
                      'kaiser_approximation': model.get('kaiser_approximation', False),
                      'kaiser_coord_shift': model.get('kaiser_coord_shift', True),
                      'assume_isotropic': model['realspace_ccf'].get('assume_isotropic', True),
                      'realspace_ccf_from_data': self.realspace_ccf_from_data,
                      'matter_model': self.matter_model,
                      'excursion_set_options': model['matter_ccf'].get('excursion_set_options', {}),
                      'bias': model['matter_ccf'].get('bias', 1.9),
                      'mean_model': model['velocity_pdf']['mean'].get('model', 'linear'),
                      'pdf_form': model['velocity_pdf'].get('form', 'gaussian'),
                      'empirical_corr': model['velocity_pdf']['mean'].get('empirical_corr', False),
                     }

    def _load_realspace_ccf(self, realspace_ccf, input_data):
        """Private class method to read realspace ccf input data from file"""

        # format in which the ccf information is provided (multipoles or rmu)
        format = realspace_ccf.get('format', 'multipoles')
        # whether ccf has a dependence on reconstruction beta parameter
        self.fixed_real_input = not(realspace_ccf.get('reconstruction', False))
        # keys to identify relevant information
        ccf_keys = np.atleast_1d(realspace_ccf['ccf_keys'])

        # first get reconstruction beta information, if applicable
        if not self.fixed_real_input:  # get beta
            beta_key = realspace_ccf.get('beta_key', None)
            if beta_key is None: raise InputError('Reconstruction specified for realspace ccf but no beta key provided')
            if not beta_key in input_data:
                raise InputError(f'Key {beta_key} not found in input model data file')
            self.beta = input_data[beta_key]
            # minimal check that this makes sense
            if not np.all((self.beta[1:] - self.beta[:-1]) > 0):
                raise InputError('Realspace beta grid must be strictly monotonically increasing')

        # check the ccf keys
        bad_keys = (format=='multipoles' and len(ccf_keys)<2) or (format=='rmu' and len(ccf_keys)!=3)
        if bad_keys: raise InputError(f'Wrong number of ccf keys provided for ccf format {format}')
        for key in ccf_keys:
            if not key in input_data:
                raise InputError(f'Key {key} not found in input model data file')

        # if the file contains multiple versions of ccf information corresponding to different simulation
        # realisations, we allow the specification of which simulation number to access
        isim = realspace_ccf.get('simulation_number', None)

        # load the ccf data
        if format=='multipoles':
            self.r = input_data[ccf_keys[0]]
            names = ['monopole', 'quadrupole', 'hexadecapole'][:len(ccf_keys)-1]
            self.poles_r = np.atleast_1d([0, 2, 4][:len(ccf_keys)-1])
            self.real_multipoles = {}
            for i, name in enumerate(names):
                if isim is None:
                    self.real_multipoles[f'{self.poles_r[i]}'] = input_data[ccf_keys[i+1]]
                elif isinstance(isim, int):
                    self.real_multipoles[f'{self.poles_r[i]}'] = input_data[ccf_keys[i+1]][isim]
                else:
                    raise InputError('If provided, simulation_number must be an integer')
            # sense check the data
            for i, ell in enumerate(self.poles_r):
                mult_shape = self.real_multipoles[f'{ell}'].shape
                if self.fixed_real_input:
                    if not mult_shape==self.r.shape:
                        raise InputError(f'Shape of real ccf {name[i]} is {mult_shape}, expected {self.r.shape}')
                else:
                    x, y = len(self.beta), len(self.r)
                    if not mult_shape==(x, y):
                        raise InputError(f'Shape of real ccf {name[i]} is {mult_shape}, expected ({x}, {y})')
        elif format=='rmu':
            self.r = input_data[ccf_keys[0]]
            mu = input_data[ccf_keys[1]]
            if isim is None:
                real_ccf = input_data[ccf_keys[2]]
            elif isinstance(isim, int):
                real_ccf = input_data[ccf_keys[2]][isim]
            else:
                raise InputError('If provided, simulation_number must be an integer')
            # sense check the data and convert to multipoles
            self.poles_r = np.array([0, 2, 4])
            ccf_shape = real_ccf.shape
            if self.fixed_real_input:
                if not ccf_shape==(len(self.r), len(mu)):
                    raise InputError(f'Shape of real ccf is {ccf_shape}, expected ({len(self.r)}, {len(mu)})')
                ccf_interp = si.interp2d(self.r, mu, real_ccf.T)
                self.real_multipoles = utils.multipoles_from_fn(ccf_interp, self.r, ell=self.poles_r)
            else:
                if not ccf_shape==(len(self.beta), len(self.r), len(mu)):
                    raise InputError(f'Shape of real ccf is {ccf_shape}, expected ({len(self.beta)}, {len(self.r)}, {len(mu)})')
                self.real_multipoles = {}
                for ell in self.poles_r:
                    self.real_multipoles[f'{ell}'] = np.zeros((len(self.beta), len(self.r)))
                for i in range(len(self.beta)):
                    ccf_interp = si.interp2d(self.r, mu, real_ccf[i].T)
                    tmp_mults = utils.multipoles_from_fn(ccf_interp, self.r, ell=self.poles_r)
                    for ell in self.poles_r:
                        self.real_multipoles[f'{ell}'][i] = tmp_mults[f'{ell}']

    def _set_matter_ccf_template(self, matter_ccf, input_data):
        """Private class method to read matter ccf template input data from file"""

        # information about the template
        self.template_sigma8 = matter_ccf.get('template_sigma8', None)
        if not self.template_sigma8:
            raise InputError('When using template model for the matter ccf, template_sigma8 must be provided')
        template_keys = np.atleast_1d(matter_ccf.get('template_keys'))
        integrated = matter_ccf.get('integrated', False)

        # check the ccf keys
        if not len(template_keys)==2:
            raise InputError('Wrong number of matter ccf template keys provided: expected 2 (radial distance and monopole)')
        for key in template_keys:
            if not key in input_data:
                raise InputError(f'Key {key} not found in input model data file')

        # get the template data
        r_for_delta = input_data[template_keys[0]]
        delta = input_data[template_keys[1]]

        # sense check the data
        # TODO: any additions to deal with a template that is not spherically symmetric should be added here
        if not len(r_for_delta) == len(delta):
             raise InputError(f'Shape of matter ccf template is {len(delta)}, expected {len(r_for_delta)}')

        # calculate profile interpolating functions
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
            self.integrated_delta = _spline(r, integral, ext=3)

    def _set_velocity_pdf(self, velocity_pdf, input_data):
        """Private class method to set velocity pdf input options and read data from file"""

        # model of the mean of the velocity pdf
        mean_model = velocity_pdf['mean'].get('model', 'linear')
        if mean_model=='template':  # template option sometimes used for specific testing
            template_keys = np.atleast_1d(velocity_pdf['mean'].get('template_keys'))
            if not len(template_keys) == 2:
                raise InputError(f'{len(template_keys)} velocity mean template keys provided, require 2')
            for key in template_keys:
                if not key in input_data:
                    raise InputError(f'Key {key} not found in input model data file')
            r_for_v = input_data[template_keys[0]]
            vr = input_data[template_keys[1]]
            if not len(r_for_v) == len(vr):
                 raise InputError(f'Shape of mean velocity template is {len(vr)}, expected {len(r_for_v)}')
            self.radial_velocity = _spline(r_for_v, vr, ext=3)
            self.has_velocity_template = True
        else:
            self.has_velocity_template = False

        if mean_model=='nonlinear' and not self.matter_model == 'excursion_set':
            raise InputError('Cannot have nonlinear mean velocity model unless using excursion_set matter model')

        # model of the dispersion of the pdf (which is assumed Gaussian, at least for now)
        dispersion = velocity_pdf.get('dispersion', {})
        disp_model = dispersion.get('model', 'constant')
        if disp_model == 'template':
            # read in a template
            template_keys = np.atleast_1d(dispersion.get('template_keys'))
            # check the keys
            if len(template_keys) < 2 or len(template_keys) > 3:
                raise InputError(f'{len(template_keys)} velocity dispersion template keys provided, require 2 or 3')
            for key in template_keys:
                if not key in input_data:
                    raise InputError(f'Key {key} not found in input model data file')
            # first two keys should always provide sigmav and r
            self.r_for_sv = input_data[template_keys[0]]
            sv = input_data[template_keys[-1]]
            if len(template_keys)==2:
                # dispersion template is isotropic, sigmav = sigmav(r) - but for coding convenience we
                # want to add a dummy mu dimension, with the template being independent of mu
                self.mu_for_sv = np.linspace(0, 1)
                sv = (np.ones((len(self.mu_for_sv), len(self.r_for_sv))) * sv).T
            else:
                # dispersion template is anisotropic, sigmav = sigmav(r, mu), so we also pick up mu
                self.mu_for_sv = input_data[template_keys[1]]
            # sense check dimensions
            if not sv.shape==(len(self.r_for_sv), len(self.mu_for_sv)):
                lenr, lenmu = len(self.r_for_sv), len(self.mu_for_sv)
                raise InputError(f'Dispersion template shape {sv.shape} does not match expected ({lenr, lenmu})')
            # if desired, filter template along radial direction with a savgol filter
            if dispersion.get('filter', True):
                from scipy.signal import savgol_filter
                window = dispersion.get('filter_window', 3)
                polyorder = dispersion.get('filter_order', 1)
                sv = np.array([savgol_filter(sv[:, i], window, polyorder) for i in range(sv.shape[1])]).T
        elif disp_model=='constant':
            self.r_for_sv = self.r
            self.mu_for_sv = np.linspace(0, 1)
            self.sv_rmu = np.ones((len(self.mu_for_sv), len(self.r_for_sv)))
        else:
            raise InputError(f"Bad choice '{disp_model}' for dispersion model, options are 'constant' or 'template'")

        if sv.shape[0] == len(self.r_for_sv):
            sv = sv.T

        # normalise the function based on amplitude of its monopole at large r
        sv_rmu = si.interp2d(self.r_for_sv, self.mu_for_sv, sv)
        sv_monopole = utils.multipoles_from_fn(sv_rmu, self.r_for_sv, ell=[0])
        self.sv_rmu = sv / sv_monopole['0'][-1]

    def get_interpolated_real_multipoles(self, beta=None):
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
            if beta is None:
                raise InputError('Need to supply a valid value of beta for interpolation')
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
            bias = params.get('bias', model['bias'])
            beta = params.get('beta', None)
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
            delta = _spline(r, integrated_delta(r) + r * derivative / 3, ext=3)
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
        delta = _spline(r, delta_r, ext=3)
        int_delta = _spline(r, integrated_delta_r, ext=3)

        # set the term proportional to growth rate: different in different models
        if model['matter_model'] == 'linear_bias':
            # here we want to multiply by growth rate f alone, since there is a 1/b factor already included
            # in delta and int_delta; we obtain this as beta*b (so that bias values cancel out)
            if model['realspace_ccf_from_data']:
                growth_term = params['beta'] * params.get('bias', model['bias'])
            else:
                growth_term = params['fsigma8'] / self.template_sigma8
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
                Av = params.get('Av', 0) # defaults to 0 unless sampled/set
                vr = -growth_term * r * int_delta(r) * (1 + Av * delta(r)) / (3 * self.iaH)
                # build a finer grid to better estimate derivative numerically
                rgrid = np.linspace(0.1, self.r.max(), 100)
                vr_grid = -growth_term * rgrid * int_delta(rgrid) * (1 + Av * delta(rgrid)) / (3 * self.iaH)
                dvr_interp = _spline(rgrid, np.gradient(vr_grid, rgrid), ext=3)
                dvr = dvr_interp(r)
        if model['mean_model'] == 'nonlinear':
            excursion_model = self.set_ESM_params(params, model)
            # model prediction for derivative of enclosed density wrt log scale factor
            logderiv_Delta = excursion_model.density_evolution(self.z_eff, params['b10'], params['b01'], params['Rp'],
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
                Av = params.get('Av', 0) # defaults to 0 unless sampled/set
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
            dvr_interp = _spline(rgrid, np.gradient(self.radial_velocity(rgrid), rgrid), ext=3)
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
        omm = params.get('Omega_m', 0.31)
        omk = params.get('Omega_k', 0)
        omb = params.get('Omega_b', 0.048)
        s80 = params.get('sigma_8_0', 0.81)
        h = params.get('H0', 67.5) / 100
        ns = params.get('ns', 0.96)
        mnu = params.get('mnu', 0.96)
        deltac = params.get('delta_c', 1.686)
        eisenstein_hu = model['excursion_set_options'].get('use_eisenstein_hu', False)
        accuracy = model['excursion_set_options'].get('camb_accuracy', 1)

        # Initialize
        excursion_model = get_excursion_set_model(h, omm, omb, mnu, ns, omk, self.z_eff, eisenstein_hu, accuracy)
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
        s = np.atleast_1d(s)
        mu = np.atleast_1d(mu)
        if np.ndim(s) == 2 and np.ndim(mu) == 2:
            if not s.shape == mu.shape:
                raise InputError('theory_xi: If arguments s and mu are 2D arrays they must have same shape')
            # S, Mu, X = np.meshgrid(s[0], mu[:, 0], x)
            S, Mu, X = np.meshgrid(np.unique(s), np.unique(mu), x)
        elif np.ndim(s) == 1 and np.ndim(mu) == 1:
            S, Mu, X = np.meshgrid(s, mu, x)
        else:
            raise InputError('theory_xi: arguments s and mu have incompatible dimensions')

        if self.fixed_real_input and not model['matter_model']=='linear_bias':
            # this is the only case in which the value of beta is irrelevant, so we allow it not to be provided
            beta = 0.40
        else:
            beta = params['beta']
        # following allows for differences in which combination of AP parameters are sampled
        if 'epsilon' in params:
            epsilon = params['epsilon']
            apar = params.get('alpha', 1) * epsilon**(-2/3)
            aperp = epsilon * apar
        else:
            aperp = params.get('aperp', 1)
            apar = params.get('apar', 1)
            epsilon = aperp / apar

        # --- rescale real-space functions to account for Alcock-Paczynski dilation --- #
        mu_vals = np.linspace(1e-10, 1)
        mu_integral = np.trapz(apar * np.sqrt(1 + (1 - mu_vals**2) * (epsilon**2 - 1)), mu_vals)
        reference_r = self.r
        rescaled_r = reference_r * mu_integral
        # real-space correlation
        ccf_mult = self.get_interpolated_real_multipoles(beta)
        real_multipoles  = {}
        for i, ell in enumerate(self.poles_r): 
            if model['realspace_ccf_from_data']:
                real_multipoles[f'{ell}'] = _spline(reference_r, ccf_mult[i], ext=3)
            else:
                real_multipoles[f'{ell}'] = _spline(rescaled_r, ccf_mult[i], ext=3)
        # velocity terms: note here the little hack to get an estimate at r=0 which helps control an interpolation
        # artifact that affects only the approximate Kaiser and special Euclid calculations (it is also only
        # cosmetic, but this helps produce nicer figures)
        vr, dvr = self.velocity_terms(np.append([0.01], reference_r), params, **kwargs)
        if model['matter_model'] == 'excursion_set':
            # this is a special case: as the model predicts the absolute scale of void-matter ccf we do not rescale
            # in principle, means using shape of void-matter ccf monopole to extract information; in practice this has 
            # very little information because of the additional nuisance parameters (and because realspace multipoles 
            # of xi still get rescaled)
            vr_interp = _spline(np.append([0.01], reference_r), vr, ext=3)
            dvr_interp = _spline(np.append([0.01], reference_r), dvr, ext=3)
        else:
            # rescale as normal
            vr_interp = _spline(np.append([0.01*mu_integral], rescaled_r), vr, ext=3)
            dvr_interp = _spline(np.append([0.01*mu_integral], rescaled_r), dvr, ext=3)
        if model['rsd_model'] in ['streaming', 'dispersion']:
            # we will rescale the actual dispersion function later, but scale the amplitude with apar here
            sigma_v = params.get('sigma_v', 380) * apar

        # apply AP corrections to shift input coordinates in the fiducial cosmology to those in true cosmology
        mu_s = Mu
        mu_s[Mu>0] = 1 / np.sqrt(1 + epsilon**2 * (1 / Mu[Mu>0]**2 - 1))
        s_perp = S * np.sqrt(1 - mu_s**2) * aperp
        s_par = S * mu_s * apar
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
                r_par = (s_par - v_par * self.iaH) / (1 + self.iaH * vr_interp(s) / s)
                for i in range(model.get('niter', 5)):
                    r = np.sqrt(s_perp**2 + r_par**2)
                    r_par =  (s_par - v_par * self.iaH) / (1 + self.iaH * vr_interp(r) / r)
                r = np.sqrt(s_perp**2 + r_par**2)
                mu_r = r_par / r
                # now scale the dispersion function for AP dilation and then evaluate
                sv_spl = si.RectBivariateSpline(self.r_for_sv * mu_integral, self.mu_for_sv, self.sv_rmu.T)
                sv = sigma_v * sv_spl.ev(r, mu_r)
                vel_pdf = norm.pdf(v_par, loc=0, scale=sv)
                # as we've changed variables account for this in the Jacobian
                jacobian = 1 / (1 + vr_interp(r)*self.iaH/r + self.iaH * mu_r**2 * (dvr_interp(r) - vr_interp(r)/r))

            # if the real-space CCF comes from the data, not a template, apply inverse AP corrections
            # to shift coordinates from true cosmology to the fiducial one and evaluate CCF at these adjusted positions
            if model['realspace_ccf_from_data']:
               r_par_fid  = r_par/apar
               r_perp_fid = s_perp/aperp
               r = np.sqrt(r_par_fid**2  + r_perp_fid**2)
               mu_r = r_par_fid/r 
            # build the real-space ccf at each point
            if model['assume_isotropic']:
                # following is equivalent to multiplying by 1, but more explicitly shows what is happening!
                xi_rmu =  real_multipoles['0'](r) * legendre(0)(mu_r)
            else:
                xi_rmu = np.zeros_like(r)
                for ell in self.poles_r:
                    xi_rmu = xi_rmu + real_multipoles[f'{ell}'](r) * legendre(ell)(mu_r)

            # integrate for model prediction
            xi_smu = simps((1 + xi_rmu) * jacobian * vel_pdf, x=v_par, axis=2) - 1

        elif model['rsd_model'] == 'kaiser':

            # get possible additional nuisance parameters if included, as per Hamaus et al 2020
            M = params.get('M', 1.0)
            Q = params.get('Q', 1.0)

            if model.get('kaiser_coord_shift', True):
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

            # evaluate velocity terms in the Jacobian with added nuisance parameters M and Q
            J = M*vr_interp(r)*self.iaH/r + M*Q*mu_r**2*self.iaH*(dvr_interp(r) - vr_interp(r)/r)

            # if the real-space CCF comes from the data, not a template, apply inverse AP corrections
            # to shift coordinates from true cosmology to the fiducial one and evaluate CCF at these adjusted positions
            if model['realspace_ccf_from_data']:
               r_par_fid  = r_par/apar
               r_perp_fid = s_perp/aperp
               r = np.sqrt(r_par_fid**2  + r_perp_fid**2)
               mu_r = r_par_fid/r 
            # build the real-space ccf at each point
            if model['assume_isotropic']:
                xi_rmu =  real_multipoles['0'](r) * legendre(0)(mu_r)
            else:
                xi_rmu = np.zeros_like(r)
                for ell in self.poles_r:
                    xi_rmu = xi_rmu + real_multipoles[f'{ell}'](r) * legendre(ell)(mu_r)

            # we now obtain the model without integration (ie assuming the velocity pdf is a delta function)
            if not model.get('kaiser_approximation', False):
                # use full expression for Jacobian but with added nuisance parameters M and Q
                jacobian = 1 / (1 + J)
                xi_smu = (1 + M * xi_rmu) * jacobian - 1
            else:
                # approximate Jacobian by a series expansion truncated at linear order in velocity terms (note the nuisance
                # parameters M and Q as well) and also truncate expression for xi_smu to same order
                jacobian = -J
                xi_smu = M * xi_rmu + jacobian

            # drop the unnecessary dimension
            xi_smu = xi_smu[:, :, 0]

        elif model['rsd_model'] == 'euclid_special':

            # get possible additional nuisance parameters if included, as per Hamaus et al 2020
            M = params.get('M', 1.0)
            Q = params.get('Q', 1.0)

            if model.get('kaiser_coord_shift', True):
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

            # NOTE: the new factors of 3 and 2 on the first and second terms respectively!
            J = 3*M*vr_interp(r)*self.iaH/r + 2*M*Q*mu_r**2*self.iaH*(dvr_interp(r) - vr_interp(r)/r)

            # if the real-space CCF comes from the data, not a template, apply inverse AP corrections
            # to shift coordinates from true cosmology to the fiducial one and evaluate CCF at these adjusted positions
            if model['realspace_ccf_from_data']:
               r_par_fid  = r_par/apar
               r_perp_fid = s_perp/aperp
               r = np.sqrt(r_par_fid**2  + r_perp_fid**2)
               mu_r = r_par_fid/r 
            # build the real-space ccf at each point
            if model['assume_isotropic']:
                xi_rmu =  real_multipoles['0'](r) * legendre(0)(mu_r)
            else:
                xi_rmu = np.zeros_like(r)
                for ell in self.poles_r:
                    xi_rmu = xi_rmu + real_multipoles[f'{ell}'](r) * legendre(ell)(mu_r)

            # we now obtain the model without integration (ie assuming the velocity pdf is a delta function)
            xi_smu = M * xi_rmu - J

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
        s, mu = smu(*np.meshgrid(sperp, spar))
        xi_smu = np.zeros_like(s)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                xi_smu[i, j] = self.theory_xi(s[i, j], mu[i, j], params, **kwargs)
        xi_model = si.interp2d(sperp, spar, xi_smu)
        return xi_model

    def xi_2D_from_multipoles(self, params, rmax=85, **kwargs):
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

        s = np.linspace(0.01, rmax)
        multipoles = self.theory_multipoles(s, params, poles=[0, 2, 4], **kwargs)
        for ell in [0, 2, 4]:
            multipoles[f'{ell}'] = _spline(s, multipoles[f'{ell}'])

        sperp = np.linspace(0.01, rmax)
        spar = np.linspace(-rmax, rmax)  # allow for potential odd functions
        def smu(sigma, pi):
            s = np.sqrt(sigma**2 + pi**2)
            return s, pi/s
        s, mu = smu(*np.meshgrid(sperp, spar))

        xi_2D_grid = np.zeros_like(s)
        for ell in [0, 2, 4]:
            xi_2D_grid = xi_2D_grid + multipoles[f'{ell}'](s) * legendre(ell)(mu)
        xi_2D = si.interp2d(sperp, spar, xi_2D_grid)
        return xi_2D

    def plot_model_multipoles(self, *parameters, s=None, ell=2, diff=False, ax=None, **kwargs):
        """
        Method to plot model predictions for a given multipole evaluated at different input parameter values

        Parameters
        ----------
        parameters : tuple of dicts
            Each dict in this tuple must contain a valid set of parameters to be passed to the theory call; it can
            optionally also contain a key `options` containing the optional kwargs to be passed to the theory call,
            a key `label` containing legend text for the plotted theory curve, and a key `plot_kwargs` containing other
            kwargs (color, linewidth etc) to be passed to `ax.plot`

        s : array
            Radial distance coordinate values at which to evaluate theory multipoles. If not provided, uses the
            values in :attr:`r` by default

        ell : int, default=2
            The order of Legendre multipole to plot

        diff : bool, default=False
            Whether to plot redshift-space multipole as a difference with respect to the real-space version (useful for
            visualising the monopole)

        ax : Matplotlib Axes instance, default None
            The axis to plot to (optional)

        kwargs : dict
            Optional dict which may be used to provide strings `xlabel` (default: r'$s\;[h^{-1}\mathrm{Mpc}]$') and
            `ylabel` (default: '') to label the plot axes

        Returns
        -------
        ax : Matplotlib Axes instance
        """

        ax = ax or plt.gca()
        xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else r'$s\;[h^{-1}\mathrm{Mpc}]$'
        ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
        if s is None: s = self.r
        for params in parameters:
            options = params.get('options', {})
            label = params.get('label', None)
            plot_kwargs = params.get('plot_kwargs', {})

            theory = self.theory_multipoles(s, params, poles=ell, **options)[f'{ell}']
            ind = [0, 2, 4].index(ell)
            if diff:
                refth = np.interp(s, self.r, self.get_interpolated_real_multipoles(params.get('beta', None))[ind])
            else:
                refth = np.zeros_like(theory)
            ax.plot(s, theory - refth, label=label, **plot_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plot_realspace_multipoles(self, *parameters, r=None, ell=2, ax=None, **kwargs):
        """
        Method to plot input realspace multipoles

        Parameters
        ----------
        parameters:  tuple of dicts
            Each dict corresponds to one real-space multipole plot. If :attr:`fixed_real_input` is `False`, each dict
            must contain a key `beta` with the value of the reconstruction parameter :math:`\beta`. The dict can also
            optionally contain a key `label` containing legend text for the plot, and a key `plot_kwargs` of kwargs to
            be passed to ax.plot

        r : array, default None
            Radial distance coordinate values at which to plot (interpolated) multipole. If not provided, uses the
            values in :attr:`r` instead (and then no interpolation is required)

        ell : int, default=2
            The order of Legendre multipole to plot

        ax : Matplotlib Axes instance, default None
            The axis to plot to (optional)

        kwargs : dict
            Optional dict which may be used to provide strings `xlabel` (default: r'$s\;[h^{-1}\mathrm{Mpc}]$') and
            `ylabel` (default: '') to label the plot axes

        Returns
        -------
        ax : Matplotlib Axes instance
        """

        ax = ax or plt.gca()
        xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else r'$s\;[h^{-1}\mathrm{Mpc}]$'
        ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

        # for fixed realspace input this method could be called without passing any parameters – catch this
        if self.fixed_real_input and len(parameters)==0:
            parameters = [{}]

        if r is None: r = self.r

        for params in parameters:
            label = params.get('label', None)
            plot_kwargs = params.get('plot_kwargs', {})
            ind = [0, 2, 4].index(ell)
            multipole = np.interp(r, self.r, self.get_interpolated_real_multipoles(params.get('beta', None))[ind])
            ax.plot(r, multipole, label=label, **plot_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

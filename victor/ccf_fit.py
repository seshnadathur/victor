import os
import copy
import h5py
import numpy as np
import scipy.interpolate as si
from .utils import InputError
from .ccf_model import CCFModel
import matplotlib.pyplot as plt

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

        # --- check the data and covariance matrix files --- #
        base_dir = data.get('dir', '')
        data_fn = os.path.join(base_dir, data['redshift_space_ccf'].get('data_file'))
        cov_fn = os.path.join(base_dir, data['covariance_matrix'].get('data_file'))
        for fn in [data_fn, cov_fn]:
            if not os.path.isfile(fn):
                raise InputError(f'Data file {fn} not found')

        self._load_redshiftspace_ccf(data['redshift_space_ccf'], data_fn)
        self._load_covariance_matrix(data['covariance_matrix'], cov_fn)
        # set likelihood evaluation options
        self.fit_options = {'beta_interpolation': data.get('beta_interpolation', 'datavector'),
                            'likelihood': data.get('likelihood', {'form': 'Gaussian'})}

    def _load_redshiftspace_ccf(self, ccf, input_fn):
        """Private class method to read redshift-space ccf data from file"""

        # load the data from file
        for file_format, exts in self.extensions.items():
            if any(input_fn.endswith(ext) for ext in exts):
                break
        if file_format == 'npy':
            input_data = np.load(input_fn, allow_pickle=True).item()
        elif file_format == 'hdf5':
            with h5py.File(input_fn, 'r') as f:
                input_data = {}
                for key in list(f.keys()):
                    input_data[key] = f[key][:]

        # if the file contains multiple versions of ccf information corresponding to different simulation
        # realisations, we allow the specification of which simulation number to access
        isim = ccf.get('simulation_number', None)

        self.fixed_data = not(ccf.get('reconstruction', False))
        if not self.fixed_data:
            # check if beta information is provided in file
            beta_key = ccf.get('beta_key', None)
            if beta_key and beta_key in input_data:
                self.beta_ccf = input_data[beta_key]
                if not np.all(self.beta_ccf[1:] - self.beta_ccf[:-1] > 0):
                    raise InputError('Redshift-space beta grid must be strictly monotonically increasing')
            else:
                # beta information is not provided, so default to the same beta grid as for the realspace ccf
                if self.fixed_real_input:
                    raise InputError('Reconstruction beta information required for redshift-space ccf but not found')
                else:
                    self.beta_ccf = self.beta

        # format in which ccf information is provided
        format = ccf.get('format', 'multipoles')
        # keys to identify relevant information
        ccf_keys = np.atleast_1d(ccf['ccf_keys'])
        # check the keys
        bad_keys = (format=='multipoles' and len(ccf_keys)<2) or (format=='rmu' and len(ccf_keys)!=3)
        if bad_keys: raise InputError(f'Wrong number of redshift-space ccf keys provided for format {format}')
        for key in ccf_keys:
            if not key in input_data:
                raise InputError(f'Key {key} not found in file {input_fn}')

        if format=='multipoles':
            self.s = input_data[ccf_keys[0]]
            names = ['monopole', 'quadrupole', 'hexadecapole'][:len(ccf_keys)-1]
            self.poles_s = np.atleast_1d([0, 2, 4][:len(ccf_keys)-1])
            self.redshift_multipoles = {}
            for i, name in enumerate(names):
                if isim is None:
                    self.redshift_multipoles[f'{self.poles_s[i]}'] = input_data[ccf_keys[i+1]]
                elif isinstance(isim, int):
                    self.redshift_multipoles[f'{self.poles_s[i]}'] = input_data[ccf_keys[i+1]][isim]
                else:
                    raise InputError('If provided, simulation_number must be an integer')
            # sense check the data
            for i, ell in enumerate(self.poles_s):
                mult_shape = self.redshift_multipoles[f'{ell}'].shape
                if self.fixed_data:
                    if not mult_shape==self.s.shape:
                        raise InputError(f'Shape of redshift ccf {name[i]} is {mult_shape}, expected {self.r.shape}')
                else:
                    x, y = len(self.beta_ccf), len(self.s)
                    if not mult_shape==(x, y):
                        raise InputError(f'Shape of redshift ccf {name[i]} is {mult_shape}, expected ({x}, {y})')
        else:
            raise InputError('Currently only multipole format is supported for redshift-space ccf data and covmat')

        del input_data

    def _load_covariance_matrix(self, covariance, input_fn):
        """Private class method to load covariance matrix information from file"""

        # load the data from file
        for file_format, exts in self.extensions.items():
            if any(input_fn.endswith(ext) for ext in exts):
                break
        if file_format == 'npy':
            input_data = np.load(input_fn, allow_pickle=True).item()
        elif file_format == 'hdf5':
            with h5py.File(input_fn, 'r') as f:
                input_data = {}
                for key in list(f.keys()):
                    input_data[key] = f[key][:]

        if not self.fixed_data:
            # redshift-space ccf has reconstruction dependence but covmat may or may not
            self.fixed_covmat = covariance.get('fixed_beta', True)
            if not self.fixed_covmat:
                # check if beta information is provided in file
                beta_key = covariance.get('beta_key', None)
                if beta_key and beta_key in input_data:
                    self.beta_covmat = input_data[beta_key]
                    if not np.all(self.beta_covmat[1:] - self.beta_covmat[:-1] > 0):
                        raise InputError('Covariance beta grid must be strictly monotonically increasing')
                else:
                    # beta information is not provided, so default to the same beta grid as for the redshift-space ccf
                    self.beta_covmat = self.beta_ccf
        else:
            # if the redshift-space ccf has no reconstruction dependence then the covariance matrix also cannot
            self.fixed_covmat = True

        cov_key = covariance['cov_key']
        if not cov_key in input_data:
            raise InputError(f'Key {cov_key} not found in file {input_fn}')
        covmat = input_data[cov_key]

        # sense check the data
        if self.fixed_covmat:
            if not (covmat.shape == ((len(self.s) * len(self.poles_s), len(self.s) * len(self.poles_s)))):
                try:
                    covmat = covmat[:len(self.s) * len(self.poles_s), :len(self.s) * len(self.poles_s)]
                    print('Warning, (fixed) covariance matrix was cut down to match the number of selected multipoles')
                except:
                    raise InputError('Unexpected shape of (fixed) covariance matrix')
        else:
            if not (covmat.shape == ((len(self.beta_covmat), len(self.s) * len(self.poles_s), len(self.s) * len(self.poles_s)))):
                try:
                    covmat = covmat[:, :len(self.s) * len(self.poles_s), :len(self.s) * len(self.poles_s)]
                    print('Warning, (beta-varying) covariance matrix was cut down to match the number of selected multipoles')
                except:
                    raise InputError('Unexpected shape of (beta-varying) covariance matrix')

        self.covmat = covmat
        self.icov = np.linalg.inv(self.covmat)

        del input_data

    def get_interpolated_redshift_multipoles(self, beta=None):
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
            if beta is None:
                raise InputError('Need to supply a valid value of beta for interpolation')
            multipoles = np.empty((len(self.poles_s), len(self.beta_ccf), len(self.s)))
            for i, ell in enumerate(self.poles_s):
                multipoles[i] = self.redshift_multipoles[f'{ell}']
            return np.atleast_2d(si.PchipInterpolator(self.beta_ccf, multipoles, axis=1)(beta))

    def get_interpolated_covariance(self, beta=None):
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
            if beta is None:
                raise InputError('Need to supply a valid value of beta for interpolation')
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

    def get_interpolated_precision(self, beta=None):
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
            if beta is None:
                raise InputError('Need to supply a valid value of beta for interpolation')
            if beta < self.beta_covmat.min(): return self.icov[0]
            if beta > self.beta_covmat.max(): return self.icov[-1]

            if beta in self.beta_covmat:
                return self.icov[np.where(self.beta_covmat==beta)[0][0]]

            # if nothing else, linearly inteprolate between two bracketing entries
            lowind = np.where(self.beta_covmat < beta)[0][-1]
            highind = np.where(self.beta_covmat >= beta)[0][-1]
            t = (beta - self.beta_covmat[lowind]) / (self.beta_covmat[highind] - self.beta_covmat[lowind])
            return (1 - t) * self.icov[lowind] + t * self.icov[highind]

    def correlation_matrix(self, beta=None):
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

        covmat = self.get_interpolated_covariance(beta)
        corrmat = np.zeros_like(covmat)
        diagonals = np.sqrt(np.diag(covmat))
        for i in range(corrmat.shape[0]):
            for j in range(corrmat.shape[1]):
                if not (diagonals[i] * diagonals[j] == 0):
                    corrmat[i, j] = covmat[i, j] / (diagonals[i] * diagonals[j])
        return corrmat

    def diagonal_errors(self, beta=None):
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

    def multipole_datavector(self, beta=None):
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
        data_vector = self.multipole_datavector(params.get('beta', None))
        cov = self.get_interpolated_covariance(params.get('beta', None))
        icov = self.get_interpolated_precision(params.get('beta', None))

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
            like_factor = (low_like_factor, high_like_factor)
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

    def plot_multipole_comparison(self, *parameters, s=None, ell=2, diff=False, ax=None, **kwargs):
        r"""
        Method to plot data vs model predictions for a given multipole, with models possibly evaluated at different
        input parameter values

        Parameters
        ----------
        parameters : tuple of dicts
            Each dict in this tuple must contain a valid set of parameters to be passed to the likelihood call; it can
            optionally also contain a key `options` containing the optional kwargs to be passed to the theory call,
            a key `label` containing legend text for the plotted theory curve, and a key `plot_kwargs` containing other
            kwargs (color, linewidth etc) to be passed to `ax.plot`

        s : array
            Radial distance coordinate values at which to evaluate theory multipoles. If not provided, uses the
            values in :attr:`s` by default

        ell : int, default=2
            The order of Legendre multipole to plot

        diff : bool, default=False
            Whether to plot redshift-space multipole as a difference with respect to the real-space version (useful for
            visualising the monopole)

        ax : Matplotlib Axes instance, default None
            The axis to plot to (optional)

        kwargs : dict
            Optional dict which may be used to provide strings `xlabel` (default: r'$s\;[h^{-1}\mathrm{Mpc}]$') and
            `ylabel` (default: '') to label the plot axes, and bool `chi2` (default False) indicating whether to
            calculate the chi-square for the fit and add it to the plot legend (appended to other labels if provided)

        Returns
        -------
        ax : Matplotlib Axes instance
        """

        ax = ax or plt.gca()
        xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else r'$s\;[h^{-1}\mathrm{Mpc}]$'
        ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
        calculate_chi2 = kwargs['chi2'] if 'chi2' in kwargs else False

        # first determine how many versions of the data we need to plot (data points may change with beta)
        if self.fixed_data:
            dv = 1
        else:
            for i, params in enumerate(parameters):
                if i==0:
                    betaref = [params['beta']]
                elif params['beta'] not in betaref:
                    betaref.append(params['beta'])
            dv = len(betaref)

        if s is None: s = self.s

        # now plot
        for i, params in enumerate(parameters):
            options = params.get('options', {})
            label = params.get('label', None)

            # retrieve plot color option if provided
            plot_kwargs = params.get('plot_kwargs', {})
            if 'c' in plot_kwargs:
                color = plot_kwargs['c']
            elif 'color' in plot_kwargs:
                color = plot_kwargs['color']
            else:
                color = f'C{i}'

            # get the chi2 value if required and append to plot label
            if calculate_chi2:
                chi2, _ = self.chi_squared(params, **options)
                if label is None:
                    label = f'$\chi^2={chi2:.2f}$'
                else:
                    label += f' $\chi^2={chi2:.2f}$'

            theory = self.theory_multipoles(s, params, poles=ell, **options)[f'{ell}']
            ind = [0, 2, 4].index(ell)
            errs = self.diagonal_errors(params.get('beta', None))[ind]
            data = self.get_interpolated_redshift_multipoles(params.get('beta', None))[ind]
            if diff:
                real_mult = self.get_interpolated_real_multipoles(params.get('beta', None))[ind]
                refth = np.interp(s, self.r, real_mult)
                refdata = np.interp(self.s, self.r, real_mult)
            else:
                refth = np.zeros_like(theory)
                refdata = np.zeros_like(data)
            if dv==1:
                if i==0:
                    ax.errorbar(self.s, data - refdata, yerr=errs, fmt='.', markersize='8', c='k',
                                label=kwargs.get('data_label', None))
            else:
                ax.errorbar(self.s, data - refdata, yerr=errs, fmt='.', markersize='8', c=color)
            ax.plot(s, theory - refth, label=label, **plot_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

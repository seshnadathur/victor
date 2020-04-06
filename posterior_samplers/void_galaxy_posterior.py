import os
import sys
import numpy as np
from utilities.utilities import UtilMethods, Cosmology, NonFlatBackground
from scipy.integrate import quad
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d
from utilities.multipoles import correlation_multipoles


class VoidGalaxyPosterior:
    """
    Class to evaluate the likelihood for joint RSD-AP fits to void-galaxy cross-correlation data
    """

    def __init__(self, parms):
        """
        Initialize instance

        :param parms: a module containing all the parameter options specified in the void-galaxy_params.py file

        """

        self.root = parms.root
        self.output_folder = parms.output_folder

        # fiducial cosmology values for converting velocities into distances
        if parms.fiducial_omega_m + parms.fiducial_omega_l == 1:
            # flat background case
            cosmo = Cosmology(omega_m=parms.fiducial_omega_m)
        else:
            # non-flat background
            cosmo = NonFlatBackground(omega_m=parms.fiducial_omega_m, omega_l=parms.fiducial_omega_l)
        self.iaH = (1 + parms.eff_z) / cosmo.get_ez(parms.eff_z)

        # effective redshift
        self.eff_z = parms.eff_z
        # covmat details
        self.nmocks_covmat = parms.nmocks_covmat

        self.theory_model = parms.theory_model
        # set various data values that depend on theory model
        if self.theory_model == 3:
            # Model 3 has the simplest data vector and theory options
            self.use_recon = False
            if parms.reconstructed_data:
                print('Theory model 2 does not require reconstruction!' )
            # load the multipole data vector
            if not os.access(parms.s_space_multipole_file, os.F_OK):
                sys.exit('s_space_multipole_file %s not found, aborting' % parms.s_space_multipole_file)
            multipole_data = np.load(parms.s_space_multipole_file, allow_pickle=True).item()
            self.data_rvals = multipole_data['rvals']
            self.nrbins = len(self.data_rvals)
            if multipole_data['multipoles'].shape == (2 * self.nrbins,):
                self.s_multipoles = multipole_data['multipoles']
            else:
                sys.exit('Wrong shape for s-space multipole data array')
            # load the covariance matrix
            if not os.access(parms.covmat_file, os.F_OK):
                sys.exit('covmat_file %s not found, aborting' % parms.covmat_file)
            self.covmat = np.load(parms.covmat_file)
            if not self.covmat.shape == (2 * self.nrbins, 2 * self.nrbins):
                sys.exit('Covariance matrix shape does not match that of data vector')
            # evaluate the inverse covariance matrix for later use
            self.icov = np.linalg.inv(self.covmat)

            # just for concreteness ...
            self.assume_lin_bias = True

            # Model 3 has only one free parameter: beta
            self.ndim = 1

            # pre-calculate these quantities for later use
            self.xi_s_mono = InterpolatedUnivariateSpline(self.data_rvals, self.s_multipoles[:len(self.data_rvals)])
            integral = np.zeros_like(self.data_rvals)
            for i in range(len(integral)):
                integral[i] = quad(lambda x: self.xi_s_mono(x) * x**2, 0, self.data_rvals[i], full_output=1)[0]
            self.int_xi_s_mono = InterpolatedUnivariateSpline(self.data_rvals, 3*integral/self.data_rvals**3, ext=3)

            self.beta_prior_range = parms.beta_prior_range
        else:  # ie Models 1 and 2 - these both need real-space information, only the theory differs
            self.use_recon = parms.reconstructed_data

            # load the redshift-space multipole data vector
            if not os.access(parms.s_space_multipole_file, os.F_OK):
                sys.exit('s_space_multipole_file %s not found, aborting' % parms.s_space_multipole_file)
            s_multipole_data = np.load(parms.s_space_multipole_file, allow_pickle=True).item()
            self.data_rvals = s_multipole_data['rvals']
            self.nrbins = len(self.data_rvals)
            self.s_multipoles = s_multipole_data['multipoles']

            # load the covariance matrix
            if not os.access(parms.covmat_file, os.F_OK):
                sys.exit('covmat_file %s not found, aborting' % parms.covmat_file)
            self.covmat = np.load(parms.covmat_file)

            # options for the DM density profile
            self.assume_lin_bias = parms.assume_lin_bias
            if not self.assume_lin_bias:
                delta_data = np.load(parms.delta_file, allow_pickle=True).item()
                if not np.all(delta_data['rvals'] == self.data_rvals):
                    sys.exit('Sorry, the mutipoles and the DM density information need to have been ' +
                             'provided in the same radial bins. Aborting')
                # ext=3 below to control extrapolation beyond the r range to sensible values
                self.delta_r = InterpolatedUnivariateSpline(self.data_rvals, delta_data['delta'], ext=3)
                # integrate for the cumulative density profile
                integral = np.zeros_like(self.data_rvals)
                for i in range(len(integral)):
                    integral[i] = quad(lambda x: self.delta_r(x) * x**2, 0, self.data_rvals[i], full_output=1)[0]
                # following function is the Delta(r) function appearing in the theory model
                self.int_delta_r = InterpolatedUnivariateSpline(self.data_rvals, 3*integral/self.data_rvals**3, ext=3)
                self.sig8_norm = parms.sig8_norm  # only needed for Model 1 when not assuming linear bias in voids

                # NOTE: we are assuming that the DM density profile monopole does not change with the reconstruction so
                # doesn't depend on beta, which is actually true: the differences in delta(r) produced by different recon
                # beta values have are been checked in simulation and affect theory predictions by << statistical
                # uncertainty in current data (BUT, might need to revisit this for DESI/Euclid data)

                # if not assuming linear bias, there will be 4 free parameters: fs8, bs8, sigma_v and epsilon
                self.ndim = 4
            else:
                # if assuming linear bias, there will be only 3 free parameters: beta, sigma_v and epsilon
                self.ndim = 3

            # both these models use velocity dispersion
            if parms.constant_dispersion:
                self.sv_norm_func = InterpolatedUnivariateSpline(self.data_rvals, np.ones_like(self.data_rvals), ext=3)
            else:
                if not os.access(parms.dispersion_file, os.F_OK):
                    sys.exit('dispersion_file %s not found, aborting' % parms.dispersion_file)
                sv_data = np.load(parms.dispersion_file, allow_pickle=True).item()
                if not np.all(sv_data['rvals'] == self.data_rvals):
                    sys.exit('Sorry, the mutipoles and the velocity dispersion information need to have been ' +
                             'provided in the same radial bins. Aborting')
                # measured dispersion values are often noisy, so lightly smooth using a Savitsky-Golay filter
                normed_sv = savgol_filter(sv_data['sigma_v_los'] / sv_data['sigma_v_los'][-1], 3, 1)
                self.sv_norm_func = InterpolatedUnivariateSpline(self.data_rvals, normed_sv, ext=3)

            if self.use_recon:
                # get the grid of beta values for which reconstruction was performed
                if not os.access(parms.beta_grid_file, os.F_OK):
                    sys.exit('beta_grid_file %s not found, aborting' % parms.beta_grid_file)
                self.beta_grid = np.load(parms.beta_grid_file)
                beta_min = max(parms.beta_prior_range[0], np.min(self.beta_grid))
                beta_max = min(parms.beta_prior_range[1], np.max(self.beta_grid))
                self.beta_prior_range = [beta_min, beta_max]
                self.nbetabins = len(self.beta_grid)
                # load the real-space multipole data vector
                if not os.access(parms.r_space_multipole_file, os.F_OK):
                    sys.exit('r_space_multipole_file %s not found, aborting' % parms.r_space_multipole_file)
                r_multipole_data = np.load(parms.r_space_multipole_file, allow_pickle=True).item()
                # check number of r bins is consistent
                if not len(r_multipole_data['rvals']) == self.nrbins:
                    sys.exit('Why were real-space and redshift-space multipoles not measured using the same ' +
                             'radial binning, you monster? Aborting ...')
                self.r_multipoles = r_multipole_data['multipoles']
                # check the shapes wrt beta
                if not (self.s_multipoles.shape[0] == self.nbetabins and self.r_multipoles.shape[0] == self.nbetabins):
                    sys.exit('Multipoles not provided on the same grid as beta, cannot interpolate. Aborting')
                self.fixed_covmat = parms.fixed_covmat
                if self.fixed_covmat:
                    if not self.covmat.shape == (2 * self.nrbins, 2 * self.nrbins):
                        sys.exit('Covariance matrix is the wrong shape. Aborting')
                    self.icov = np.linalg.inv(self.covmat)
                else:
                    if not self.covmat.shape == (self.nbetabins, 2 * self.nrbins, 2 * self.nrbins):
                        sys.exit('Covariance matrix wrong shape or not provided on same grid as beta. Aborting')
                # and check the multipole vector lengths
                if not (self.s_multipoles.shape[1] == 2*self.nrbins and self.r_multipoles.shape[1] == 2*self.nrbins):
                    sys.exit('Why do the multipole vectors not match the radial bins you provided? Aborting')
            else:
                # Model 1 + no recon is for special case where real-space galaxy positions are known exactly, eg because
                # we have simulation information, so there is no grid of beta
                if not os.access(parms.r_space_multipole_file, os.F_OK):
                    sys.exit('r_space_multipole_file %s not found, aborting' % parms.r_space_multipole_file)
                r_multipole_data = np.load(parms.r_space_multipole_file, allow_pickle=True).item()
                if not len(r_multipole_data['rvals']) == self.nrbins:
                    sys.exit('Why were real-space and redshift-space multipoles not measured using the same ' +
                             'radial binning, you monster? Aborting ...')
                self.r_multipoles = r_multipole_data['multipoles']
                if not self.covmat.shape == (2 * self.nrbins, 2 * self.nrbins):
                    sys.exit('Covariance matrix is the wrong shape. Aborting')
                # invert covariance matrix
                self.icov = np.linalg.inv(self.covmat)

                self.beta_prior_range = parms.beta_prior_range


    def interpolated_data_vector(self, beta):
        """
        For case where data vector depends on beta due to reconstruction, return the interpolated redshift-space
        data vector at given specific value of beta
        """

        if self.use_recon:
            # build the interpolation for real-space multipoles over the grid of beta values
            s_multipoles = np.zeros(2 * self.nrbins)
            try:
                for i in range(2 * self.nrbins):
                    interpolator = InterpolatedUnivariateSpline(self.beta_grid, self.s_multipoles[:, i], ext=2)
                    s_multipoles[i] = interpolator(beta)
            except ValueError:
                return np.zeros_like(s)
        else:
            s_multipoles = self.s_multipoles

        return s_multipoles


    def interpolated_covmat(self, beta):
        """
        For case where covariance matrix depends on beta due to reconstruction, return the interpolated covariance
        matrix at given specific value of beta
        """

        if self.use_recon and not self.fixed_covmat:
            # build the interpolation for real-space multipoles over the grid of beta values
            cov = np.empty_like(self.covmat[0])
            try:
                for i in range(self.covmat.shape[1]):
                    for j in range(self.covmat.shape[2]):
                        interpolator = InterpolatedUnivariateSpline(self.beta_grid, self.covmat[:, i, j], ext=2)
                        cov[i, j] = interpolator(beta)
            except ValueError:
                return np.zeros_like(self.covmat[0])
        else:
            cov = self.covmat

        return cov


    def interpolated_corrmat(self, beta=0.41):
        """
        For case where covariance matrix depends on beta due to reconstruction, return the interpolated correlation
        matrix at given specific value of beta
        """

        if self.use_recon and not self.fixed_covmat:
            # build the interpolation for real-space multipoles over the grid of beta values
            cov = np.empty_like(self.covmat[0])
            try:
                for i in range(self.covmat.shape[1]):
                    for j in range(self.covmat.shape[2]):
                        interpolator = InterpolatedUnivariateSpline(self.beta_grid, self.covmat[:, i, j], ext=2)
                        cov[i, j] = interpolator(beta)
            except ValueError:
                return np.zeros_like(self.covmat[0])
        else:
            cov = self.covmat

        corrmat = np.zeros_like(cov)
        diagonals = np.sqrt(np.diag(cov))
        for i in range(corrmat.shape[0]):
            for j in range(corrmat.shape[1]):
                if not (diagonals[i] * diagonals[j] == 0):
                    corrmat[i, j] = cov[i, j] / (diagonals[i] * diagonals[j])

        return corrmat


    def interpolated_r_multipoles(self, beta):
        """
        For case where the real-space void-galaxy multipoles depend on beta due to reconstruction, return the
        interpolated multipole vector at given specific value of beta
        """

        if self.use_recon:
            # build the interpolation for real-space multipoles over the grid of beta values
            r_multipoles = np.zeros(2 * self.nrbins)
            try:
                for i in range(2 * self.nrbins):
                    interpolator = InterpolatedUnivariateSpline(self.beta_grid, self.r_multipoles[:, i], ext=2)
                    r_multipoles[i] = interpolator(beta)
            except ValueError:
                return np.zeros_like(s)
        else:
            print("Mistaken call to interpolated_r_multipoles: your real-space multipoles do not depend on beta!")
            r_multipoles = self.r_multipoles

        return r_multipoles


    def model1_theory_without_lin_bias(self, fs8, bs8, sigma_v, alpha_perp, alpha_par, s):
        """
        Method to calculate theoretical monopole and quadrupole of redshift-space void-galaxy cross-correlation in Model 1
        """

        beta = fs8 / bs8
        scaled_fs8 = fs8 / self.sig8_norm

        if self.use_recon:
            r_multipoles = self.interpolated_r_multipoles(beta)
        else:  # put this here just to avoid error message when call is intentional
            r_multipoles = self.r_multipoles
        # now build interpolation over r
        xi_r = InterpolatedUnivariateSpline(self.data_rvals, r_multipoles[:self.nrbins], ext=3)

        # Now we rescale the input monopole functions - xi_r(r), delta(r), Delta(r) and sigma_v(r) – to account for
        # the Alcock-Paczynski alpha shifts. To do this we rescale the argument of the functions according to eq (12)
        # of 1904.01030, and then re-interpolate the values using this new argument
        mu = np.linspace(0, 1, 100)
        rescaled_r = np.zeros_like(self.data_rvals)
        for i, r in enumerate(self.data_rvals):
            sqrt_term = np.sqrt(1 + (1 - mu**2) * ((alpha_perp / alpha_par)**2 - 1))
            rescaled_r[i] = np.trapz((r * alpha_par) * sqrt_term, mu)
        # NOTE: removed a hack to control extrapolation here, because I think it is unnecessary – check?
        rescaled_xi_r = InterpolatedUnivariateSpline(rescaled_r, xi_r(self.data_rvals), ext=3)
        rescaled_delta_r = InterpolatedUnivariateSpline(rescaled_r, self.delta_r(self.data_rvals), ext=3)
        rescaled_int_delta_r = InterpolatedUnivariateSpline(rescaled_r, self.int_delta_r(self.data_rvals), ext=3)
        rescaled_sv_norm_func = InterpolatedUnivariateSpline(rescaled_r, self.sv_norm_func(self.data_rvals), ext=3)

        # and rescale the normalization of the sigma_v(r) function
        sigma_v = alpha_par * sigma_v

        # now we calculate the model xi(s, mu) on a grid of s and mu values
        # here the argument s refers to the observed radial separation, and therefore differs from the "true" s by the
        # AP factors
        mu_grid = np.linspace(0, 1, 51)  # 51 is a compromise, should check this
        s_grid = s
        S, Mu = np.meshgrid(s_grid, mu_grid)
        true_sperp = S * np.sqrt(1 - Mu**2) * alpha_perp
        true_spar = S * Mu * alpha_par
        true_s = np.sqrt(true_spar**2 + true_sperp**2)
        true_mu = true_spar / true_s

        # r_par is the parallel separation in real-space; this is the value for coherent outflow with zero vel. dispersion
        r_par = true_spar + true_s * scaled_fs8 * rescaled_int_delta_r(true_s) * true_mu / 3

        # now integrate over y at each (s, mu) combination
        xi_model_grid = np.zeros_like(S)
        for i in range(xi_model_grid.shape[0]):
            for j in range(xi_model_grid.shape[1]):
                # express the velocity dispersion in terms of distances
                sy_central = sigma_v * rescaled_sv_norm_func(np.sqrt(true_sperp[i, j]**2 + r_par[i, j]**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5* sy_central, 100)
                # the value of r_par including dispersion
                rpary = r_par[i, j] - y
                r = np.sqrt(true_sperp[i, j]**2 + rpary**2)
                sy = sigma_v * rescaled_sv_norm_func(r) * self.iaH
                integrand = (1 + rescaled_xi_r(r)) * \
                            (1 + (scaled_fs8 * rescaled_int_delta_r(r) / 3 - y * true_mu[i, j] / r)
                             * (1 - true_mu[i, j]**2) +
                             scaled_fs8 * (rescaled_delta_r(r) - 2 * rescaled_int_delta_r(r) / 3) * true_mu[i, j]**2)
                integrand = integrand * np.exp(- (y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                # print(np.trapz(integrand, y) - 1)
                xi_model_grid[i, j] = np.trapz(integrand, y) - 1

        # now build the true model by interpolating over this grid
        xi_model = interp2d(s_grid, mu_grid, xi_model_grid, kind='cubic')

        # and get the multipoles
        theory_multipoles = np.zeros(2 * len(s))
        theory_multipoles[:len(s)] = correlation_multipoles(xi_model, s, ell=0)
        theory_multipoles[len(s):] = correlation_multipoles(xi_model, s, ell=2)

        return theory_multipoles

    def model1_theory_with_lin_bias(self, beta, sigma_v, alpha_perp, alpha_par, s):
        """
        Method to calculate theoretical monopole and quadrupole of redshift-space void-galaxy cross-correlation
        in Model 1 + with assumption of constant linear galaxy bias within voids
        """

        if self.use_recon:
            r_multipoles = self.interpolated_r_multipoles(beta)
        else:  # put this here just to avoid error message when call is intentional
            r_multipoles = self.r_multipoles
        # now build interpolation over r
        xi_r = InterpolatedUnivariateSpline(self.data_rvals, r_multipoles[:self.nrbins], ext=3)

        # Now we rescale the input monopole functions - xi_r(r), and sigma_v(r) – to account for
        # the Alcock-Paczynski alpha shifts. To do this we rescale the argument of the functions according to eq (12)
        # of 1904.01030, and then re-interpolate the values using this new argument
        mu = np.linspace(0, 1, 100)
        rescaled_r = np.zeros_like(self.data_rvals)
        for i, r in enumerate(self.data_rvals):
            sqrt_term = np.sqrt(1 + (1 - mu**2) * ((alpha_perp / alpha_par)**2 - 1))
            rescaled_r[i] = np.trapz((r * alpha_par) * sqrt_term, mu)
        rescaled_xi_r = InterpolatedUnivariateSpline(rescaled_r, xi_r(self.data_rvals), ext=3)
        rescaled_sv_norm_func = InterpolatedUnivariateSpline(rescaled_r, self.sv_norm_func(self.data_rvals), ext=3)
        # integrate the rescaled xi_vg monopole
        integral = np.zeros_like(rescaled_r)
        for i in range(len(integral)):
            integral[i] = quad(lambda x: rescaled_xi_r(x) * x**2, 0, rescaled_r[i], full_output=1)[0]
        rescaled_int_xi_r = InterpolatedUnivariateSpline(rescaled_r, 3*integral/rescaled_r**3, ext=3)
        # and rescale the normalization of the sigma_v(r) function
        sigma_v = alpha_par * sigma_v

        # now we calculate the model xi(s, mu) on a grid of s and mu values
        # here the argument s refers to the observed radial separation, and therefore differs from the
        # "true" s by the AP factors
        mu_grid = np.linspace(0, 1, 51)  # 51 is a compromise, should check this
        s_grid = s
        S, Mu = np.meshgrid(s_grid, mu_grid)
        true_sperp = S * np.sqrt(1 - Mu**2) * alpha_perp
        true_spar = S * Mu * alpha_par
        true_s = np.sqrt(true_spar**2 + true_sperp**2)
        true_mu = true_spar / true_s

        # r_par is the parallel separation in real-space; this is the value for coherent outflow with zero
        # velocity dispersion
        r_par = true_spar + true_s * beta * rescaled_int_xi_r(true_s) * true_mu / 3

        # now integrate over y at each (s, mu) combination
        xi_model_grid = np.zeros_like(S)
        for i in range(xi_model_grid.shape[0]):
            for j in range(xi_model_grid.shape[1]):
                # express the velocity dispersion in terms of distances
                sy_central = sigma_v * rescaled_sv_norm_func(np.sqrt(true_sperp[i, j]**2 + r_par[i, j]**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5* sy_central, 100)
                # the value of r_par including dispersion
                rpary = r_par[i, j] - y
                r = np.sqrt(true_sperp[i, j]**2 + rpary**2)
                sy = sigma_v * rescaled_sv_norm_func(r) * self.iaH
                integrand = (1 + rescaled_xi_r(r)) * \
                            (1 + (beta * rescaled_int_xi_r(r) / 3 - y * true_mu[i, j] / r)
                             * (1 - true_mu[i, j]**2) +
                             beta * (rescaled_xi_r(r) - 2 * rescaled_int_xi_r(r) / 3) * true_mu[i, j]**2)
                integrand = integrand * np.exp(- (y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                xi_model_grid[i, j] = np.trapz(integrand, y) - 1

        # now build the true model by interpolating over this grid
        xi_model = interp2d(s_grid, mu_grid, xi_model_grid, kind='cubic')

        # and get the multipoles
        theory_multipoles = np.zeros(2 * len(s))
        theory_multipoles[:len(s)] = correlation_multipoles(xi_model, s, ell=0)
        theory_multipoles[len(s):] = correlation_multipoles(xi_model, s, ell=2)

        return theory_multipoles

    def model2_theory_without_lin_bias(self, fs8, bs8, sigma_v, alpha_perp, alpha_par, s):
        """
        Method to calculate theoretical monopole and quadrupole of redshift-space void-galaxy cross-correlation in Model 2
        """

        beta = fs8 / bs8
        scaled_fs8 = fs8 / self.sig8_norm

        if self.use_recon:
            r_multipoles = self.interpolated_r_multipoles(beta)
        else:  # put this here just to avoid error message when call is intentional
            r_multipoles = self.r_multipoles
        # now build interpolation over r
        xi_r = InterpolatedUnivariateSpline(self.data_rvals, r_multipoles[:self.nrbins], ext=3)

        # Now we rescale the input monopole functions - xi_r(r), delta(r), Delta(r) and sigma_v(r) – to account for
        # the Alcock-Paczynski alpha shifts. To do this we rescale the argument of the functions according to eq (12)
        # of 1904.01030, and then re-interpolate the values using this new argument
        mu = np.linspace(0, 1, 100)
        rescaled_r = np.zeros_like(self.data_rvals)
        for i, r in enumerate(self.data_rvals):
            sqrt_term = np.sqrt(1 + (1 - mu**2) * ((alpha_perp / alpha_par)**2 - 1))
            rescaled_r[i] = np.trapz((r * alpha_par) * sqrt_term, mu)
        rescaled_xi_r = InterpolatedUnivariateSpline(rescaled_r, xi_r(self.data_rvals), ext=3)
        rescaled_int_delta_r = InterpolatedUnivariateSpline(rescaled_r, self.int_delta_r(self.data_rvals), ext=3)
        rescaled_sv_norm_func = InterpolatedUnivariateSpline(rescaled_r, self.sv_norm_func(self.data_rvals), ext=3)

        # and rescale the normalization of the sigma_v(r) function
        sigma_v = alpha_par * sigma_v

        # now we calculate the model xi(s, mu) on a grid of s and mu values
        # here the argument s refers to the observed radial separation, and therefore differs from the "true" s by the
        # AP factors
        mu_grid = np.linspace(0, 1, 51)  # 51 is a compromise, should check this
        s_grid = s
        S, Mu = np.meshgrid(s_grid, mu_grid)
        true_sperp = S * np.sqrt(1 - Mu**2) * alpha_perp
        true_spar = S * Mu * alpha_par
        true_s = np.sqrt(true_spar**2 + true_sperp**2)
        true_mu = true_spar / true_s

        # a major difference between model 1 and model 2 is that model 2 neglects the shift in the coordinates themselves
        # so compare this next line to the equivalent in model 1 functions
        r_par = true_spar

        # now integrate over y at each (s, mu) combination
        xi_model_grid = np.zeros_like(S)
        for i in range(xi_model_grid.shape[0]):
            for j in range(xi_model_grid.shape[1]):
                # component of coherent outflow velocity along the line-of-sight
                vel_par = -scaled_fs8 * true_s[i, j] * rescaled_int_delta_r(true_s[i, j]) / (3 * self.iaH) * true_mu[i, j]

                # array of velocity values around this mean
                sv_central = sigma_v * rescaled_sv_norm_func(true_s[i, j])
                v = vel_par + np.linspace(-5 * sv_central, 5* sv_central, 100)

                # the value of r_par and r accounting for the velocity shift
                rpary = r_par[i, j] - v * self.iaH
                r = np.sqrt(true_sperp[i, j]**2 + rpary**2)
                # (note again that s and r are interchangeable in this model)

                # the value of sigma_v(r)
                sv = sigma_v * rescaled_sv_norm_func(r)

                # and the value of the coherent radial velocity component v(r) times mu
                vel_r_mu = -scaled_fs8 * r * rescaled_int_delta_r(r) / (3 * self.iaH) * true_mu[i, j]

                # now the integral
                integrand = (1 + rescaled_xi_r(r)) * np.exp(-0.5 * ((v - vel_r_mu) / sv)**2)
                integrand = integrand / (np.sqrt(2 * np.pi) * sv)

                xi_model_grid[i, j] = np.trapz(integrand, v) - 1

        # now build the true model by interpolating over this grid
        xi_model = interp2d(s_grid, mu_grid, xi_model_grid, kind='cubic')

        # and get the multipoles
        theory_multipoles = np.zeros(2 * len(s))
        theory_multipoles[:len(s)] = correlation_multipoles(xi_model, s, ell=0)
        theory_multipoles[len(s):] = correlation_multipoles(xi_model, s, ell=2)

        return theory_multipoles


    def model2_theory_with_lin_bias(self, beta, sigma_v, alpha_perp, alpha_par, s):
        """
        Method to calculate theoretical monopole and quadrupole of redshift-space void-galaxy cross-correlation in Model 2
        """

        if self.use_recon:
            r_multipoles = self.interpolated_r_multipoles(beta)
        else:  # put this here just to avoid error message when call is intentional
            r_multipoles = self.r_multipoles
        # now build interpolation over r
        xi_r = InterpolatedUnivariateSpline(self.data_rvals, r_multipoles[:self.nrbins], ext=3)

        # Now we rescale the input monopole functions - xi_r(r), delta(r), Delta(r) and sigma_v(r) – to account for
        # the Alcock-Paczynski alpha shifts. To do this we rescale the argument of the functions according to eq (12)
        # of 1904.01030, and then re-interpolate the values using this new argument
        mu = np.linspace(0, 1, 100)
        rescaled_r = np.zeros_like(self.data_rvals)
        for i, r in enumerate(self.data_rvals):
            sqrt_term = np.sqrt(1 + (1 - mu**2) * ((alpha_perp / alpha_par)**2 - 1))
            rescaled_r[i] = np.trapz((r * alpha_par) * sqrt_term, mu)
        rescaled_xi_r = InterpolatedUnivariateSpline(rescaled_r, xi_r(self.data_rvals), ext=3)
        rescaled_sv_norm_func = InterpolatedUnivariateSpline(rescaled_r, self.sv_norm_func(self.data_rvals), ext=3)
        # integrate the rescaled xi_vg monopole
        integral = np.zeros_like(rescaled_r)
        for i in range(len(integral)):
            integral[i] = quad(lambda x: rescaled_xi_r(x) * x**2, 0, rescaled_r[i], full_output=1)[0]
        rescaled_int_xi_r = InterpolatedUnivariateSpline(rescaled_r, 3*integral/rescaled_r**3, ext=3)

        # and rescale the normalization of the sigma_v(r) function
        sigma_v = alpha_par * sigma_v

        # now we calculate the model xi(s, mu) on a grid of s and mu values
        # here the argument s refers to the observed radial separation, and therefore differs from the "true" s by the
        # AP factors
        mu_grid = np.linspace(0, 1, 51)  # 51 is a compromise, should check this
        s_grid = s
        S, Mu = np.meshgrid(s_grid, mu_grid)
        true_sperp = S * np.sqrt(1 - Mu**2) * alpha_perp
        true_spar = S * Mu * alpha_par
        true_s = np.sqrt(true_spar**2 + true_sperp**2)
        true_mu = true_spar / true_s

        # a major difference between model 1 and model 2 is that model 2 neglects the shift in the coordinates themselves
        # so compare this next line to the equivalent in model 1 functions
        r_par = true_spar

        # now integrate over y at each (s, mu) combination
        xi_model_grid = np.zeros_like(S)
        for i in range(xi_model_grid.shape[0]):
            for j in range(xi_model_grid.shape[1]):
                # component of coherent outflow velocity along the line-of-sight
                vel_par = -beta * true_s[i, j] * rescaled_int_xi_r(true_s[i, j]) / (3 * self.iaH) * true_mu[i, j]

                # array of velocity values around this mean
                sv_central = sigma_v * rescaled_sv_norm_func(true_s[i, j])
                v = vel_par + np.linspace(-5 * sv_central, 5* sv_central, 100)

                # the value of r_par and r accounting for the velocity shift
                rpary = r_par[i, j] - v * self.iaH
                r = np.sqrt(true_sperp[i, j]**2 + rpary**2)
                # (note again that s and r are interchangeable in this model)

                # the value of sigma_v(r)
                sv = sigma_v * rescaled_sv_norm_func(r)

                # and the value of the coherent radial velocity component v(r) times mu
                vel_r_mu = -beta * r * rescaled_int_xi_r(r) / (3 * self.iaH) * true_mu[i, j]

                # now the integral
                integrand = (1 + rescaled_xi_r(r)) * np.exp(-0.5 * ((v - vel_r_mu) / sv)**2)
                integrand = integrand / (np.sqrt(2 * np.pi) * sv)

                xi_model_grid[i, j] = np.trapz(integrand, v) - 1

        # now build the true model by interpolating over this grid
        xi_model = interp2d(s_grid, mu_grid, xi_model_grid, kind='cubic')

        # and get the multipoles
        theory_multipoles = np.zeros(2 * len(s))
        theory_multipoles[:len(s)] = correlation_multipoles(xi_model, s, ell=0)
        theory_multipoles[len(s):] = correlation_multipoles(xi_model, s, ell=2)

        return theory_multipoles


    def model3_theory_quadrupole(self, beta, monopole, int_monopole):
        """
        Method to calculate the redshift-space quadrupole of the void-galaxy correlation function, using Model 3
        """

        quadrupole = 2 * beta * (monopole - int_monopole) / 3

        return quadrupole


    def lnlike(self, theta):
        """
        Log likelihood function

        :param theta:   array_like, dimensions (self.ndim)
                        vector position in parameter space
        :return: lnlkl: log likelihood value
        """

        if self.theory_model == 3:
            beta = theta
        else:
            if self.assume_lin_bias:
                beta, sigma_v, epsilon = theta
                apar = epsilon ** (-2./3)
                aperp = epsilon * apar
            else:
                fs8, bs8, sigma_v, epsilon = theta
                beta = fs8 / bs8
                apar = epsilon ** (-2./3)
                aperp = epsilon * apar
            if self.use_recon:
                s_multipoles = self.interpolated_data_vector(beta)
                cov = self.interpolated_covmat(beta)
                if np.all(s_multipoles == 0) or np.all(cov == 0):
                    # interpolation failed for some reason, return negative infinite log likelihood
                    return np.inf
                icov = np.linalg.inv(cov)
            else:
                s_multipoles = self.s_multipoles
                icov = self.icov

        # get the theory values
        if self.theory_model == 1:
            if self.assume_lin_bias:
                theory = self.model1_theory_with_lin_bias(beta, sigma_v, aperp, apar, self.data_rvals)
            else:
                theory = self.model1_theory_without_lin_bias(fs8, bs8, sigma_v, aperp, apar, self.data_rvals)
        elif self.theory_model == 2:
            if self.assume_lin_bias:
                theory = self.model2_theory_with_lin_bias(beta, sigma_v, aperp, apar, self.data_rvals)
            else:
                theory = self.model2_theory_without_lin_bias(fs8, bs8, sigma_v, aperp, apar, self.data_rvals)
        else:  # ie, model 3
            theory = self.model3_theory_quadrupole(beta, self.xi_s_mono(self.data_rvals), self.int_xi_s_mono(self.data_rvals))

        if np.all(theory == 0):
            # some error in the theory module, so return negative infinite log likelihood
            return -np.inf

        # get the chi-squared values
        if self.theory_model == 1 or self.theory_model == 2:
            chisq = np.dot(np.dot(theory - s_multipoles, icov), theory - s_multipoles)
            like_factor = 0
            if self.use_recon and not self.fixed_covmat: #  account for beta dependence of covariance in likelihood
                cov = np.linalg.inv(icov)
                det = np.linalg.slogdet(cov)
                if not det[0] == 1:
                    # something has gone dramatically wrong!
                    return -np.inf
                like_factor = -0.5 * det[1]
        elif self.theory_model == 3:
            # this one is a little different: we use the prescription in Cai et al 1603.05184
            G_factor = 2 * beta / (3 + beta)
            cov_tot = self.covmat[self.nrbins:, self.nrbins:] + G_factor * self.covmat[:self.nrbins, :self.nrbins] - \
                      2 * G_factor * self.covmat[:self.nrbins, self.nrbins:]
            icov = np.linalg.inv(cov_tot)
            chisq = np.dot(np.dot(theory - self.s_multipoles[self.nrbins:], icov), theory - self.s_multipoles[self.nrbins:])
            like_factor = 0
            det = np.linalg.slogdet(cov_tot)
            if not det[0] == 1:
                # something has gone dramatically wrong!
                return -np.inf
            like_factor = -0.5 * det[1]

        # and return the appropriate log likelihood
        if self.nmocks_covmat == 0:
            # treat the covariance matrix estimate as without error
            # obviously not strictly correct, but I haven't yet figured out what the correct thing to do would be
            lnlkl = -0.5 * chisq + like_factor
        else:
            # use the Sellentin & Heavens approach to propagate the uncertainty in the covariance estimation
            lnlkl = -self.nmocks_covmat * np.log(1 + chisq / (self.nmocks_covmat - 1)) / 2 + like_factor

        return lnlkl


    def lnprior(self, theta):
        """
        Log prior function, to ensure proper priors

        :param theta: array_like, dimensions (self.ndim)
                      vector position in parameter space
        :return: log prior value
        """

        if self.theory_model == 3:
            beta = theta
            if self.beta_prior_range[0] < beta < self.beta_prior_range[1]:
                return 0.
        else:
            if self.assume_lin_bias:
                beta, sigma_v, epsilon = theta
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] and \
                    250 < sigma_v < 500 and 0.8 < epsilon < 1.2:
                    return 0.
            else:
                fs8, bs8, sigma_v, epsilon = theta
                beta = fs8 / bs8
                apar = epsilon ** (-2./3)
                aperp = epsilon * apar
                if self.beta_prior_range[0] < beta < self.beta_prior_range[1] and \
                    0.05 < fs8 < 1.5 and 0.1 < bs8 < 2 and 250 < sigma_v < 500 and \
                    0.8 < epsilon < 1.2:
                    return 0.

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


    def minus_lnpost(self, theta):
        """
        Minus the log posterior, for maximum likelihood estimation via minimization

        :param theta: array_like, dimensions (self.ndim)
                      vector position in parameter space
        :return: minus the log posterior value
        """

        return -1 * self.lnpost(theta)

    def chisq(self, theta):
        """
        Return the chi-square for a given point in parameter space (useful because of additional factors in the log
        likelihood)

        :param theta:   array_like, dimensions (self.ndim)
                        vector position in parameter space
        :return: chisq
        """

        if self.theory_model == 3:
            beta = theta
        else:
            if self.assume_lin_bias:
                beta, sigma_v, epsilon = theta
                apar = epsilon ** (-2./3)
                aperp = epsilon * apar
            else:
                fs8, bs8, sigma_v, epsilon = theta
                beta = fs8 / bs8
                apar = epsilon ** (-2./3)
                aperp = epsilon * apar
            if self.use_recon:
                s_multipoles = self.interpolated_data_vector(beta)
                cov = self.interpolated_covmat(beta)
                if np.all(s_multipoles == 0) or np.all(cov == 0):
                    # interpolation failed for some reason, return negative infinite log likelihood
                    return np.inf
                icov = np.linalg.inv(cov)
            else:
                s_multipoles = self.s_multipoles
                icov = self.icov

        # get the theory values
        if self.theory_model == 1:
            if self.assume_lin_bias:
                theory = self.model1_theory_with_lin_bias(beta, sigma_v, aperp, apar, self.data_rvals)
            else:
                theory = self.model1_theory_without_lin_bias(fs8, bs8, sigma_v, aperp, apar, self.data_rvals)
        elif self.theory_model == 2:
            if self.assume_lin_bias:
                theory = self.model2_theory_with_lin_bias(beta, sigma_v, aperp, apar, self.data_rvals)
            else:
                theory = self.model2_theory_without_lin_bias(fs8, bs8, sigma_v, aperp, apar, self.data_rvals)
        else:  # ie, model 3
            theory = self.model3_theory_quadrupole(beta, self.xi_s_mono(self.data_rvals), self.int_xi_s_mono(self.data_rvals))

        if np.all(theory == 0):
            # some error in the theory module, so return negative infinite log likelihood
            return np.inf

        # get the chi-squared values
        if self.theory_model == 1 or self.theory_model == 2:
            chisq = np.dot(np.dot(theory - s_multipoles, icov), theory - s_multipoles)
        elif self.theory_model == 3:
            # this one is a little different: we use the prescription in Cai et al 1603.05184
            G_factor = 2 * beta / (3 + beta)
            cov_tot = self.covmat[self.nrbins:, self.nrbins:] + G_factor * self.covmat[:self.nrbins, :self.nrbins] - \
                      2 * G_factor * self.covmat[:self.nrbins, self.nrbins:]
            icov = np.linalg.inv(cov_tot)
            chisq = np.dot(np.dot(theory - self.s_multipoles[self.nrbins:], icov), theory - self.s_multipoles[self.nrbins:])

        return chisq

import numpy as np
from scipy.special import hyp2f1
from scipy.interpolate import InterpolatedUnivariateSpline
from .eisenstein_hu import EisensteinHu

class ExcursionSetProfile:
    """
    Class to calculate void matter density profiles (void-matter cross-correlation function) derived from an excursion
    set model approach. The model used is based on Massara & Sheth (arXiv:1811.03132) but also incorporates additional
    corrections not presented in that paper
    """

    def __init__(self, h, omega_m, omega_b, z=0, ns=0.965, omega_k=0, mnu=0.06,
                 npts=200, use_eisenstein_hu=False, camb_accuracy=1):
        """
        Initalize :class:`ExcursionSetProfile`.

        Parameters
        ----------
        h : float
            Hubble constant, as $h=H_0/100$ km/s/Mpc

        omega_m : float
            Total matter density parameter

        omega_b : float
            Baryon density parameter

        z : float, default=0
            Redshift

        ns : float, default=0.965
            Scale index of the primordial power spectrum.

        omega_k : float, default=0
            Curvature parameter.

        mnu : float, default=0.06
            Sum of the neutrino masses in eV.

        npts : int, default=200
            Number of points in $k$ at which to evaluate power spectrum $P(k)$
            Points are chosen on a logarithmic scale between k=1e-4 and k=2

        use_eisenstein_hu : boolean, default=False
            Whether to use the approximate Eisenstein-Hu fitting formula for the power spectrum
            If False the code will attempt to use ``camb`` instead (need to install separately)

        camb_accuracy : float, default=1
            Parameter determining numerical accuracy of power spectrum calculation by CAMB
            Used to set ``camb.model.AccuracyParams.AccuracyBoost``, ignored if not using CAMB
            Values <1 lower accuracy and speed up calculation, >1 increase accuracy and slow it down
        """

        omch2 = (omega_m - omega_b) * h**2
        ombh2 = omega_b * h**2
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.omega_l = 1 - omega_m - omega_k

        self.k = np.logspace(-4, np.log10(2), npts)

        if not use_eisenstein_hu:
            try:
                import camb
            except ImportError:
                print('========You need to install camb separately to use it!========')
                print('Proceeding instead with Eisenstein-Hu approximation to matter power spectrum')
                use_eisenstein_hu = True
        self.use_eisenstein_hu = use_eisenstein_hu

        if self.use_eisenstein_hu:
            # primordial amplitude As value not important as will be normalised later
            ehu = EisensteinHu(h, self.omega_m, self.omega_b, ns=ns, As=2e-9)
            # matter power at redshift 0
            pk_EH_0 = ehu.power_EH(self.k)
            # build the spline interpolator
            self.pk_EH_spline = InterpolatedUnivariateSpline(self.k, pk_EH_0)

            # get the sigma8 value for this power spectrum
            self.s80_fiducial = ehu.compute_sigma80()
            self.s8z_fiducial = self.s80_fiducial * self.growth_factor(z)
        else:
            pars = camb.CAMBparams()
            pars.set_accuracy(AccuracyBoost=camb_accuracy)

            #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
            pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0)
            # primordial amplitude As value not important as will be normalised later
            pars.Initpower.set_params(As=2e-9, ns=ns, r=0)
            if z > 0:
                pars.set_matter_power(redshifts=[z, 0.], kmax=2.0)
            else:
                pars.set_matter_power(redshifts=[0.], kmax=2.0)

            #Linear power spectrum
            pars.NonLinear = camb.model.NonLinear_none
            results = camb.get_results(pars)
            if z > 0:
                self.s8z_fiducial, self.s80_fiducial = results.get_sigma8()
            else:
                self.s80_fiducial = results.get_sigma8()
                self.s8z_fiducial = self.s80_fiducial
            self.pk = camb.get_matter_power_interpolator(pars, nonlinear=False)

    def growth_factor(self, z):
        """
        Linear growth factor D(z) at redshift z, normalised to unity at z=0

        Parameters
        ----------
        z : float
            Redshift
        """
        az = 1. / (1 + z)
        growth = az**2.5 * np.sqrt(self.omega_l + self.omega_m * az**-3) * \
                 hyp2f1(5/6, 3/2, 11/6, -(self.omega_l * az**3) / self.omega_m) / \
                 hyp2f1(5/6, 3/2, 11/6, -self.omega_l / self.omega_m)
        return growth

    def power(self, k, z):
        """
        Power spectrum P(k,z)

        Parameters
        ----------
        k : array
            k values, in units of h/Mpc

        z : float
            Redshift
        """

        if self.use_eisenstein_hu:
            D = self.growth_factor(z)
            return self.pk_EH_spline(k) * D**2
        else:
            return self.pk.P(z, k)

    def set_normalisation(self, sigma8, z=0):
        """
        Set the normalisation of the matter power spectrum amplitude using sigma8

        The power spectrum is calculated with a default amplitude of the primordial power, As. Typically we want to
        match to some (linear) sigma8 value instead so we set a constant normaliation factor here by which the power
        spectrum should be multipplied to achieve this

        Parameters
        ----------
        sigma8 : float
            The value of sigma8 to match to

        z : float, default=0
            The redshift at which to match to this amplitude
        """
        if z==0:
            self.normalisation = (sigma8 / self.s80_fiducial)**2
        else:
            self.normalisation = (sigma8 / self.s8z_fiducial)**2

    def _window_tophat(self, k, R):
        """
        Top hat window function in Fourier space
        """
        return 3.0 * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R)**3

    def _window(self, k, R, Rx):
        """
        Top hat window function with additional exponential cutoff
        """
        return self._window_tophat(k, R) * np.exp(-(k * R / Rx)**2 / 2)

    def _sj_pq(self, Rp, Rq, Rx, j=0):
        """
        Power spectrum variance cross term
        """
        kk, rp, rq = np.meshgrid(self.k, Rp, Rq)
        integrand = kk**(2 + 2 * j) * self.normalisation * self.power(kk, 0) * self._window(kk, rp, Rx) * \
                    self._window_tophat(kk, rq) / (2 * np.pi**2)
        return np.trapz(integrand, kk, axis=1)

    def _sj_pp(self, Rp, Rx, j=0, Rq=None):
        """
        Power spectrum variance
        """
        kk, rp, rq = np.meshgrid(self.k, Rp, Rq)
        integrand = kk**(2 + 2 * j) * self.normalisation * self.power(kk, 0) * self._window(kk, rp, Rx)**2 / (2 * np.pi**2)
        return np.trapz(integrand, kk, axis=1)

    def _sj_pp_ratio(self, Rp, Rx, Rq=None):
        """
        Ratio of _sj_pp(j=0) / _sj_pp(j=1) when both evaluated at the same Rp, Rx, Rq values
        This implementation is faster than using individual repeated calls to the method above
        """
        kk, rp, rq = np.meshgrid(self.k, Rp, Rq)
        _window = self._window_tophat(kk, rp) * np.exp(-(kk * rp / Rx)**2 / 2)
        integrand0 = kk**2 * self.normalisation * self.power(kk, 0) * _window**2 / (2 * np.pi**2)
        integrand1 = kk**2 * integrand0
        j_zero = np.trapz(integrand0, kk, axis=1)
        j_one = np.trapz(integrand1, kk, axis=1)

        return j_zero / j_one

    def _s0_derivative_term(self, Rp, Rq, Rx):
        """
        Derivative ds_0^pq / ds_0^pp appearing in EST model for Lagrangian density profile
        """
        step = 0.01 * Rp
        rp = Rp + np.array([-2, -1, 1, 2]) * step
        deriv_sjpq = (-self._sj_pq(rp[3], Rq, Rx, 0) + 8 * self._sj_pq(rp[2], Rq, Rx, 0) - 8 *
                      self._sj_pq(rp[1], Rq, Rx, 0) + self._sj_pq(rp[0], Rq, Rx, 0)) / (12 * step)
        deriv_sjpp = (-self._sj_pp(rp[3], Rx, 0) + 8 * self._sj_pp(rp[2], Rx, 0) - 8 * self._sj_pp(rp[1], Rx, 0) +
                      self._sj_pp(rp[0], Rx, 0)) / (12 * step)
        return deriv_sjpq / deriv_sjpp

    def _lagrangian_profile(self, Rq, b10, b01, Rp, Rx):
        """
        Excursion set model for the enclosed density profile around voids in Lagrangian space

        Parameters
        ----------
        Rq : array
            Distances from the void centre in Lagrangian space at which to evaluate the profile

        b10 : float
            bias parameter

        b01 : float
            bias parameter

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff
        """
        return b10 * self._sj_pq(Rp, Rq, Rx, 0) + b01 * 2 * self._sj_pp(Rp, Rx, 0) * self._s0_derivative_term(Rp, Rq, Rx)

    def _eulerian_1halo(self, r_lagrange, z, b10, b01, Rp, Rx, delta_c=1.686):
        """
        Contribution to the modelled enclosed density profile in Eulerian space arising from spherical evolution
        alone (referred to as the "1-halo" piece)

        Parameters
        ----------
        r_lagrange : array
            Distances from the void centre in Lagrangian space at which to evaluate the profile

        z : float
            Redshift of void

        b10 : float
            bias parameter

        b01 : float
            bias parameter

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff

        delta_c : float, default=1.686
            Critical density for collapse of an overdensity in spherical evolution model

        Returns
        -------
        r_euler : array of same shape as ``r_lagrange``
            Corresponding distances from void centre in Eulerian space

        one_halo : array of same shape as ``r_lagrange``
            1-halo contribution to enclosed density profile at Eulerian distance ``r_euler``
        """

        one_halo = (1 - self.growth_factor(z) * self._lagrangian_profile(r_lagrange, b10, b01, Rp, Rx) / delta_c)**(-delta_c) - 1
        r_euler = r_lagrange / (1 + one_halo)**(1/3)
        return r_euler, one_halo

    def _eulerian_2halo(self, r_euler, Rp, Rx):
        """
        Additional contribution to the modelled enclosed density profile in Eulerian space arising from void motion
        (referred to as the "2-halo" piece)

        Parameters
        ----------
        r_euler : array
            Distances from the void centre in Eulerian space at which to evaluate the profile

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff


        Returns
        -------
        2-halo term contribution at Eulerian distance ``r_euler``
        """
        # faster
        bv = 1 - self.k**2 * self._sj_pp_ratio(Rp, Rx)
        # equivalent but slower
        # bv = 1 - self.k**2 * self._sj_pp(Rp, Rx, 0) / self._sj_pp(Rp, Rx, 1)
        integrand = bv * self._window(self.k, Rp, Rx) * self._window_tophat(self.k, r_euler) * self.normalisation * \
                    self.power(self.k, 0) * self.k**2 / (2 * np.pi**2)
        return np.trapz(integrand, self.k)

    def model_enclosed_density_profile(self, r, z, b10, b01, Rp, Rx, delta_c=1.686):
        r"""
        Full model calculation of enclosed matter density profile around voids in Eulerian space

        Parameters
        ----------
        r : array
            Array of distances from the void centre. This sets the range of distances over which the interpolating
            function for the matter density profile is calculated (``r`` is intially used as the Lagrangian distance)

        z : float
            Redshift of void

        b10 : float
            bias parameter

        b01 : float
            bias parameter

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff

        delta_c : float, default=1.686
            Critical density for collapse of an overdensity in spherical evolution model

        Returns
        -------
        An instance of ``scipy.interpolate.InterpolatedUnivariateSpline`` representing the Eulerian enclosed matter
        density profile :math:`\Delta(r)`
        """

        # calculate Eulerian distances and 1-halo term
        r_euler, model_1halo = self._eulerian_1halo(r, z, b10, b01, Rp, Rx, delta_c)
        r_euler = r_euler[0]; model_1halo = model_1halo[0]

        # check for NaNs in RqE and remove if necessary
        valid = np.logical_not(np.isnan(r_euler))
        r_euler = r_euler[valid]
        model_1halo = model_1halo[valid]

        # check for shell-crossing and correct if necessary
        aux = np.where(r_euler[1:] - r_euler[:-1] < 0)[0]
        if aux.size != 0:
            choose_r = r_euler[aux[-1]+1]
            to_erase = np.where(r_euler > choose_r)[0]
            to_erase = to_erase[to_erase <= aux[-1]]
            r_euler = np.delete(r_euler, to_erase)
            model_1halo = np.delete(model_1halo, to_erase)
            print(f'Shell crossing occurred for b10={b10}, b01={b01}, Rp={Rp}, Rx={Rx}')
            print(f'Other parameters: Omega_m={self.omega_m}, Omega_L={self.omega_l}, Omega_b={self.omega_b}')
            print(f'\t s8={self.s80_fiducial *self.normalisation**0.5}')

        # calculate the 2-halo term
        model_2halo = np.zeros_like(r_euler)
        for i, rqe in enumerate(r_euler):
            model_2halo[i] = self._eulerian_2halo(rqe, Rp, Rx)

        # full model
        model_full = model_1halo + self.growth_factor(z)**2 * model_2halo
        return InterpolatedUnivariateSpline(r_euler, model_full)

    def model_density_profile(self, r, z, b10, b01, Rp, Rx, delta_c=1.686):
        r"""
        Full model calculation of the matter density profile around voids (equivalent to the void-matter
        cross-correlation) in Eulerian space

        Parameters
        ----------
        r : array
            Array of distances from the void centre. This sets the range of distances over which the interpolating
            function for the matter density profile is calculated (``r`` is intially used as the Lagrangian distance)

        z : float
            Redshift of void

        b10 : float
            bias parameter

        b01 : float
            bias parameter

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff

        delta_c : float, default=1.686
            Critical density for collapse of an overdensity in spherical evolution model

        Returns
        -------
        An instance of ``scipy.interpolate.InterpolatedUnivariateSpline`` representing the Eulerian matter density profile
        :math:`\delta(r)`
        """

        enclosed_density = self.eulerian_model_profiles(r, z, b10, b01, Rp, Rx, delta_c)
        derivative = np.gradient(enclosed_density(r), r)
        return InterpolatedUnivariateSpline(r, enclosed_density(r) + r * derivative / 3)

    def density_evolution(self, z, b10, b01, Rp, Rx, delta_c=1.686, r_max=120, pairwise=False):
        r"""
        Calculates the full non-linear time evolution of the enclosed density profile around voids in the excursion set model.
        In practice, it calculates $1/f$ times the derivative with respect to $\ln a$ of the enclosed density
        $\Delta(r)$, which can be related to the matter velocity profile around voids via the conservation equation
        \begin{equation}
        \frac{1}{f}\frac{\mathrm{d}\Delta(r)}{\mathrm{d}\ln a} = 3(1+\delta(r))\frac{v_r(r)}{faHr}
        \end{equation}
        where $f$ is the linear growth rate and $v_r(r)$ is the velocity profile

        Parameters
        ----------
        z : float
            Redshift of void

        b10 : float
            bias parameter

        b01 : float
            bias parameter

        Rp : float
            Smoothing scale at which the modelled void is an excursion set trough in Lagrangian density field

        Rx : float
            nuisance parameter setting the scale of the window function cutoff

        delta_c : float, default=1.686
            Critical density for collapse of an overdensity in spherical evolution model

        r_max : float, default=120
            Maximum distance from the void at which the profile evolution should be calculated

        pairwise : boolean, default=False
            If True, the evolution term calculated such that it is related via the conservation equation to the *pairwise*
            void-matter velocity profile, including a term intended to describe the void centre motion
            If False, the calculated evolution term is related to the matter velocities around a stationary point, i.e. any
            motion of the void centre is disregarded
            *For use in caclulating velocity profiles relevant to the void-galaxy cross-correlation almost certainly*
            ``pairwise=False`` *is required*

        Returns
        -------
        An instance of ``scipy.interpolate.InterpolatedUnivariateSpline``
        """

        x = np.linspace(0.1, r_max)
        r_euler, dSph = self._eulerian_1halo(x, z, b10, b01, Rp, Rx, delta_c)
        r_euler = r_euler[0]; model_1halo = model_1halo[0]

        # check for NaNs in RqE and remove if necessary
        valid = np.logical_not(np.isnan(r_euler))
        r_euler = r_euler[valid]
        dSph = dSph[valid]

        # build interpolating function and derivative of 1-halo term
        dSph = InterpolatedUnivariateSpline(r_euler, dSph)
        dSph_deriv = InterpolatedUnivariateSpline(r_euler, np.gradient(dSph(r_euler), r_euler))

        # calculate delta_2 term
        delta2 = np.zeros_like(r_euler)
        for i, re in enumerate(r_euler):
            delta2[i] = self.growth_factor(z) * self._eulerian_2halo(re, Rp, Rx)

        # Lagrangian profile
        DeltaL = self._lagrangian_profile(x, b10, b01, Rp, Rx)

        if pairwise:
            model = delta_c * (1 + dSph(r_euler) + r_euler * dSph_deriv(r_euler)/3) * \
                    ((1 + dSph(r_euler))**(1/delta_c) - 1) + 2 * self.growth_factor(z) * delta2
        else:
            model = delta_c * (1 + dSph(r_euler) + r_euler * dSph_deriv(r_euler)/3) * \
                    ((1 + dSph(r_euler))**(1/delta_c) - 1) + self.growth_factor(z) * delta2

        return InterpolatedUnivariateSpline(r_euler, model)

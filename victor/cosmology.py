import numpy as np
from astropy.cosmology import LambdaCDM
import scipy.constants as const
from scipy.special import hyp2f1

class BackgroundCosmology:
    """
    Class to calculate background cosmological quantities and some useful approximations

    Background quantities such as distances and Hubble rate will be accurately calculated even in non-flat
    models, although the FRW metric is always assumed. Approximations to the growth rate and similar quantities
    may only be good in flat models - note that this class does not compute the full transfer function correctly
    and only uses approximate fitting formulas!
    """

    def __init__(self, cosmology):
        """
        Initialise :class:`BackgroundCosmology`

        Parameters
        ----------
        cosmology : dict
            Dictionary of relevant cosmological parameters
        """

        self.c = const.c / 1000 # speed of light in km/s
        self.OmegaM = cosmology.get('Omega_m', 0.31)
        self.OmegaK = cosmology.get('Omega_K', 0)
        self.OmegaL = 1 - self.OmegaM - self.OmegaK
        self.H0 = cosmology.get('H0', 100*cosmology.get('h', 0.675))
        self.rd = cosmology.get('sound_horizon', 148.1)
        self.sigma8 = cosmology.get('sigma8', 0.81)
        self.cosmo = LambdaCDM(H0=self.H0, Om0=self.OmegaM, Ode0=self.OmegaL)

    def H(self, z):
        """
        Hubble parameter (in km/s/Mpc) at redshift z
        """
        return self.cosmo.H(z).value

    def Ez(self, z):
        """
        Normalised Hubble parameter H(z)/H0 at redshift z
        """
        return self.cosmo.H(z).value / self.H0

    def Om(self, z):
        """
        Matter density parameter at redshift z
        """
        return self.cosmo.Om(z)


    def comoving_distance(self, z, mpc_units=False):
        r"""
        Comoving distance :math:`D_C(z)` at redshift z

        Parameters
        ----------
        z : float or array
            Redshift

        mpc_units : bool, default=False
            Whether to return value in units of Mpc or Mpc/h (default)
        """
        if mpc_units:
            return self.cosmo.comoving_distance(z).value
        else:
            return self.cosmo.comoving_distance(z).value * self.H0 / 100

    def comoving_transverse_distance(self, z, mpc_units=False):
        r"""
        Comoving transverse distance :math:`D_M(z)` at redshift z

        Also known as proper motion distance or proper distance (see Hogg astro-ph/9905116)

        Parameters
        ----------
        z : float or array
            Redshift

        mpc_units : bool, default=False
            Whether to return value in units of Mpc or Mpc/h (default)
        """
        if mpc_units:
            return self.cosmo.comoving_transverse_distance(z).value
        else:
            return self.cosmo.comoving_transverse_distance(z).value * self.H0 / 100

    def hubble_distance(self, z, mpc_units=False):
        """
        Hubble distance :math:`D_H(z)` at redshift z

        Parameters
        ----------
        z : float or array
            Redshift

        mpc_units : bool, default=False
            Whether to return value in units of Mpc or Mpc/h (default)
        """
        if mpc_units:
            return self.c / self.H(z)
        else:
            return self.c / self.Ez(z)

    def angular_diameter_distance(self, z, mpc_units=False):
        """
        Angular diameter distance :math:`D_A(z)` at redshift z

        Parameters
        ----------
        z : float or array
            Redshift

        mpc_units : bool, default=False
            Whether to return value in units of Mpc or Mpc/h (default)
        """
        return self.comoving_transverse_distance(z, mpc_units) / (1 + z)

    def F_AP(self, z):
        r"""
        Alcock-Paczynski parameter :math:`F_{AP}(z)` at redshift z
        """
        return self.comoving_transverse_distance(z) / self.hubble_distance(z)

    def y(self, z):
        """
        A version of the Alcock-Paczynski parameter, divided by the redshift
        """
        return self.F_AP(z) / z

    def DH_over_rd(self, z, rd=None, mpc_units=False):
        """
        Hubble distance at redshift z in units of the sound horizon, useful for BAO

        Parameters
        ----------
        z : float or array
            Redshift

        rd : float, default=None
            Sound horizon in units of Mpc. If `None`, will use the value saved during initialization

        mpc_units : bool, default=False
            Whether to calculate Hubble distance in units of Mpc of Mpc/h (default)
        """
        if rd is None:
            rd = self.rd
        return self.hubble_distance(z, mpc_units) / rd

    def DM_over_rd(self, z, rd=None, mpc_units=False):
        """
        Transverse comoving distance at redshift z in units of the sound horizon, useful for BAO

        Parameters
        ----------
        z : float or array
            Redshift

        rd : float, default=None
            Sound horizon in units of Mpc. If `None`, will use the value saved during initialization

        mpc_units : bool, default=False
            Whether to calculate tranverse comoving distance in units of Mpc of Mpc/h (default)
        """
        if rd is None:
            rd = self.rd
        return self.comoving_transverse_distance(z, mpc_units) / rd

    def DV_over_rd(self, z, rd=None, mpc_units=False):
        r"""
        BAO angle-averaged distance :math:`D_V` at redshift z in units of the sound horizon

        Parameters
        ----------
        z : float or array
            Redshift

        rd : float, default=None
            Sound horizon in units of Mpc. If `None`, will use the value saved during initialization

        mpc_units : bool, default=False
            Whether to calculate D_V in units of Mpc of Mpc/h (default)
        """
        if rd is None:
            rd = self.rd
        return (z*self.comoving_transverse_distance(z, mpc_units)**2 * self.hubble_distance(z, mpc_units)**(1/3)) / rd

    def DA_over_rd(self, z, rd=None, mpc_units=False):
        """
        Angular diameter distance at redshift z in units of the sound horizon, useful for BAO

        Parameters
        ----------
        z : float or array
            Redshift

        rd : float, default=None
            Sound horizon in units of Mpc. If `None`, will use the value saved during initialization

        mpc_units : bool, default=False
            Whether to calculate angular diameter distance in units of Mpc of Mpc/h (default)
        """
        if rd is None:
            rd = self.rd
        return self.angular_diameter_distance(z, mpc_units) / rd

    def Hz_rd(self, z, rd=None, h_units=True, factor=1e3):
        """
        Hubble rate at redshift z times the sound horizon, useful for BAO

        This quantity can be returned in units of km/s or h km/s either of these two divided by a specified factor
        (to allow matching with some BAO papers which report this quantity in units of 1000 km/s)

        Parameters
        ----------
        z : float or array
            Redshift

        rd : float, default=None
            Sound horizon in units of Mpc. If `None`, will use the value saved during initialization

        h_units : bool, default=True
            Whether to divide by `h` (default) or not

        factor : float, default=1e3
            Additional convenience factor by which to divide the result
        """
        if rd is None:
            rd = self.rd
        return (self.c / self.hubble_distance(z, mpc_units=h_units)) * rd / factor

    def growth_factor(self, z):
        r"""
        Approximation to the linear growth factor :math:`D(z)` at redshift z
        """
        az = 1. / (1 + z)
        growth = az ** 2.5 * np.sqrt(self.OmegaL + self.OmegaM * az ** (-3.)) * \
                 hyp2f1(5. / 6, 3. / 2, 11. / 6, -(self.OmegaL * az ** 3.) / self.OmegaM) / \
                 hyp2f1(5. / 6, 3. / 2, 11. / 6, -self.OmegaL / self.OmegaM)
        return growth

    def growth_rate(self, z, gamma=0.545):
        r"""
        Approximation to the linear growth rate :math:`f(z)` at redshift z

        Calculated as Omega_m(z)**gamma for an input gamma factor

        Parameters
        ----------
        z : float
            Redshift

        gamma : float, default=0.545
            Value of the exponent to use in approximation (default is fit value for GR which is approximately 0.545)
        """
        return self.Om(z)**gamma

    def sigma8z(self, z, sigma80=None):
        """
        Approximation to linear theory amplitude of matter fluctuations on 8 Mpc/h scales, sigma8, at redshift z

        Calculated simply by scaling input value at redshift 0 using the linear growth factor!

        Parameters
        ----------
        z : float
            Redshift

        sigma80 : float, default=None
            The value of sigma8 at redshift 0. If not provided, defaults to the value set during initialization
        """
        if sigma80 is None:
            sigma80 = self.sigma8
        return sigma80 * self.growth_factor(z)

    def fsigma8(self, z, sigma80=None, gamma=0.545):
        """
        Approximation to RSD parameter :math:`f\sigma_8(z)` at redshift z

        Parameters
        ----------
        z : float
            Redshift

        sigma80 : float, default=None
            The value of sigma8 at redshift 0. If not provided, defaults to the value set during initialization

        gamma : float, default=0.545
            Value of the exponent to use in approximation (default is fit value for GR which is approximately 0.545)
        """
        return self.growth_rate(z, gamma) * self.sigma8z(z, sigma80)

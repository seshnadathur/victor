import numpy as np
from scipy import special
from scipy.integrate import quad

"""
This simple class computes the Eisenstein-Hu fitting formula for the matter power spectrum P(k,z=0)
Internal (private) methods of the class use units of 1/Mpc while the public methods assume k in h/Mpc
and give P(z=0,k) in units of (Mpc/h)^3
Some of this code is copied and modified from PyCosmo: https://cosmology.ethz.ch/research/software-lab/PyCosmo.html
"""

class EisensteinHu:

    def __init__(self, h, omega_m, omega_b, ns=0.965, As=2e-9, Tcmb=2.7255):
        """
        Take in cosmological parameters and compute all the parameters
        like sound horizon, equality scale etc needed by the fitting function
        """
        self.h = h
        self.Tcmb = Tcmb
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.omega_cdm = omega_m - omega_b
        self.ns = ns
        self.As = As
        omh2 = self.omega_m * self.h**2
        self.sigma_27 = self.Tcmb / 2.7
        self.z_equality = 2.5e4 * omh2 * self.sigma_27**(-4)
        b_1 = 0.313 * omh2 ** -0.419 * (1. + 0.607 * omh2 ** 0.674)
        b_2 = 0.238 * omh2 ** 0.223
        self.z_drag = (1291. * (omh2 ** 0.251) / (1. + 0.659 * omh2 ** 0.828) * (1. + b_1 *
                                                                                 (self.omega_b * self.h ** 2) ** b_2))
        self.k_eq = (7.46e-2 * omh2 * self.sigma_27 ** -2)
        self.k_silk = (1.6 * (self.omega_b * self.h ** 2) ** 0.52 * (omh2) ** 0.73 * (1. + (10.4 * omh2) ** -0.95))
        self.R_drag = (31.5 * self.omega_b * self.h**2 * self.sigma_27 ** (-4) * (self.z_drag / 10 ** 3) ** (-1.))
        self.R_eq = (31.5 * self.omega_b * self.h**2 * self.sigma_27 ** (-4) * (self.z_equality / 10 ** 3) ** (-1.))
        self.sound_horizon = (2. / (3. * self.k_eq) * np.sqrt(6 / self.R_eq) * np.log(( np.sqrt(1. + self.R_drag)
          + np.sqrt(self.R_drag + self.R_eq))/ (1. + np.sqrt(self.R_eq))))
        self.a1 = (46.9 * omh2) ** 0.670 * (1. + (32.1 * omh2) ** -0.532)
        self.a2 = (12. * omh2) ** 0.424 * (1. + (45. * omh2) ** -0.582)
        self.alpha_c = self.a1 ** ( -self.omega_b / self.omega_m ) * self.a2 ** (-(self.omega_b / self.omega_m) ** 3.)
        self.b1 = 0.944 * (1. + (458. * omh2) ** -0.708) ** -1
        self.b2 = (0.395 * omh2) ** -0.0266
        self.beta_c = (1. + self.b1 * ( (self.omega_cdm / self.omega_m) ** self.b2 - 1. )) ** -1.
        yy = (1. + self.z_equality) / (1. + self.z_drag)
        G = yy * (-6. * np.sqrt(1. + yy) + (2. + 3. * yy) * np.log((np.sqrt(1. + yy) + 1.) / (np.sqrt(1. + yy) - 1.)))
        self.alpha_b = (2.07 * self.k_eq * self.sound_horizon * (1. + self.R_drag)**(-.75) * G)
        self.beta_b = (0.5 + self.omega_b / self.omega_m + (3 - 2 * self.omega_b / self.omega_m) *
                       np.sqrt((17.2 * omh2) ** 2 + 1))
        self.beta_node = 8.41 * omh2**0.435

    def get_pofk_EH(self, k):
        """
        Input: k in h/Mpc
        Output: P(k,z=0) in (Mpc/h)^3
        """
        norm = 2.0 * np.pi ** 2 * self.As / self.h * 4.15e12
        pk = norm * (k*self.h/0.05)**self.ns * self.__transfer_EH(k*self.h)**2
        return pk

    def compute_sigma80(self):
        """
        Output: sigma8(z=0)
        """
        integrand = lambda x : 1.0/(2.0 * np.pi**2) * self.get_pofk_EH(x/8.0) * (x/8.0)**3 * \
                               (3.0/x**3 * (np.sin(x) - x * np.cos(x)))**2 / x
        sigma8_squared = quad(integrand, 1e-5, 20.0, full_output=1)[0]
        return sigma8_squared**0.5

    #------------ Internal methods -------------

    def __transfer_EH(self, k):
        """
        Internal function
        Input: k in 1/Mpc
        Output: T_matter(k)
        """
        T = (self.omega_b * self.__T_b(k) + self.omega_cdm * self.__T_cdm(k)) / self.omega_m
        return T

    def __T_b(self, k):
        """
        Internal function
        Input: k in 1/Mpc
        Output: T_baryon(k)
        """
        s_tilde = self.sound_horizon / (1. + (self.beta_node / (k * self.sound_horizon)) ** 3) ** (1. / 3.)
        T_b = (self.__T_0(k, 1., 1.) / (1. + (k * self.sound_horizon / 5.2) ** 2)
            + self.alpha_b / (1. + (self.beta_b / (k * self.sound_horizon)) ** 3.)
            * np.exp(-(k / self.k_silk) ** 1.4)) * np.sin(k * s_tilde) / (k * s_tilde)
        return T_b

    def __T_cdm(self, k):
        """
        Internal function
        Input: k in 1/Mpc
        Output: T_cdm(k)
        """
        f = 1. / (1. + (k * self.sound_horizon / 5.4) ** 4)
        T_c = f * self.__T_0(k, 1., self.beta_c) + (1. - f) * self.__T_0(k, self.alpha_c, self.beta_c)
        return T_c

    def __T_0(self, k, alpha_c, beta_c):
        """
        Internal function
        """
        q = k / (13.41 * self.k_eq)
        C = 14.2 / alpha_c + 386. / (1. + 69.9 * q ** 1.08)
        T_0 = np.log(np.e + 1.8 * beta_c * q) / (np.log(np.e + 1.8 * beta_c * q) + C * q ** 2)
        return T_0

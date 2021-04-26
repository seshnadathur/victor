import numpy as np
from scipy.integrate import quad
from scipy import constants
from scipy.special import hyp2f1

class Cosmology:
    """
    Calculate background cosmological quantities for 0<z<4
    """

    def __init__(self, omega_m=0.31, omega_l=0.69, h=0.676):
        c = constants.c / 1000
        omega_k = 1. - omega_m - omega_l
        ztab = np.linspace(0, 4, 1000)
        rtab = np.zeros_like(ztab)
        for i in range(len(ztab)):
            rtab[i] = quad(lambda x: 0.01 * c \
            / np.sqrt(omega_m * (1 + x)**3 + omega_k * (1 + x)**2 + omega_l), 0, ztab[i])[0]

        self.h = h
        self.omega_m = omega_m
        self.omegaL = omega_l
        self.omegaK = omega_k
        self.ztab = ztab
        self.rtab = rtab

    # comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return np.interp(z, self.ztab, self.rtab)

    def get_ez(self, z):
        return 100 * np.sqrt(self.omega_m * (1 + z)**3 +
                             self.omegaK * (1 + z)**2 + self.omegaL)

    def get_hubble(self, z):
        return self.h * self.get_ez(z)

    def get_redshift(self, r):
        return np.interp(r, self.rtab, self.ztab)

    def get_fz(self, z, gamma=0.55):
        return ((self.omega_m * (1 + z)**3.) / (self.omega_m * (1 + z)**3 + self.omegaL))**gamma

    def get_sigma8z(self, z, sigma_8_0):
        az = 1. / (1 + z)
        growth = az ** 2.5 * np.sqrt(self.omegaL + self.omega_m * az ** (-3.)) * \
                      hyp2f1(5. / 6, 3. / 2, 11. / 6, -(self.omegaL * az ** 3.) / self.omega_m) / \
                      hyp2f1(5. / 6, 3. / 2, 11. / 6, -self.omegaL / self.omega_m)
        return sigma_8_0 * growth

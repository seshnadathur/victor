import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, colorConverter
from scipy.ndimage import gaussian_filter
import os


class Cosmology:

    def __init__(self, omega_m=0.308, h=0.676):
        # print('Initializing cosmology with omega_m = %.3f' % omega_m)
        c = 299792.458
        omega_l = 1.-omega_m
        ztab = np.linspace(0, 4, 1000)
        rtab = np.zeros_like(ztab)
        for i in range(len(ztab)):
            rtab[i] = quad(lambda x: 0.01 * c / np.sqrt(omega_m * (1 + x) ** 3 + omega_l), 0, ztab[i])[0]

        self.h = h
        self.c = c
        self.omega_m = omega_m
        self.omegaL = omega_l
        self.ztab = ztab
        self.rtab = rtab

    # comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return np.interp(z, self.ztab, self.rtab)

    def get_redshift(self, r):
        return np.interp(r, self.rtab, self.ztab)


class UtilMethods:
    """
    Class providing access to various static methods
    """
    def __init__(self):
        """
        No need to do anything really
        """
        return

    @staticmethod
    def resize_vector(input_vector, k):
        """
        Method to resize composite multipole data vector in form (monopole, quadrupole) to keep only
        the first k indices in each multipole

        :param input_vector: numpy array, vector of multipoles to resize
        :param k: integer, largest index to keep
        :return: the truncated multipole vector
        """

        n_r = len(input_vector) / 2
        output_vector = np.zeros(2 * k)
        for k in range(n_r):
            if k < k:
                output_vector[k] = input_vector[k]
                output_vector[k + k] = input_vector[n_r + k]

        return output_vector

    @staticmethod
    def truncate_covmat(input_cm, k):
        """
        Method to truncate composite covariance matrix for multipole data vector (monopole and quadrupole), to keep
        only the first k indices in each

        :param input_cm: numpy array, input covariance matrix to resize
        :param k: integer, largest index to keep
        :return: the truncated covariance matrix
        """

        bin_range = input_cm.shape[0] / 2
        output_cm = np.zeros((2 * k, 2 * k))
        for i in range(input_cm.shape[0]):
            for j in range(input_cm.shape[0]):
                if i < k and j < k:
                    output_cm[i, j] = input_cm[i, j]
                elif bin_range <= i < bin_range + k and j < k:
                    output_cm[i + k - bin_range, j] = input_cm[i, j]
                elif i < k and bin_range <= j < bin_range + k:
                    output_cm[i, j + k - bin_range] = input_cm[i, j]
                elif bin_range <= i < bin_range + k and bin_range <= j < bin_range + k:
                    output_cm[i + k - bin_range, j + k - bin_range] = input_cm[i, j]

        return output_cm

    @staticmethod
    def truncate_grid_covmat(input_cm, k):
        """
        Method to truncate composite covariance matrix for multipole data vector (monopole and quadrupole), to keep
        only the first k indices in each, in the case that the covariance matrix is on a grid of beta values

        :param input_cm: numpy array, input covariance matrix to resize
        :param k: integer, largest index to keep
        :return: the truncated covariance matrix
        """

        bin_range = input_cm.shape[1] / 2
        output_cm = np.zeros((input_cm.shape[0], 2 * k, 2 * k))
        for k in range(input_cm.shape[0]):
            for i in range(input_cm.shape[1]):
                for j in range(input_cm.shape[1]):
                    if i < k and j < k:
                        output_cm[k, i, j] = input_cm[k, i, j]
                    elif bin_range <= i < bin_range + k and j < k:
                        output_cm[k, i + k - bin_range, j] = input_cm[k, i, j]
                    elif i < k and bin_range <= j < bin_range + k:
                        output_cm[k, i, j + k - bin_range] = input_cm[k, i, j]
                    elif bin_range <= i < bin_range + k and bin_range <= j < bin_range + k:
                        output_cm[k, i + k - bin_range, j + k - bin_range] = input_cm[k, i, j]

        return output_cm

    @staticmethod
    def quadrupole(xirmu, mu):
        """
        Method to return the quadrupole moment xi_2(r) of a given 2D correlation function xi(r, mu)

        :param xirmu: numpy array, dimensions (N_r x N_mu)
        :param mu: numpy array, dimensions (N_mu)
        :return: numpy array, dimensions (N_r), containing quadrupole moment
        """

        quadr = np.zeros(xirmu.shape[0])
        for j in range(xirmu.shape[0]):
            mufunc = InterpolatedUnivariateSpline(mu, xirmu[j, :], k=3)
            quadr[j] = quad(lambda x: mufunc(x) * 5 * (3. * x ** 2 - 1) / 2., 0, 1, full_output=1)[0]

        return quadr

    @staticmethod
    def next_pow_two(n):
        """
        Method to return the largest power of two smaller than a given positive integer

        :param n:  integer
        :return:   integer
        """
        i = 1
        while i < n:
            i = i << 1
        return i

    @staticmethod
    def autocorr_func_1d(x, norm=True):
        """
        Method to calculate the autocorrelation of a 1D array using FFT

        :param x:    numpy array
        :param norm: bool, optional
                     whether to normalize the autocorrelation (default =True)
        :return:     autocorrelation
        """
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = UtilMethods.next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2*n)
        acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
        acf /= 4*n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    @staticmethod
    def auto_window(taus, c):
        """
        Automated windowing procedure following Sokal (1989)

        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    @staticmethod
    def autocorrelation(y, c=5.0):
        """
        Method to estimate the autocorrelation time for an ensemble of chains

        """

        f = np.zeros(y.shape[1])
        for yy in y:
            f += UtilMethods.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = UtilMethods.auto_window(taus, c)
        return taus[window]


class FigureUtilities:

    def __init__(self):

        self.wad_colours = ['#2A4BD7', '#1D6914', '#814A19', '#8126C0',
                            '#9DAFFF', '#81C57A', '#E9DEBB', '#AD2323',
                            '#29D0D0', '#FFEE1F', '#FF9233', '#FFCDF3',
                            '#000000', '#575757', '#A0A0A0']

        self.kelly_colours = ['#FFB300',  # 0 Vivid Yellow
                              '#803E75',  # 1 Strong Purple
                              '#FF6800',  # 2 Vivid Orange
                              '#A6BDD7',  # 3 Very Light Blue
                              '#C10020',  # 4 Vivid Red
                              '#CEA262',  # 5 Grayish Yellow
                              '#817066',  # 6 Medium Gray
                              '#007D34',  # 7 Vivid Green
                              '#F6768E',  # 8 Strong Purplish Pink
                              '#00538A',  # 9 Strong Blue
                              '#FF7A5C',  # 10 Strong Yellowish Pink
                              '#53377A',  # 11 Strong Violet
                              '#FF8E00',  # 12 Vivid Orange Yellow
                              '#B32851',  # 13 Strong Purplish Red
                              '#F4C800',  # 14 Vivid Greenish Yellow
                              '#7F180D',  # 15 Strong Reddish Brown
                              '#93AA00',  # 16 Vivid Yellowish Green
                              '#593315',  # 17 Deep Yellowish Brown
                              '#F13A13',  # 18 Vivid Reddish Orange
                              '#232C16',  # 19 Dark Olive Green
                              ]

        self.RdYlGn = np.asarray(self.kelly_colours)[[7, 16, 14, 0, 12, 2, 18, 4, 15]]
        self.basic_RdYlBu = np.asarray(self.kelly_colours)[[11, 9, 3, 0, 12, 2, 18, 4, 15]]
        self.RdYlBu = np.array(['#3130ff', '#3366ff', self.wad_colours[4], self.kelly_colours[3],
                                self.kelly_colours[14], self.kelly_colours[0], self.kelly_colours[12],
                                self.kelly_colours[18], self.kelly_colours[4]])
        self.pointstyles = np.array(['o', '^', 'v', 's', 'D', '<', '>', 'p', 'd', '*', 'h', 'H', '8'])

        parchment_file = "/Users/seshadri/Workspace/colormaps/PlanckParchment.txt"
        if os.access(parchment_file, os.F_OK):
            self.parchment_cmap = ListedColormap(np.loadtxt(parchment_file)/255.)

    @staticmethod
    def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        """
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
                  Defaults to 0.0 (no lower ofset). Should be between
                  0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
                  0.5 (no shift). Should be between 0.0 and 1.0. In
                  general, this should be  1 - vmax/(vmax + abs(vmin))
                  For example if your data range from -15.0 to +5.0 and
                  you want the center of the colormap at 0.0, `midpoint`
                  should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
                  Defaults to 1.0 (no upper ofset). Should be between
                  `midpoint` and 1.0.
        """
        
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap

    @staticmethod
    def contour_2d(vectorx, vectory, weights=None, bins=20, levels=[0.683, 0.952]):

        H, X, Y = np.histogram2d(vectorx, vectory, bins=bins, weights=weights)
        H = gaussian_filter(H, 1)
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

        return X2, Y2, H2, V, H.max(), levels

    @staticmethod
    def plot_contour_2d(X2, Y2, H2, V, Hmax, levels, color1='cornflowerblue', color2='navy'):

        rgba_color = colorConverter.to_rgba(color1)
        contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
        for i, l in enumerate(levels):
            contour_cmap[i][-1] *= float(i) / (len(levels) + 1)
        contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        # contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
        plt.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [Hmax * (1 + 1e-4)]]), **contourf_kwargs)
        contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color2)
        plt.contour(X2, Y2, H2.T, V, **contour_kwargs)


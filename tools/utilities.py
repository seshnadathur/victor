import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, colorConverter
from scipy.ndimage import gaussian_filter
import os


def resize_datavector(input_vector, maxind, minind=0, multipoles=2):
    """
    Method to truncate a multipole data vector to keep only the entries between some minimum
    and maximum indices in each multipole (corresponding to some desired range of scales for a fit)

    :param input_vector: numpy array, vector of multipoles to resize
    :param maxind: integer, largest index to keep
    :param minind: integer, smallest index to keep (default 0)
    :param multipoles: integer, the number of multipoles in the data vector (default 2)
    :return: the truncated multipole vector
    """

    nbins_in = int(len(input_vector) / multipoles)
    nbins_out = int(maxind - minind)
    output_vector = np.zeros(int(multipoles * nbins_out))
    for i in range(nbins_in):
        if minind <= i < maxind:
            for j in range(multipoles):
                output_vector[j*nbins_out + i - minind] = input_vector[j*nbins_in + i]

    return output_vector


def truncate_covmat(input_cm, maxind, minind=0, multipoles=2):
    """
    Method to truncate the covariance matrix for a multipole data vector, to keep only the
    entries between some minimum and maximum indices in each multipole (corresponding to
    some desired range of scales for a fit)

    :param input_cm: numpy array, input covariance matrix to resize
    :param maxind: integer, largest index to keep in each multipole
    :param minind: integer, smallest index to keep in each multipole (default 0)
    :param multipoles: integer, the number of multipoles in the data vector (default 2)
    :return: the truncated covariance matrix
    """

    nbins_in = int(input_cm.shape[0] / multipoles)
    nbins_out = int(maxind - minind)
    output_cm = np.zeros((int(nbins_out * multipoles), int(nbins_out * multipoles)))
    for j in range(nbins_in):
        for k in range(nbins_in):
            if minind <= j < maxind and minind <= k < maxind:
                for i in range(multipoles):
                    # truncate the current diagonal block
                    output_cm[i*nbins_out + j - minind, i*nbins_out + k - minind] = input_cm[i*nbins_in + j, i*nbins_in + k]
                    # now do the off-diagonal blocks
                    for l in range(i, multipoles):
                        output_cm[l*nbins_out + j - minind, i*nbins_out + k - minind] = input_cm[l*nbins_in + j, i*nbins_in + k]
                        output_cm[i*nbins_out + j - minind, l*nbins_out + k - minind] = input_cm[i*nbins_in + j, l*nbins_in + k]

    return output_cm



def truncate_grid_covmat(input_cm, k):
    """
    Method to truncate a grid of covariance matrices for a multipole data vector, to keep only
    the entries between some minimum and maximum indices in each multipole (corresponding to
    some desired range of scales for a fit)

    :param input_cm: numpy array, input covariance matrices to resize - assume first index is grid index
    :param maxind: integer, largest index to keep in each multipole
    :param minind: integer, smallest index to keep in each multipole (default 0)
    :param multipoles: integer, the number of multipoles in the data vector (default 2)
    :return: the truncated covariance matrix
    """

    nbins_in = int(input_cm.shape[0] / multipoles)
    nbins_out = int(maxind - minind)
    output_cm = np.zeros((input_cm.shape[0], int(multipoles * nbins_out), int(multipoles * nbins_out)))
    for m in range(input_cm.shape[0]):
        for j in range(nbins_in):
            for k in range(nbins_in):
                if minind <= j < maxind and minind <= k < maxind:
                    for i in range(multipoles):
                        # truncate the current diagonal block
                        output_cm[m, i*nbins_out + j - minind, i*nbins_out + k - minind] = input_cm[m, i*nbins_in + j, i*nbins_in + k]
                        # now do the off-diagonal blocks
                        for l in range(i, multipoles):
                            output_cm[m, l*nbins_out + j - minind, i*nbins_out + k - minind] = input_cm[m, l*nbins_in + j, i*nbins_in + k]
                            output_cm[m, i*nbins_out + j - minind, l*nbins_out + k - minind] = input_cm[m, i*nbins_in + j, l*nbins_in + k]

    return output_cm

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

def auto_window(taus, c):
    """
    Automated windowing procedure following Sokal (1989)

    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

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

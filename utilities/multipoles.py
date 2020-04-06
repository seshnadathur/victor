import numpy as np
from scipy.integrate import quad
from scipy.special import legendre

def correlation_multipoles(xirmu, r, ell=0):
	"""
	Calculate the Legendre multipoles of a correlation function xi(r, mu)

	:param xirmu: an instance of scipy.interpolate.interp2d providing an interpolation for xi(r, mu)
	:param r: array of r values at which to calculate the multipoles
	:param ell: integer, multipole order

	Note that xi(r, mu) is assumed an even function of mu, so integral is evaluated over 0<mu<1 and then doubled

	Returns:
		array of values at the input r
	"""

	output = np.zeros(len(r))
	lmu = legendre(ell)
	for i in range(len(r)):
		output[i] = quad(lambda x: xirmu(r[i], x) * lmu(x) * (2 * ell + 1), 0, 1, full_output=1)[0]

	return output


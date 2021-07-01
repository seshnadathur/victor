import numpy as np
from scipy.special import legendre

def multipoles(xirmu, r, ell=0):
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
	# np.trapz with 200 pts gives factor ~4x speedup relative to quad, and differences of <0.1% in quadrupole
	npts = 200
	muarr = np.linspace(0.0, 1.0, npts)
	lmu = legendre(ell)(muarr)
	for i in range(len(r)):
		y = xirmu(r[i], muarr).T[0]
		output[i] = np.trapz(y * lmu * (2 * ell + 1), muarr)

	return output

def multipoles_singleshot(xirmu, r):
	"""
	Calculate the monopole, quadrupole and hexadecapole of a correlation function xi(r, mu)

	:param xirmu: an instance of scipy.interpolate.interp2d providing an interpolation for xi(r, mu)
	:param r: array of r values at which to calculate the multipoles

	Note that xi(r, mu) is assumed an even function of mu, so integral is evaluated over 0<mu<1 and then doubled

	Returns:
		3 arrays of multipoles evaluated at the input r
	"""

	output0 = np.zeros(len(r))
	output2 = np.zeros(len(r))
	output4 = np.zeros(len(r))

	# np.trapz with 200 pts gives factor ~4x speedup relative to quad, and differences of <0.1% in quadrupole
	npts = 200
	muarr = np.linspace(0.0, 1.0, npts)
	lmu = np.zeros((3, len(muarr)))
	for i, ell in enumerate([0, 2, 4]):
		lmu[i] = legendre(ell)(muarr)
	for i in range(len(r)):
		y = xirmu(r[i], muarr).T[0]
		output0[i] = np.trapz(y * lmu[0], muarr)     # monopole
		output2[i] = np.trapz(y * lmu[1] * 5, muarr) # quadrupole
		output4[i] = np.trapz(y * lmu[2] * 9, muarr) # hexadecapole

	return output0, output2, output4

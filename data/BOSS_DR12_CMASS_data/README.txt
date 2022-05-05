This folder contains 3 different file types: model files, data files and covariance matrix files. A full data analysis
requires one file of each type in the input. Model calculations only require a model file.

Model files:
------------

These files have the form <sample_identifier>_<type>_model.hdf5 where <sample_identifier> is
CMASS_zobovVoids_reconRs10_0.43z0.7_medianRvcut and indicates the galaxy sample (BOSS CMASS combined NGC+SGC), void
type, reconstruction information, redshift range and void sample cuts used. <type> distinguishes the two model
types:

* <type> = PatchyMean: indicates that the real-space CCF input is taken from a template, determined from the mean of
the measured real-space CCFs in the 1000 Patchy mock samples that are constructed to match the BOSS CMASS (NGC+SGC)
sample

* <type> = measured: indicates that the real-space CCF input is taken directly from the measurement on the BOSS CMASS
data, as for the data vector

Data files:
-----------

These files have the form <sample_identifier>_data.hdf5. All comparisons of model to data are made via compression to
Legendre multipoles, so these files contain data vectors in multipole space (here monopole and quadrupole only).

The presence of 'CMASS' at the start of the sample identifier indicates that the redshift-space CCF data vector is
that measured in the BOSS CMASS data sample; instead 'PatchyMean' indicates the mean of the redshift-space CCF data
vectors measured in the 1000 Patchy mocks.

Covariance matrix files:
------------------------

These files have names of the form *_covariance.hdf5. All covariance matrices are determined from the spread over the
1000 Patchy mocks. The following 'dictionary' explains the meaning of other terms appearing in the file name:

* 'CMASS': covariance matrix appropriate to a single realisation of the CMASS data (i.e. appropriate for use in fits
to the data sample)

* 'PatchyMean': covariance matrix appropriate to fits to the mean of 1000 Patchy mocks (estimated as 1/1000 times the
'CMASS' covariance)

* '_fixed': fixed covariance matrix assuming no dependence on reconstruction beta parameter (calculated at fixed
fiducial beta)

* '_variable': array of covariance matrices at different values of beta, provided in the file

* '_D': data-only covariance; i.e. determined from variation of redshift-space CCF data vectors over 1000 Patchy mocks.
This covariance is appropriate for use in cases where the input real-space CCF model has no correlation with the
redshift-space data vector, e.g. using PatchyMean model file and CMASS data file

* '_MD': covariance matrix accounting for correlation between real-space input and redshift-space data vector. This
covariance is appropriate to use when real-space and redshift-space quantities are generated from the same underlying
data so are correlated: e.g. when using PatchyMean model file and PatchyMean data file, or CMASS data file and measured
model file. There are two versions of this type of covariance matrix, labelled with 'isotropic' or 'anisotropic', which
indicates whether they are to be used when assuming the real-space CCF input is isotropic or using full anisotropic
information (using or not using anisotropic information naturally changes the correlation structure between the real-
space input and the multipoles of the redshift-space data vector).

Note that there is no fixed-beta version of the _MD covariances! The correlation between real-space input and the
redshift-space data vector strongly depends on beta (e.g. at beta=0 the two are identical!) so in this case it is not
appropriate to use the fixed-beta approximation.

Cosmology:
----------

All measurements on BOSS data and Patchy mocks where performed with fiducial cosmological model matching that of the
Big MultiDark simulation:
OmegaM = 0.307
OmegaL = 0.693
Omegab = 0.0482
sigma_8 = 0.8228
ns = 0.96
h = 0.6777

The matter ccf template was calibrated using the z=0.52 snapshot of Big MultiDark, which has implied sigma8 value of
sigma8(0.52) = 0.6228.

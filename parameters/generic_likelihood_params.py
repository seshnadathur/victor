# This file sets all the options for void-galaxy posterior sampling for the BigMD cubic simulation box

# three theory model options are provided:
# 1: the *correct* model with velocity dispersion (Nadathur & Percival 1712.04575)
# 2: the "quasi-linear" model of Cai et al (1603.05184) including velocity dispersion
# 3: the multipole ratio model (Cai et al 1603.05184, as used in Hamaus et al 1705.05328)
# NOTE: the required properties of the data vectors are different for different models!
# Model 3 requires fixed redshift-space information only, and cannot incorporate Alcock-Paczynski distortions (only RSD)
theory_model = 1

# if using theory models 1 or 2, specify reconstruction options
reconstructed_data = True
# True if real-space galaxy/void positions were obtained from reconstruction
# False for special case where they were known in simulation so no reconstruction was used
# if True, all data vectors will depend on beta=f/b so interpolation will be used
# NOTE: for theory model 3, it is assumed reconstruction was not performed and so always treated as False

# if reconstructed_data is True, provide filename for numpy pickle (.npy) file with the grid of beta values
beta_grid_file = ''

# file containing redshift-space multipole data vector - numpy pickle (.npy) format
s_space_multipole_file = ''
# this file should contain a dict, with items 'rvals' and 'multipoles'
# the 'rvals' item should be an array of length N containing the r values
# 1. if reconstructed_data is True, 'multipoles' should contain an array of size (M, 2*N) where M = no. of beta values
#    in the grid, and at each value of beta, the column contains the redshift-space monopole and quadrupole at each r
# 2. if reconstructed_data is False, 'multipoles' should contain a vector of length 2*N containing the redshift-space
#    monopole and quadrupole at each r

# file containing real-space multipole data vector - numpy pickle (.npy) format
r_space_multipole_file = ''
# same format at redshift-space version above
# NOTE: only required if using theory models 1 and 2!

# if reconstructed_data is True, is the covariance matrix fixed or also to be interpolated over the beta grid?
fixed_covmat = False
# NOTE: in principle if the data vectors vary with beta, the covariance matrix should too, but we allow this
# usage for testing

# filename for covariance matrix file (stored as numpy pickle, .npy)
covmat_file = ''
# if fixed_covmat is True, this should be an array of dimensions

# does the theory calculation use calibrated DM density profile delta(r), or determine it from the real-space
# monopole and assumption of linear galaxy bias within voids? Model 3 always assumes linear bias
assume_lin_bias = False

# if assume_lin_bias is False, provide filename for numpy pickle (.npy) file with calibrated delta information
# (this file should contain a dict, with item 'rvals' specifying r values and item 'delta' delta(r))
delta_file = ''

# central starting values for the MCMC over parameter space: should have the correct length for your chosen model!
start_values = [0.5, 1.2, 380, 1]
# rough estimate of posterior width in each parameter: this is used as the width of the Gaussian distribution from
# which the actual starting values of the chain are sampled
scales = [0.05, 0.05, 10, 0.03]
# stop condition for emcee chain: length of chain in units of autocorrelation length of most correlated parameter
stop_factor = 100

# specify sigma8 value at the redshift z_sim of the simulation used for calibration (this is only used for Models 1
# and 2, if assume_lin_bias is False)
sig8_norm = 0.628  # this is the value for the BigMD box at z_sim=0.52 used for BOSS calibration

# should we assume constant amplitude of the velocity dispersion as function of distance from void centre?
constant_dispersion = False
# NOTE: only relevant for models 1 and 2, model 3 does not include dispersion

# if constant_dispersion is False, provide filename for numpy pickle (.npy) file with velocity dispersion information
# (this file should contain a dict, with at least items 'rvals' and 'sigma_v_los' containing arrays of r, sigma_v(r))
dispersion_file = ''

# fiducial cosmology for converting velocity dispersion to distance scales
fiducial_omega_m = 0.31
fiducial_omega_l = 0.69

# effective redshift of the data
eff_z = 0.57

# number of mocks from which the covariance matrix was estimated (0 if covmat not estimated from mocks)
nmocks_covmat = 0
# this number will be used to evaluate the likelihood appropriately using Sellentin & Heavens (1511.05969) prescription

# output folder and root options
output_folder = ''
root = ''

# This file sets all the options for void-galaxy posterior sampling for the BOSS DR12 CMASS data, based on
# the combined measured void-galaxy cross-correlations from NGC and SGC data, using ZOBOV voids and related
# calibrated quantities (DM density profile and velocity dispersion profile) measured from mocks in the Big
# MultiDark N-body simulation

import os
data_dir = os.path.join(os.getcwd(), 'BOSS_DR12_CMASS_data/')

theory_model = 1

reconstructed_data = True
beta_grid_file = data_dir + 'beta_grid_for_recon.npy'
s_space_multipole_file = data_dir + 'cmass_combined_recon_zobov-Voids_xi_s_multipoles_R0.50.npy'
r_space_multipole_file = data_dir + 'patchy_combined_recon_zobov-Voids_xi_p_multipoles_R0.50.npy'
# NOTE: the real-space cross-corr. monopole is taken from the mean of the Patchy mocks rather than from the
# BOSS data for reasons explained in 1904.01030

fixed_covmat = False
covmat_file = data_dir + 'patchy_combined_recon_zobov-Voids_xi_s_multipoles_covmat_R0.50.npy'

assume_lin_bias = False
delta_file = data_dir + 'cmass_combined_recon_zobov-Voids_dm-profiles_R0.50.npy'
sig8_norm = 0.628  # this is the value for the BigMD box at z_sim=0.52 used for BOSS calibration
constant_dispersion = False
dispersion_file = data_dir + 'cmass_combined_recon_zobov-Voids_velocity-profiles_R0.50.npy'

fiducial_omega_m = 0.308
fiducial_omega_l = 0.692
eff_z = 0.57
nmocks_covmat = 1000

beta_prior_range = [0.16, 0.65]
# NOTE: this is indicative only; the beta grid provided interpolation is slightly narrower than this, and will
# determine the true prior range used

# output folder and root options
output_folder = os.path.join(os.getcwd(), 'chains/BOSS_DR12_CMASS/')
root = 'zobov_Model1'

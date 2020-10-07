# This file sets the options for void-galaxy posterior sampling for the
# cross-correlations measured in the eBOSS DR16 LRG+CMASS data, as performed in
# Nadathur et al (arXiv:2008.06060, MNRAS in press).
# Data files are provided based on the void-galaxy measurement from the combined
# NGC+SGC data, using void catalogues obtained from the Revolver code. Template
# functions (for the void-matter density profile and void velocity dispersion
# profile) were calibrated from mocks in the Big MultiDark N-body simulation.

import os
data_dir = os.path.join(os.getcwd(), 'eBOSS_DR16_LRG_data/')

mock_dir = '/Users/seshadri/sciama_data/eBOSS_CMASS/EZmocks/processed_output/'
data_dir = '/Users/seshadri/sciama_data/eBOSS_CMASS/dr16/processed_data/'

theory_model = 1

reconstructed_data = True
beta_grid_file = data_dir + 'beta_grid_for_recon.npy'
s_space_multipole_file = data_dir + 'dr16_xi_redshift_NGC+SGC_zobov-Voids_multipoles_R0.50.npy'
r_space_multipole_file = data_dir + 'ezmocks_xi_real_NGC+SGC_zobov-Voids_multipoles_R0.50.npy'

fixed_covmat = False
covmat_file = data_dir + 'covariance_xi_redshift_NGC+SGC_zobov-Voids_R0.50.npy'

assume_lin_bias = False
delta_file = data_dir + 'ebosscmass_zobov-voids_mean-dm-profile_R0.50.npy'
sig8_norm = 0.579  # value for the BigMD box at z_sim=0.71 used for eBOSS calibration
dispersion_file = data_dir + 'ebosscmass_zobov-voids_velocity-profile_R0.50.npy'

start_values = [0.47, 1.3, 350, 1]
scales = [0.001, 0.001, 1, 0.001]
stop_factor = 1000
max_steps = 2500

fiducial_omega_m = 0.31
fiducial_omega_l = 0.69
eff_z = 0.70
nmocks_covmat = 1000

beta_prior_range = [0.15, 0.57]
output_folder = data_dir + 'chains/'
root = 'dr16_ebosscmass_zobov-Voids_R0.50_Model1'

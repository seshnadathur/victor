import numpy as np
import yaml
from importlib.util import spec_from_file_location, module_from_spec
from cobaya.likelihood import Likelihood
from models import VoidGalaxyCCF, VoidGalaxyPosterior

class VoidGalaxyLikelihood(Likelihood):

    def initialize(self):
        """

        """
        if self.settings is None or self.paths is None:
            # these are then to be read from the separate spec_file provided
            try:
                with open(self.spec_file) as file:
                    info = yaml.full_load(file)
                self.settings = info['settings']
                self.paths = info['paths']
            except:
                raise KeyError('If settings and paths not provided in input yaml, spec_file must point to valid file with this info')

        # hacked wrapper to allow backwards compatibility with older code implementation
        if self.settings.get('use_old_code', False):
            self.use_old_code = True
            print(f'Use old code: {self.use_old_code}')
            # this step is just to get the module object in the format required by the old code
            # we then rewrite the attribute values by hand for back-compatibility
            spec = spec_from_file_location("name", 'parameter_files/boss_cmass_params.py')
            pars = module_from_spec(spec)
            spec.loader.exec_module(pars)
            pars.beta_grid_file = self.paths['multipole_beta_grid_file']
            pars.s_space_multipole_file = self.paths['redshiftspace_multipole_file']
            pars.r_space_multipole_file = self.paths['realspace_multipole_file']
            pars.fixed_covmat = self.settings.get('fixed_covmat', False)
            pars.covmat_file = self.paths['covariance_matrix_file']
            pars.assume_lin_bias = self.settings['delta_profile'] == 'use_linear_bias'
            pars.delta_file = self.paths.get('delta_template_file', '')
            pars.dispersion_file = self.paths.get('velocity_dispersion_template_file', None)
            if pars.dispersion_file is None:
                pars.constant_dispersion = True
            if self.settings['model'] == 'dispersion':
                pars.theory_model = 1
            elif self.settings['model'] == 'streaming':
                pars.theory_model = 2
                print('NOTE: Old code had a bug in the streaming model calculation.')
                print('Proceeding with buggy version for sake of comparison!')
            else:
                print('Old code can only calculate dispersion or streaming models. Setting to dispersion')
                pars.theory_model = 1
            pars.fiducial_omega_m = self.settings['fiducial_omega_m']
            pars.fiducial_omega_l = self.settings['fiducial_omega_l']
            pars.sig8_norm = self.settings['template_sigma8']
            pars.eff_z = self.settings['effective_redshift']
            pars.beta_prior_range = [0.01, 1.4]  # make it very broad for general applicability, as this will get overwritten anyway
            if not 'Sellentin' in self.settings['likelihood_type']:
                print('Sorry, the old code only allows use of the Sellentin & Heavens likelihood.')
                if 'Hartlap' in self.settings['likelihood_type']:
                    pars.nmocks_covmat = self.settings['likelihood_type']['Hartlap']['nmocks']
                    print('Proceeding anyway, but results may differ compared to Harltap correction')
                elif 'Gaussian' in self.settings['likelihood_type']:
                    pars.nmocks_covmat = 100000
                    print('Artificially setting num_mocks = 1e5 to approximately reproduce Gaussian likelihood')
            else:
                self.use_old_code = False
                pars.nmocks_covmat = self.settings['likelihood_type']['Sellentin']['nmocks']
            self.oldvgfitter = VoidGalaxyPosterior(pars)
        else:
            self.vgfitter = VoidGalaxyCCF(self.paths, self.settings)

    def logp(self, **params_values):
        """

        """
        if self.use_old_code:
            epsilon = params_values.get('aperp') / params_values.get('apar')
            if self.oldvgfitter.assume_lin_bias:
                theta = [params_values.get('beta'), params_values.get('sigma_v'), epsilon]
                return self.oldvgfitter.lnpost(theta)
            else:
                theta = [params_values.get('fsigma8'), params_values.get('bsigma8'), params_values.get('sigma_v'), epsilon]
                return self.oldvgfitter.lnpost(theta)
        else:
            return self.vgfitter.lnlike_multipoles(params_values, self.settings)

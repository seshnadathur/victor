import os
import yaml
from cobaya.likelihood import Likelihood
from victor import CCFFit

class CCFLikelihood(Likelihood):

    def initialize(self):
        """
        Initialise the likelihood
        """
        if self.model is None or self.data is None:
            print(f"'model' or 'data' blocks not provided, attempting to read them from config file {self.config_file}")
            # these are then to be read from the configuration file provided
            if os.path.isfile(self.config_file):
                with open(self.config_file) as file:
                    info = yaml.full_load(file)
                self.model = info['model']
                self.data = info['data']
            else:
                raise KeyError(f'config file {self.config_file} not found')

        self.ccf = CCFFit(self.model, self.data)

    def get_can_provide_params(self):
        return ['fsigma8']

    def calculate(self, state, want_derived=True, **params_values):
        """
        Calculate the likelihood and derived parameters
        """

        lnlike, chisq = self.ccf.log_likelihood(params_values)
        state['logp'] = lnlike
        state['derived'] = {'chi2_ccf_correct': chisq}
        if self.settings['delta_profile'] == 'use_excursion_model':
            # add fsigma8 as a derived parameter
            state['derived']['fsigma8'] = params_values['f'] * self.ccf.s8z

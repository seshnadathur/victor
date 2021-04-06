import numpy as np
import yaml
from cobaya.likelihood import Likelihood
from models import VoidGalaxyCCF

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

        self.vgfitter = VoidGalaxyCCF(self.paths, self.settings)

    def logp(self, **params_values):
        """

        """
        return self.vgfitter.lnlike_multipoles(params_values, self.settings)

import numpy as np
import yaml
from cobaya.likelihood import Likelihood
from models import VoidGalaxyCCF

class VoidGalaxyLikelihood(Likelihood):

    def initialize(self):
        """

        """
        with open(self.spec_file) as file:
            info = yaml.full_load(file)
        self.settings = info['settings']
        self.paths = info['paths']
        self.vgfitter = VoidGalaxyCCF(self.paths, self.settings)

    def logp(self, **params_values):
        """

        """
        return self.vgfitter.lnlike_multipoles(params_values, self.settings)

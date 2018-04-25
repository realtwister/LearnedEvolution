import numpy as np;

from .covariance_target import CovarianceTarget;

class ConstantCovariance(CovarianceTarget):
    def _calculate(self, population, evaluated_fitness):
        self._target = np.eye(population.shape[1])
        return self._target;

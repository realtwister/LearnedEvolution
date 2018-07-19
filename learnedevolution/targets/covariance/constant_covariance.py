import numpy as np;

from .covariance_target import CovarianceTarget;

class ConstantCovariance(CovarianceTarget):
    def __init__(self, variance=1):
        self._variance = variance;
    def _calculate(self, population, evaluated_fitness):
        self._target = self._variance*np.eye(population.shape[1])
        return self._target;

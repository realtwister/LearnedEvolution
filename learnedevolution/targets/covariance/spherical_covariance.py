import numpy as np;

from .covariance_target import CovarianceTarget;

class SphericalCovariance(CovarianceTarget):
    def __init__(self, alpha = 0.1):
        self.alpha = alpha

    def _reset(self, initial_mean, initial_covariance):
        self.variance = 1;
        self._mean = initial_mean;

    def _update_mean(self, mean):
        mean_diff = np.linalg.norm(self._mean-mean);
        self._mean = mean;
        self.variance = mean_diff*self.alpha;
        self.variance = np.clip(self.variance, 1e-30, 100);


    def _calculate(self, population, evaluated_fitness):
        self._target = self.variance*np.eye(population.shape[1]);

        return self._target;

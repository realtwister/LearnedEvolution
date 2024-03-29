import numpy as np;

from .covariance_target import CovarianceTarget;

class DiagonalCovariance(CovarianceTarget):
    def __init__(self, decay_rate=0.9, threshold = [1, 3]):
        self.gamma = decay_rate;
        self.threshold = threshold;

    def _reset(self, initial_mean, initial_covariance):
        self.variance = 1;

    def _update_mean(self, mean):
        mean_diff = np.linalg.norm(self._mean-mean);
        self._mean = mean;
        if True:
            new_variance = mean_diff
            log_variance = self.gamma*np.log(new_variance)+ (1-self.gamma)*np.log(self.variance)
            self.variance = np.exp(np.clip(log_variance, -7,3))
            return;


        if mean_diff < self.threshold[0]*np.sqrt(self.variance):
            self.variance *=self.gamma;
        elif mean_diff > self.threshold[1]*np.sqrt(self.variance):
            self.variance /= self.gamma;

        self.variance = np.clip(self.variance, 1e-8, 100)

    def _calculate(self, population, evaluated_fitness):
        self._target = self.variance*np.eye(population.shape[1]);

        return self._target;

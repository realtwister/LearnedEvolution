import numpy as np;

from .covariance_target import CovarianceTarget;

class AdaptiveCovariance(CovarianceTarget):
    def __init__(self, decay_rate= 1.0, percentile=.25):
        self.gamma = decay_rate;
        self.percentile = percentile;

    def _reset(self, initial_mean, initial_covariance):
        self._covariance = initial_covariance;
        self._mean = initial_mean;
        self._old_mean = initial_mean;

    def _update_mean(self, mean):
        self._old_mean = self._mean;
        self._mean = mean;
        return;
    def _calculate(self, population, F):
        new_covariance = 0;
        sel_idx = F.argsort()[-np.ceil(self.percentile*population.shape[0]).astype(int):][::-1]
        selection = population[sel_idx];
        for individual in selection:
            delta = individual-self._old_mean;
            new_covariance += np.outer(delta,delta)
        new_covariance /= (selection.shape[0]+1)
        new_covariance += np.eye(population.shape[1])*1e-25;
        self._covariance += self.gamma*(new_covariance-self._covariance);

        return self._covariance;

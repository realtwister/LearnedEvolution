import numpy as np;

from .covariance_target import CovarianceTarget;

class AdaptiveCovarianceEig(CovarianceTarget):
    def __init__(self, decay_rate= 0.5, percentile=.5, epsilon = 1e-27):
        self.gamma = decay_rate;
        self.percentile = percentile;
        self.epsilon = epsilon;

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
        new_covariance /= (selection.shape[0]-1)
        # We need to ensure the condition number is OK to avoid singular matrix.
        u,s,_ = np.linalg.svd(new_covariance);
        s_max  = np.max(s)
        s_max  = np.clip(s_max, self.epsilon, 1e3);
        s = np.clip(s, s_max/1e3, s_max);
        new_covariance = u*s@u.T

        self._old_cov = self._covariance;
        self._covariance += self.gamma*(new_covariance-self._covariance);

        return self._covariance;
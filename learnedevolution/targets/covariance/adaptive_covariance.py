import numpy as np;

from .covariance_target import CovarianceTarget;

class AdaptiveCovariance(CovarianceTarget):
    def __init__(self, decay_rate= 1.0, percentile=.50):
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
        biggest_delta = 0;
        for individual in selection:
            delta = individual-self._old_mean;
            if np.linalg.norm(delta)>biggest_delta:
                biggest_delta = np.linalg.norm(delta);
            new_covariance += np.outer(delta,delta)
        new_covariance /= (selection.shape[0]+1)
        new_covariance +=1e-40*np.eye(selection.shape[1])
        if np.linalg.norm(new_covariance)<1e-50:
            print(self._old_mean);
            print(self._covariance);
            print(set(list(tuple(l)for l in population)));
            print(biggest_delta);
            print(np.linalg.norm(new_covariance));
            exit();
        self._covariance += self.gamma*(new_covariance-self._covariance);

        return self._covariance;

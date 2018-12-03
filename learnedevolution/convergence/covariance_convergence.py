import numpy as np;
from .convergence import Convergence

class CovarianceConvergence(Convergence):
    def __init__(self,
        threshold = 1e-10,
        epsilon = 1e-30,
        max_iter = 200,
        reward_per_step =1,
        wait = 0):
        self.threshold = threshold;
        self.epsilon = epsilon;
        self.max_iter = max_iter;
        self.reward_per_step = reward_per_step;
        self._wait = wait;

    def reset(self, mean, covariance):

        self._wait -= 1;
        self.converged = False;
        self.iter = 0;

    def __call__(self, population):
        if not self.converged:
            self.converged = self.is_converged(population);
        return self.converged;

    def is_converged(self, population):
        self.iter +=1;
        if self.iter >= self.max_iter:
            return True;

        # Check if converging
        if self._wait<0:
            return np.max(population.svd[1])<self.threshold;
        return False;

    @property
    def reward_factor(self):
        if self.converged:
            return (self.max_iter-self.iter);
        return 0;

    @property
    def reward(self):
        return self.reward_factor*self.reward_per_step;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'threshold',
            'epsilon',
            'max_iter',
            'reward_per_step',
            'wait'
        )
        cls._config_defaults(
            threshold = 1e-10,
            epsilon = 1e-30,
            max_iter = 200,
            reward_per_step =1,
            wait = 0
        )

        return super()._get_kwargs(config, key = key);

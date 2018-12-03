import numpy as np;
from .convergence import Convergence

class TimeConvergence(Convergence):
    def __init__(self,
        max_iter = 100):
        self.max_iter = max_iter;
        self.epsilon = 1;

    def reset(self, mean, covariance):
        self.iter = 0;
        self.converged = False;

    def __call__(self, population):
        if not self.converged:
            self.converged = self.is_converged();
        return self.converged;

    def is_converged(self):
        self.iter +=1;
        if self.iter >= self.max_iter:
            return True;
        return False;

    @property
    def reward_factor(self):
        return 0;

    @property
    def reward(self):
        return 0;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'max_iter'
        )
        cls._config_defaults(
            max_iter = 200,
        )

        return super()._get_kwargs(config, key = key);

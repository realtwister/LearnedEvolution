import numpy as np;
from .convergence import Convergence

class PaperConvergence(Convergence):
    def __init__(self,
        max_iter = 100,
        threshold=1e-8):
        self.max_iter = max_iter;
        self.threshold = threshold

    def reset(self, mean, covariance):
        self.iter = 0;
        self.converged = False;
        self.hit = False;
        self.cooldown = 10

    def __call__(self, population):
        if not self.converged:
            if not self.hit and np.abs(np.mean(population.fitness)) < self.threshold:
                self.hit = True;
            self.converged = self.is_converged();
        return self.converged;

    def is_converged(self):
        self.iter +=1;
        if self.cooldown <= 0:
            return True
        if self.hit:
            self.cooldown -= 1
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
            'max_iter',
            'threshold'
        )
        cls._config_defaults(
            max_iter = 1000,
            threshold = 1e-8
        )

        return super()._get_kwargs(config, key = key);

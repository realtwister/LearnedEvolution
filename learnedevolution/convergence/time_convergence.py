import numpy as np;

class TimeConvergence(object):
    def __init__(self,
        max_iter = 100):
        self.max_iter = max_iter;
        self.epsilon = 1;

    def reset(self, mean, covariance):
        self.iter = 0;
        self.converged = False;

    def __call__(self, fitness, mean, covariance):
        if not self.converged:
            self.converged = self.is_converged(fitness, mean, covariance);
        return self.converged;

    def is_converged(self, fitness, mean, covariance):
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

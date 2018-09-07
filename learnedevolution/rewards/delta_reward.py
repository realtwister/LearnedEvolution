import numpy as np;

from .reward import Reward;


class DeltaReward(Reward):
    def _reset(self):
        self.prev_mean_fitness = 0;

    def __call__(self,population,  fitness):
        mean_fitness = np.mean(fitness);
        reward = mean_fitness - self.prev_mean_fitness;
        self.prev_mean_fitness = mean_fitness;
        return reward;

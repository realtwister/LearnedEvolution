import numpy as np;

from .reward import Reward;

class DifferentialReward(Reward):
    def _reset(self):
        self._step = 0;
        self._fitness_diff = 0;
        self._mean_fitness = 0;
        self._prev_fitness_diff = 1;
        self._prev_mean_fitness = 0;

    def __call__(self,population,  fitness):
        self._prev_mean_fitness = self._mean_fitness;
        self._mean_fitness = np.mean(fitness);
        self._fitness_diff = self._prev_mean_fitness - self._mean_fitness;

        reward = (self._fitness_diff / max(np.abs(self._prev_fitness_diff),1e-6)) if self._step > 1 else 0;
        reward = np.clip(reward, -5,5);
        self._prev_fitness_diff = self._fitness_diff;
        self._reward = reward;
        self._step += 1;
        return reward;

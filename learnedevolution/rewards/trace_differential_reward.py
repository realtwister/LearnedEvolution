import numpy as np;

from .reward import Reward;

class TraceDifferentialReward(Reward):
    def __init__(self, gamma=0.5):
        self._gamma = gamma;

    def _reset(self):
        self._step = 0;
        self._fitness_diff = 0;
        self._mean_fitness = 0;
        self._trace = float('NaN');
        self._prev_mean_fitness = 0;

    def __call__(self,population,  fitness):
        self._prev_mean_fitness = self._mean_fitness;
        self._mean_fitness = np.max(fitness);
        self._fitness_diff = self._mean_fitness-self._prev_mean_fitness;

        reward = (self._trace/self._fitness_diff ) if self._step > 1 else 0;
        reward = np.clip(reward, -5,5);
        if self._step == 1:
            self._trace = np.abs(self._fitness_diff);
        self._trace += self._gamma*(np.abs(self._fitness_diff)-self._trace);
        self._reward = reward;
        self._step += 1;
        return reward;

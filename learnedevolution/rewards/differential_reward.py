import numpy as np;

from .reward import Reward;

class DifferentialReward(Reward):
    def __init__(self,
        epsilon):
        self.epsilon = epsilon

    def _reset(self):
        self._step = -1;
        self._fitness_diff = 0;
        self._mean_fitness = 0;
        self._prev_fitness_diff = 1;
        self._prev_mean_fitness = 0;
        self._max = -float('Inf');

    def __call__(self, population,  fitness):
        self._step += 1;
        self._max = max(np.max(fitness), self._max);
        self._prev_mean_fitness = self._mean_fitness;
        self._mean_fitness = np.mean(fitness);
        factor = max(self._max - self._prev_mean_fitness, self.epsilon)
        reward = (self._mean_fitness-self._prev_mean_fitness)/factor if self._step > 1 else 0;
        return reward;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'epsilon'
        )
        cls._config_defaults(
        )
        return super()._get_kwargs(config, key = key);

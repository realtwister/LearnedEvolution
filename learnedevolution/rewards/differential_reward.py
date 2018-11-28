import numpy as np;

from .reward import Reward;

class DifferentialReward(Reward):
    def __init__(self, epsilon = 1e-20):
        self.epsilon = epsilon;
    def _reset(self):
        self._step = -1;
        self._fitness_diff = 0;
        self._mean_fitness = 0;
        self._prev_fitness_diff = 1;
        self._prev_mean_fitness = 0;
        self._max=-float('Inf');

    def __call__(self,population,  fitness):
        self._step += 1;
        if np.max(fitness)> self._max:
            self._max= np.max(fitness);
        self._prev_mean_fitness = self._mean_fitness;
        self._mean_fitness = np.mean(fitness);
        factor =max(self._max-self._mean_fitness, self.epsilon)
        reward = (self._mean_fitness-self._prev_mean_fitness)/factor if self._step > 1 else 0;
        return reward;
        self._fitness_diff = self._mean_fitness-self._prev_mean_fitness;

        reward = (max(np.abs(self._prev_fitness_diff),1e-6)/self._fitness_diff ) if self._step > 1 else 0;
        reward = np.clip(reward, -5,5);
        self._prev_fitness_diff = self._fitness_diff;
        self._reward = reward;
        return reward^3+reward;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'epsilon'
        )
        cls._config_defaults(
            epsilon = 1e-20
        )
        return super()._get_kwargs(config, key = key);

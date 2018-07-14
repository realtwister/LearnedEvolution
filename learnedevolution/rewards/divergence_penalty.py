import numpy as np;

from .reward import Reward;

class DivergencePenalty(Reward):
    def _reset(self):
        self._minimum = None;
        self._initial = None;

    def _calculate(self, population, fitness):
        current_minimum = np.min(fitness);
        if self._minimum is None:
            self._minimum = np.min(fitness);
            self._max = np.max(fitness);
            return 0;
        if current_minimum < self._minimum:
            self._minimum = current_minimum;

        current_mean = np.mean(fitness);
        res = np.tanh(self._minimum - current_mean + 0.5*(self._max-self._minimum));
        return res;

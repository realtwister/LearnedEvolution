import numpy as np;
from collections import deque;

from .reward import Reward;

class LaggingDifferentialReward(Reward):
    def __init__(self, lag=10):
        self._q = deque(maxlen = lag);
    def _reset(self):
        self._q.clear();
        self._max = -float('Inf');
        self._max_0 = None;
        self.step = 0;

    def __call__(self,population,  fitness):
        reward = 0;

        return -np.log(-np.mean(fitness));

        cur_max = np.max(fitness);
        self._max = max(self._max, cur_max);
        if self._max_0 is None:
            self._max_0 = cur_max;
        else:
            if len(self._q) == self._q.maxlen:
                calc_max = self._q.pop();
                if self._max - calc_max<1e-10:
                    reward=10;
                else:
                    reward = -np.log(self._max-calc_max)/(np.linalg.norm(np.std(population, axis=0)));
            self._q.appendleft(np.mean(fitness));
        self.step +=1;
        return reward;

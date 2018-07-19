import numpy as np;
from collections import deque;

from .reward import Reward;

class DecayingMinimum:
    def __init__(self, gamma=0.99):
        self.gamma = gamma;
    def reset(self):
        self.factor = 1;
        self.minimum = float('Inf');
        self.maximum = -float('Inf');

    def __call__(self, fitness):
        res = self.maximum+(self.minimum-self.maximum)*self.factor;
        self.minimum = min(self.minimum, fitness);
        self.maximum = max(self.maximum, fitness);
        self.factor *= self.gamma;
        return res;

class WindowMinimum:
    def __init__(self, window_size = 10, selection_ratio = 0.3):
        self.window = deque([],window_size);
        self.ratio = selection_ratio;

    def reset(self):
        self.window.clear();

    def __call__(self, fitness):
        self.window.append(fitness);
        return np.mean(sorted(self.window)[:np.ceil(len(self.window)*self.ratio).astype(int)]);

class InitialMinimum:
    def reset(self):
        self.minimum = None;

    def __call__(self, fitness):
        if self.minimum is None:
            self.minimum = fitness;
        return self.minimum;

class LaggingMaximum:
    def __init__(self, lag=1):
        self.queue = deque([],lag);
    def reset(self):
        self.queue.clear();
        self.maximum = -float('Inf');

    def __call__(self, fitness):
        if self.queue.maxlen == len(self.queue):
            #self.maximum =self.queue.popleft();
            self.maximum = max(self.maximum, self.queue.popleft());

        self.queue.append(fitness);

        return self.maximum;







class NormalizedFitnessReward(Reward):
    def __init__(self, minima, maxima):
        self.minima = minima;
        self.maxima = maxima;
    def _reset(self):
        for minimum in self.minima:
            minimum.reset();
        for maximum in self.maxima:
            maximum.reset();
        self._minimum = float('NaN');
        self._maximum = float('NaN');

    def _calculate(self, population, fitness):

        current_min = np.min(fitness);
        current_max = np.max(fitness);
        current_median = np.median(fitness);

        norm_minimum = -float('Inf');
        norm_maximum = -float('Inf');
        for minimum in self.minima:
            cur_min = minimum(current_median);
            if cur_min < float('Inf'):
                norm_minimum = max(norm_minimum,cur_min);

        for maximum in self.maxima:
            norm_maximum = max(norm_maximum, maximum(current_max));

        self._maximum = norm_maximum;
        self._minimum = norm_minimum;
        if norm_minimum<norm_maximum:
            #TODO: proper normalisation between -1 and 1
            return (1+np.tanh((current_median-norm_minimum)/(norm_maximum-norm_minimum)-1))/2;
        return 0;

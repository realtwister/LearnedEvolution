import numpy as np;
from collections import deque;

from .reward import Reward;
from ..utils.parse_config import ParseConfig,config_factory

def minima_classes():
    return dict(
        DecayingMinimum = DecayingMinimum,
        WindowMinimum = WindowMinimum,
        InitialMinimum = InitialMinimum
    )

    return res

def maxima_classes():
    return dict(
        DelayedMaximum = DelayedMaximum
    )


class DecayingMinimum(ParseConfig):
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

    @classmethod
    def _get_kwargs(cls, config, key= ""):
        cls._config_required(
            'gamma'
        )
        cls._config_defaults(
            gamma = 0.99
        )
        return super()._get_kwargs(config, key)

class WindowMinimum(ParseConfig):
    def __init__(self, window_size = 10, selection_ratio = 0.3):
        self.window = deque([],window_size);
        self.ratio = selection_ratio;

    def reset(self):
        self.window.clear();

    def __call__(self, fitness):
        self.window.append(fitness);
        return np.mean(sorted(self.window)[:np.ceil(len(self.window)*self.ratio).astype(int)]);

    @classmethod
    def _get_kwargs(cls, config, key= ""):
        cls._config_required(
            'window_size',
            'selection_ratio'
        )
        cls._config_defaults(
            window_size = 10,
            selection_ratio = 0.3
        )
        return super()._get_kwargs(config, key)

class InitialMinimum(ParseConfig):
    def reset(self):
        self.minimum = None;

    def __call__(self, fitness):
        if self.minimum is None:
            self.minimum = fitness;
        return self.minimum;

class DelayedMaximum(ParseConfig):
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
    @classmethod
    def _get_kwargs(cls, config, key= ""):
        cls._config_required(
            'lag'
        )
        cls._config_defaults(
            lag = 1
        )
        return super()._get_kwargs(config, key)

class NormalizedFitnessReward(Reward):
    def __init__(self, minima, maxima):
        self.minima = minima;
        self.maxima = maxima;
        print("Normalized initialized")
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
        current_mean = np.mean(fitness);

        norm_minimum = -float('Inf');
        norm_maximum = -float('Inf');
        for minimum in self.minima:
            cur_min = minimum(current_min);
            if cur_min < float('Inf'):
                norm_minimum = max(norm_minimum,cur_min);

        for maximum in self.maxima:
            norm_maximum = max(norm_maximum, maximum(current_max));

        self._maximum = norm_maximum;
        self._minimum = norm_minimum;
        if norm_minimum<norm_maximum:
            #TODO: proper normalisation between -1 and 1
            reward = (current_mean - norm_minimum)/(norm_maximum-norm_minimum);
            #return reward;
            return np.clip(10*reward**3+reward,-100,100)/100;
        return 0;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'minima',
            'maxima'
        )
        cls._config_defaults(
            minima = [
                dict(type = "InitialMinimum"),
                dict(type = "WindowMinimum"),
                dict(type= "DecayingMinimum")
            ],
            maxima = [
                dict(type = "DelayedMaximum")
            ]
        )

        kwargs = super()._get_kwargs(config, key)
        minima = []
        maxima = []
        for i, minimum in enumerate(kwargs['minima']):
            minima.append(config_factory(minima_classes, config, key+".minima."+str(i)))

        for i, maximum in enumerate(kwargs['maxima']):
            maxima.append(config_factory(maxima_classes, config, key+".maxima."+str(i)))

        kwargs['minima'] = minima
        kwargs['maxima'] = maxima

        return kwargs

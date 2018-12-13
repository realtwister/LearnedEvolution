import numpy as np
from .mean_target import MeanTarget

class CMAESMean(MeanTarget):
    def __init__(self,
        weights,
        c_m):
        self.c = c_m
        self.w = weights

    def _calculate(self, population):
        if callable(self.w):
            w = [self.w(i, len(population)) for i in range(len(population))]
        else:
            w = self.w
        assert len(w) >= len(population)
        w = np.array([max(0,w) for weight in w])
        xw = np.sum(w*population.raw_population[np.argsort(population.fitness)])


        return population.mean + self.c * population.morph(xw)


    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'weights',
            'c_m'
        )
        cls._config_defaults(
            weights = lambda i,l: (np.log((l+1)/2)-np.log(i+1))/np.sum([max(0,np.log((l+1)/2)-np.log(i)) for i in range(1,l+1)]),
            c_m = 1.
        )

        return super()._get_kwargs(config, key = key);

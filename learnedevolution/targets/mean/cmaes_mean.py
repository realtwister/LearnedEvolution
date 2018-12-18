import numpy as np
from .mean_target import MeanTarget

class CMAESMean(MeanTarget):
    def __init__(self,
        population_size,
        weights,
        c_m):
        self.c = c_m
        self.w = weights

        if callable(self.w):
            w = [self.w(i, population_size) for i in range(population_size)]
        else:
            w = list(self.w)
        assert len(w) >= population_size
        self.w = np.array(w)
        print(self.w)
    def _calculate(self, population):
        B, *_ =  population.svd
        sorted_idx = np.argsort(population.fitness)[::-1]
        w = self.w
        xw = np.sum(w[w>0,None]*population.raw_population[sorted_idx][w>0],axis=0)
        return population.mean + self.c * population.morph(xw)


    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'population_size',
            'weights',
            'c_m'
        )
        cls._config_defaults(
            weights = lambda i,l: (np.log((l+1)/2)-np.log(i+1))/np.sum([max(0,np.log((l+1)/2)-np.log(i)) for i in range(1,l+1)]),
            c_m = 1.
        )

        return super()._get_kwargs(config, key = key);

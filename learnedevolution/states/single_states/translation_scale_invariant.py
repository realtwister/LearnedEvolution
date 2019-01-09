import numpy as np
from .single_state import SingleState

class TranslationScaleInvariant(SingleState):
    def __init__(self,
        population_size,
        dimension):
        self.population_size = population_size
        self.dimension = dimension


    def _calculate(self, population, reference):
        if population is None:
            return np.zeros(shape=[self.population_size*(1+self.dimension)])
        m = reference.mean
        _,s,_ = reference.svd
        l = np.sqrt(np.max(s))

        P = (population.population-m)/l

        F_m = np.mean(reference.fitness)
        F_s = np.std(reference.fitness)
        if F_s < 1e-13:
            F_s = 1e-13
        F = (population.fitness-F_m)/F_s

        state = np.append(P, F[:, None], axis=1);
        state = state[population.fitness.argsort()];

        return state.flatten()

    def invert(self, x, reference):
        m = reference.mean
        _, s, _ = reference.svd
        l = np.sqrt(np.max(s))

        return x*l+m

    @property
    def state_space(self):
        return dict(
            type = 'float',
            shape = [(self.dimension+1)*self.population_size]
        )

    @property
    def action_space(self):
        return dict(
            type = 'float',
            shape = [self.dimension]
        )

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'population_size',
            'dimension'
        )

        cls._config_defaults(
            population_size = 100,
            dimension = 2
        )
        return super()._get_kwargs(config, key)

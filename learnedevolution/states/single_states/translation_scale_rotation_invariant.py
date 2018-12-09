import numpy as np
from .single_state import SingleState

class TranslationScaleRotationInvariant(SingleState):
    def __init__(self,
        population_size,
        dimension):
        self.population_size = population_size
        self.dimension = dimension


    def _calculate(self, population, reference):
        if population is None:
            return np.zeros(shape=[self.population_size*(1+self.dimension)])
        if reference == population:
            P = population.raw_population
        else:
            P = reference.invert(population)

        F_m = np.mean(reference.fitness)
        F_s = np.std(reference.fitness)
        F = (population.fitness-F_m)/F_s

        state = np.append(P, F[:, None], axis=1);
        state = state[population.fitness.argsort()];
        return state.flatten()

    def invert(self, x, reference):
        dx = reference.morph(x)
        n = np.linalg.norm(dx);
        if n > 1e2:
            dx *= 1e2/n
        if np.any(np.isnan(dx)):
            print(dx, n);
            raise Exception("dx is nan");
        return dx+reference.mean

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

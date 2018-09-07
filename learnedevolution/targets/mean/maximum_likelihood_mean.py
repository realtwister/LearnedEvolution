import numpy as np;

from .mean_target import MeanTarget;

class MaximumLikelihoodMean(MeanTarget):
    def __init__(self, selection_fraction = 0.35):
        super().__init__();
        self.p['selection_fraction'] = selection_fraction;

    def _calculate(self, population, evaluated_fitness):
        N_select = np.ceil(self.p['selection_fraction'] * population.shape[0]).astype(int);
        selected = population[np.argsort(evaluated_fitness)[-N_select:]];
        self._target = np.mean(selected, axis = 0);
        return self._target;

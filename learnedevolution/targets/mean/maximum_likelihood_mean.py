import numpy as np;

from .mean_target import MeanTarget;

class MaximumLikelihoodMean(MeanTarget):
    _API = 2.;
    def __init__(self, selection_fraction = 0.50):
        super().__init__();
        self.p['selection_fraction'] = selection_fraction;

    def _calculate(self, population):
        N_select = np.ceil(self.p['selection_fraction'] * population.raw_population.shape[0]).astype(int);
        selected = population.raw_population[np.argsort(population.fitness)[-N_select:]];
        self._target = population.mean + population.morph(np.mean(selected, axis = 0));
        return self._target;

    def _calculate_deterministic(self, population):
        return self._calculate(population);

    def _terminating(self, population):
        pass;

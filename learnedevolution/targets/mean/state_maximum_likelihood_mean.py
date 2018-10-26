import numpy as np;

from .mean_target import MeanTarget;

class StateMaximumLikelihoodMean(MeanTarget):
    _API = 2.;
    def __init__(self, state, selection_fraction = 0.50):
        super().__init__()
        self.state = state;
        self.p['selection_fraction'] = selection_fraction;

    def _reset(self, initial_mean, initial_covariance):
        self.state.reset();

    def _calculate(self, population):
        state = self.state.encode(population);
        state.shape = (-1,len(population), population.dimension+1);

        N_select = np.ceil(self.p['selection_fraction'] * len(population)).astype(int);
        selected = state[0,:,:2][np.argsort(state[0,:,2])[-N_select:]];
        self._target = self.state.decode(np.mean(selected, axis = 0)    );
        return self._target;

    def _calculate_deterministic(self, population):
        return self._calculate(population);

    def _terminating(self, population):
        pass;

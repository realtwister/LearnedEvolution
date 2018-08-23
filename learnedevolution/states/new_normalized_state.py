import numpy as np;
from .state import State;

class NewNormalizedState(State):
    def __init__(self, population_size, dimension):
        self.population_size = population_size;
        self.dimension = dimension;

    def _reset(self):
        self._prev_population = None;
        self._prev_fitness = None;

    def normalize_population(self, population, mean, covariance):
        translated = population - mean;
        self.population_scale_factor = np.sqrt(np.max(np.linalg.eig(covariance)[0]));
        normalized = translated/self.population_scale_factor;
        return normalized;

    def normalize_fitness(self, fitness, basis=None):
        if basis is None:
            basis = fitness;
        translated = fitness-np.mean(basis);
        normalized = translated/np.std(basis);
        return normalized;

    def _encode(self, population, fitness, mean, covariance):
        normalized_population = self.normalize_population(population, mean, covariance);
        normalized_fitness = self.normalize_fitness(fitness);
        state = np.append(normalized_population, normalized_fitness[:, None], axis=1);
        state = state[fitness.argsort(),:].flatten();
        if self._prev_population is not None:
            normalized_previous_population = self.normalize_population(self._prev_population, mean, covariance);
            normalized_previous_fitness = self.normalize_fitness(self._prev_fitness, fitness);
            prev_state = np.append(normalized_previous_population, normalized_previous_fitness[:, None], axis=1);
            prev_state = prev_state[self._prev_fitness.argsort(),:].flatten();
        else:
            prev_state = np.zeros((self.population_size*(self.dimension+1)))
        if np.any(np.isnan(state)):
            print("state is NaN");
        if np.any(np.isinf(state)):
            print("state is Inf");
        total_state = np.stack([state, prev_state]);
        self._prev_population = population;
        self._prev_fitness = fitness;
        return total_state.flatten();

    def _decode(self, action):
        return action*self.population_scale_factor;

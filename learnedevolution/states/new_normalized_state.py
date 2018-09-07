import numpy as np;
from .state import State;

class NewNormalizedState(State):
    def __init__(self, population_size, dimension):
        self.population_size = population_size;
        self.dimension = dimension;

    def _reset(self):
        self._prev_population = None;

    def normalize_fitness(self, population, reference = None):
        if reference is None:
            reference = population;
        translated = population.fitness-np.mean(reference.fitness);
        normalized = translated/np.std(reference.fitness - np.mean(reference.fitness));
        return normalized;

    def create_single_state(self, population= None, reference = None):
        if population is None:
            return np.zeros([self.population_size,self.dimension+1])
        if reference is None:
            normalized_population = population.raw_population;
            reference = population;
        else:
            normalized_population = reference.invert(population);
        normalized_fitness = self.normalize_fitness(population, reference);
        state = np.append(normalized_population, normalized_fitness[:, None], axis=1);
        state = state[population.fitness.argsort()];
        return state;

    def _encode(self, population):
        current_state = self.create_single_state(population)
        #prev_state = self.create_single_state(self._prev_population,population)
        total_state = np.stack([current_state]);
        if np.any(np.isnan(total_state)):
            print("state is NaN");
        if np.any(np.isinf(total_state)):
            print("state is Inf");
        self._prev_population = population;
        return total_state.flatten();

    def _decode(self, action):
        if self._prev_population is None:
            return action;
        dx = self._prev_population.morph(action);
        n = np.linalg.norm(dx);
        if n > 1e2:
            dx *= 1e2/n
        return dx;

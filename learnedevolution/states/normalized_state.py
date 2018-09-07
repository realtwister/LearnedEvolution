import numpy as np;
from collections import deque;

from .state import State;

class NormalizedState(State):
    def __init__(self, population_size, dimension, horizon = 2):
        self.population_size = population_size;
        self.dimension = dimension;
        self._horizon = horizon;

    def _reset(self):
        self._prev_states = deque([np.zeros([self.population_size,self.dimension+1])]*self._horizon ,maxlen = self._horizon);
        self._prev_population = None;

    def normalize_population(self, population):
        translated = population.population - population.mean;
        _,s,_ = population.svd;
        self.population_scale_factor = np.sqrt(np.max(s));
        normalized = translated/self.population_scale_factor;
        return normalized;

    def normalize_fitness(self, fitness):
        translated = fitness-np.mean(fitness);
        normalized = translated/np.std(translated);
        return normalized;

    def calculate_state(self, population):
        normalized_population = self.normalize_population(population);
        normalized_fitness = self.normalize_fitness(population.fitness);
        state = np.append(normalized_population, normalized_fitness[:, None], axis=1);
        self._prev_population = population;
        return state[population.fitness.argsort(),:]

    def _encode(self, population):
        current_state = self.calculate_state(population);
        if np.any(np.isnan(current_state)):
            print("current_state is NaN");
        if np.any(np.isinf(current_state)):
            print("current_state is Inf");
        self._prev_states.append(current_state);
        return np.array(self._prev_states).flatten();

    def _decode(self, action):
        return self._prev_population.mean + action*self.population_scale_factor;

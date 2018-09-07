import numpy as np;
from collections import deque;

from .state import State;

class BenchmarkState(State):
    def __init__(self, population_size, dimension, horizon = 2):
        self.population_size = population_size;
        self.dimension = dimension;

    def _encode(self, population):
        norm_pop = population.population - population.mean;
        norm_fitness = population.fitness - np.mean(population.fitness);
        state = np.append(norm_fitness[:,None]*norm_pop, norm_fitness);
        self.prev_population = population;
        return np.array(state).flatten();

    def _decode(self, action):
        return action+self.prev_population.mean;

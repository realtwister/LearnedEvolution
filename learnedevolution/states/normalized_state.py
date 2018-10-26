import numpy as np;
from collections import deque;
from gym.spaces import Box;

from .state import State;

class NormalizedState(State):
    def __init__(self, population_size, dimension, number_of_states = 2, epsilon=1e-20):
        self.epsilon = epsilon;
        self.population_size = population_size;
        self.dimension = dimension;
        self.number_of_states = number_of_states;

    def _reset(self):
        self._populations = deque(maxlen = self.number_of_states);
        self._prev_population = None;

    def normalize_population(self, population, reference = None):
        if reference is None:
            reference = population;
        translated = population.population - reference.mean;
        _,s,_ = reference.svd;
        self.population_scale_factor = max(self.epsilon,np.sqrt(np.max(s)));
        normalized = translated/self.population_scale_factor;
        return normalized;

    def normalize_fitness(self, fitness, reference = None):
        if reference is None:
            reference = fitness;
        translated = fitness-np.mean(reference);
        fitness_factor = max(np.std(reference),self.epsilon);
        normalized = translated/fitness_factor;
        return normalized;

    def create_single_state(self, population=None, reference=None):
        if population is None:
            return np.zeros([self.population_size,self.dimension+1])
        if reference is None:
            reference = population;
        normalized_population = self.normalize_population(population, reference);
        normalized_fitness = self.normalize_fitness(population.fitness, reference.fitness);
        state = np.append(normalized_population, normalized_fitness[:, None], axis=1);
        return state[population.fitness.argsort(),:]

    def _encode(self, population):
        self._populations.appendleft(population);
        total_state = [];
        for i in range(self.number_of_states):
            if i < len(self._populations):
                current_population = self._populations[i]
            else:
                current_population = None
            total_state.append(self.create_single_state(current_population, self._populations[0]));
        total_state = np.stack(total_state);
        if np.any(np.isnan(total_state)):
            print("state is NaN");
        if np.any(np.isinf(total_state)):
            print("state is Inf");
        return total_state.flatten();

    def _decode(self, action):
        return self._populations[0].mean + action*self.population_scale_factor;

    @property
    def state_space(self):
        single_state_size= self.population_size*(self.dimension+1);
        return dict(type='float', shape=(self.number_of_states*single_state_size,));

    @property
    def gym_state_space(self):
        single_state_size= self.population_size*(self.dimension+1);
        return Box(high = 10, low  = -10,shape=(self.number_of_states*single_state_size,));

    @property
    def action_space(self):
        return dict(type='float', shape=(self.dimension,));

    @property
    def gym_action_space(self):
        return Box(high = 100, low= -100, shape=(self.dimension,));

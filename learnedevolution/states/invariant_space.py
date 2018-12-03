import numpy as np;
from collections import deque;
from gym.spaces import Box;

from .state import State;

class InvariantSpace(State):
    def __init__(self,
        population_size,
        dimension,
        number_of_states = 2,
        epsilon = 1e-20):
        self.population_size = population_size;
        self.dimension = dimension;
        self.epsilon = epsilon;
        self.number_of_states = number_of_states;

    def _reset(self):
        self._populations = deque(maxlen = self.number_of_states);

    def append_population(self, population):
        self._populations.appendleft(population);

    def normalize_fitness(self, population, reference = None):
        if reference is None:
            reference = population;
        translated = population.fitness-np.mean(reference.fitness);
        factor =np.std(reference.fitness);
        if factor == 0.:
            if reference != population:
                return self.normalize_fitness(population);
            else:
                return translated;

        normalized = translated/factor;
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
        self.append_population(population);
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
        if len(self._populations) == 0:
            return action;
        dx = self._populations[0].morph(action);
        n = np.linalg.norm(dx);
        if n > 1e2:
            dx *= 1e2/n
        if np.any(np.isnan(dx)):
            print(dx, n);
            raise Exception("dx is nan");
        return self._populations[0].mean+dx;

    def invert(self, mean, reference = None):
        if reference is None:
            reference = self._populations[0];
        normalized_mean = reference.invert_individual(mean);

        return normalized_mean;



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

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'population_size',
            'dimension',
            'number_of_states',
            'epsilon'
        )
        cls._config_defaults(
            population_size = 100,
            dimension = 2,
            number_of_states = 2,
            epsilon = 1e-20
        )
        return super()._get_kwargs(config, key = key);

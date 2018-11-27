import numpy as np;
from collections import deque;
from gym.spaces import Box;
import matplotlib.pyplot as plt;

from .state import State;
import learnedevolution.utils.logging as logging;
logger = logging.child('ScaledState');

class ScaledState(State):
    def __init__(self, population_size, dimension,
        number_of_states = 2,
        epsilon = 1e-10):
        self.population_size = population_size;
        self.dimension = dimension;
        self.epsilon = epsilon;
        self.number_of_states = number_of_states;

    def _reset(self):
        self._populations = deque(maxlen = self.number_of_states);

        self._population_scales = [];
        self._fitness_translations =[];

    def append_population(self, population):
        self._populations.appendleft(population);

        # Population normalization variables
        self._population_scale = np.sqrt(np.max(population.svd[1]));
        self._population_scales.append(self._population_scale);
        self._population_translation = population.mean;

        # Fitness normalization variabels
        self._fitness_scale = np.std(population.fitness);
        self._fitness_translation = np.mean(population.fitness);
        self._fitness_translations.append(self._fitness_translation);

        # numerical safeguards
        if self._population_scale < self.epsilon:
            logger.debug('_population_scale (={}) smaller than epsilon (={})'.format(self._population_scale,self.epsilon));
            self._population_scale = self.epsilon;

        if self._fitness_scale < self.epsilon:
            print(self._fitness_scale);
            logger.debug('_fitness_scale (={}) smaller than epsilon (={})'.format(self._fitness_scale,self.epsilon));
            self._fitness_scale = self.epsilon;

    def normalize_population(self, population):
        translated = population.population-self._population_translation;
        normalized = translated/self._population_scale;
        return normalized;

    def normalize_fitness(self, population):
        translated = population.fitness-self._fitness_translation;
        normalized = translated/self._fitness_scale;
        return normalized;


    def create_single_state(self, population= None):
        if population is None:
            return np.zeros([self.population_size,self.dimension+1])
        normalized_population = self.normalize_population(population);
        normalized_fitness = self.normalize_fitness(population);
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
            total_state.append(self.create_single_state(current_population));
        total_state = np.stack(total_state);
        if np.any(np.isnan(total_state)):
            print("state is NaN");
        if np.any(np.isinf(total_state)):
            print("state is Inf");
        self.last_state = total_state;
        return total_state.flatten();

    def _decode(self, action):
        if np.any(np.isnan(action)):
            self.print_debug()
            raise Exception("action is nan");
        if len(self._populations) == 0:
            return action;
        dx = self._population_scale*action;
        n = np.linalg.norm(dx);
        if n > 1e10:
            self.print_debug();
            raise Exception('||dx||_2 (= {}) > 1e10'.format(n));

        if np.any(np.isnan(dx)):
            raise Exception("dx is nan");
        return self._population_translation+dx;

    def print_debug(self):
        print();
        print("------------ DEBUG: ScaledState ------------");
        properties = [
            "_population_scale",
            "_population_translation",
            "_fitness_translation",
            "_fitness_scale"
        ];
        for var in properties:
            if hasattr(self,var):
                print("\t{}: {}".format(var, getattr(self,var)));
        print("---------------- END DEBUG ----------------");
        print();
        plt.figure();
        plt.plot(self._population_scales);
        plt.figure();
        plt.semilogy(-np.array(self._fitness_translations));
        plt.show();


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

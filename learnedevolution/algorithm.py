
import numpy as np;
import tensorflow as tf;

from .utils.signals import method_event;
from .convergence.convergence_criterion import ConvergenceCriterion;

import logging;
log = logging.getLogger(__name__)

class Algorithm(object):
    __epsilon = 1e-5;
    @method_event('initialize')
    def __init__(self, dimension, mean_targets, covariance_targets, convergence_criteria,
        population_size = 100,
        maximum_iterations = 100):

        self._dimension = dimension;

        self._mean_targets = mean_targets;
        self._covariance_targets = covariance_targets;
        # recursive convergence check for logic groups
        self._convergence_criteria = convergence_criteria;

        self._population_size = population_size;
        self._maximum_iterations = maximum_iterations;

        self._steps = 0;

        self.seed(None);

    @method_event('seed')
    def seed(self, seed):
        """ Set the seed of the algorithm and its targets """
        self._random_state = np.random.RandomState(seed);
        self._call_on_targets('seed', (seed,))
        return seed;

    @method_event('reset')
    def reset(self):
        """ Reset """
        self._mean = self._initial_mean();
        self._covariance = self._initial_covariance();

        self._call_on_targets('reset', (self._mean, self._covariance));

        for criterion in self._convergence_criteria:
            criterion.reset(self._mean, self._covariance);


        self._converged = 0;
        self._steps += 1;
        self._iteration = 0;

    def _initial_mean(self):
        mean = np.zeros(self._dimension)
        return mean;

    def _initial_covariance(self):
        return np.eye(self._dimension);

    def set_target_attr(self, attr, value):
        for mean_target in self._mean_targets:
            setattr(mean_target, attr, value);

        for covariance_target in self._covariance_targets:
            setattr(covariance_target, attr, value);

    @method_event('maximize')
    def maximize(self, fitness, maximum_iterations = None, deterministic=False):
        if maximum_iterations is None:
            maximum_iterations = self._maximum_iterations;
        self.reset();
        while True:
            converged = self._step(fitness, deterministic);
            if converged or self._iteration >= maximum_iterations:
                break;
        return self._mean, self._covariance;

    @method_event('step')
    def _step(self, fitness, deterministic=False):
        # Sample new population
        population = self._sample();
        # Evaluate fitness of population
        self._evaluated_fitness = evaluated_fitness = fitness(population);

        # Calculate new mean
        mean = self._calculate_mean(population, evaluated_fitness, deterministic);

        #Calculate new covariance
        covariance = self._calculate_covariance(population, evaluated_fitness, deterministic);

        self._iteration += 1;
        if self._is_converged(evaluated_fitness):
            self._terminate(fitness,deterministic);
            return True;
        return False;




    @method_event('sample')
    def _sample(self):
        self._population = self._random_state.multivariate_normal(self._mean,
            self._covariance, self._population_size);
        return self._population;

    @method_event('calculate_mean')
    def _calculate_mean(self, population, evaluated_fitness,deterministic):
        mean = np.zeros(self._dimension);
        for mean_target, w in self._mean_targets.items():
            mean += w * mean_target(population, evaluated_fitness, deterministic);
        self._call_on_targets('update_mean', [mean])
        self._mean = mean;
        return mean;

    @method_event('calculate_covariance')
    def _calculate_covariance(self, population, evaluated_fitness,deterministic):
        covariance = np.zeros(shape=(self._dimension, self._dimension));
        for target, w in self._covariance_targets.items():
            covariance += w * target(population, evaluated_fitness,deterministic);
        self._call_on_targets('update_covariance', [covariance])
        self._covariance = covariance
        return covariance;

    @method_event('is_converged')
    def _is_converged(self, fitness):
        converged = False;
        for criterion in self._convergence_criteria:
            converged |= criterion(fitness, self._mean, self._covariance);

        return converged;

        if np.trace(self._covariance)/self._dimension < self.__epsilon:
            self._converged += 1;
            if self._converged >= 5:
                return True;
        else:
            self._converged = 0;
        return False;

    @method_event('terminate')
    def _terminate(self, fitness, deterministic):
        population = self._sample();
        self._evaluated_fitness = evaluated_fitness = fitness(population);
        self._mean_fitness = np.mean(self._evaluated_fitness);
        self._call_on_targets('terminating', [population, evaluated_fitness,deterministic]);

    @property
    def current_step(self):
        return self._steps;

    def _call_on_targets(self, method, args =[]):
        for mean_target in self._mean_targets:
            getattr(mean_target, method)(*args)

        for covariance_target in self._covariance_targets:
            getattr(covariance_target, method)(*args)

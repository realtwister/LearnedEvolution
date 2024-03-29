
import numpy as np;
import tensorflow as tf;

from .utils.signals import method_event;
from .utils.parse_config import ParseConfig, config_factory;

from .population import Population;

import logging;
log = logging.getLogger(__name__)

class Algorithm(ParseConfig):
    __epsilon = 1e-5;

    @method_event('initialize')
    def __init__(self, dimension, mean_function, covariance_function, convergence_criterion,
        population_size = 100,
        maximum_iterations = 100):

        self._dimension = dimension;

        self._mean_targets = {mean_function:1};
        self._covariance_targets = {covariance_function:1};
        self._convergence_criteria = [convergence_criterion];

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
            if converged:
                break;
        return self._mean, self._covariance;

    @method_event('step')
    def _step(self, fitness, deterministic=False):
        # Sample new population
        population = self._generate_population(fitness);

        # Calculate new mean
        mean = self._calculate_mean(population, deterministic);

        #Calculate new covariance
        covariance = self._calculate_covariance(population, deterministic);

        self._iteration += 1;
        if self._is_converged(population):
            self._terminate(fitness,deterministic);
            return True;
        return False;




    @method_event('generate_population')
    def _generate_population(self, fitness):
        population = Population(
            mean = self._mean,
            covariance = self._covariance,
            population_size = self._population_size,
            fitness_fn = fitness,
            random_state = self._random_state);
        self._population_obj = population;
        self._population = population.population;
        self._evaluated_fitness = population.fitness;
        return population;

    @method_event('calculate_mean')
    def _calculate_mean(self, population, deterministic):
        mean = np.zeros(self._dimension);
        for mean_target, w in self._mean_targets.items():
            mean += w * mean_target(population, deterministic);
        self._call_on_targets('update_mean', [mean])
        self._old_mean = self._mean;
        self._mean = mean;
        return mean;

    @method_event('calculate_covariance')
    def _calculate_covariance(self, population, deterministic):
        covariance = np.zeros(shape=(self._dimension, self._dimension));
        for target, w in self._covariance_targets.items():
            covariance += w * target(population,deterministic);
        self._call_on_targets('update_covariance', [covariance])
        self._old_covariance = self._covariance
        self._covariance = covariance
        return covariance;

    @method_event('is_converged')
    def _is_converged(self, population):
        converged = False;
        for criterion in self._convergence_criteria:
            converged |= criterion(population);

        return converged;

    @method_event('terminate')
    def _terminate(self, fitness, deterministic):
        population = self._generate_population(fitness);
        self._mean_fitness = np.mean(population.fitness);
        self._call_on_targets('terminating', [population,deterministic]);

    def close(self):
        self._call_on_targets('close');

    def save(self, savedir):
        self.call_on_targets('save', [savedir]);

    def restore(self, restoredir):
        self.call_on_targets('restore', [restoredir]);

    @property
    def current_step(self):
        return self._steps;

    def _call_on_targets(self, method, args =[]):
        for mean_target in self._mean_targets:
            getattr(mean_target, method)(*args)

        for covariance_target in self._covariance_targets:
            getattr(covariance_target, method)(*args)

    def call_on_targets(self, method, args =[]):
        for target in self._mean_targets:
            if hasattr(target,method):
                fn = getattr(target, method);
                if callable(fn):
                    fn(*args);

        for target in self._covariance_targets:
            if hasattr(target,method):
                fn = getattr(target, method);
                if callable(fn):
                    fn(*args);

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        from .convergence import convergence_classes;
        from .targets.mean import mean_classes;
        from .targets.covariance import covariance_classes;
        cls._config_required(
            'dimension',
            'population_size',
            'mean_function',
            'covariance_function',
            'convergence_criterion',
        )
        cls._config_defaults(
            dimension = 2,
            population_size = 100,
        )

        kwargs = super()._get_kwargs(config, key = key);

        # Initiate convergence criterion
        kwargs['convergence_criterion'] = config_factory(
            classes = convergence_classes,
            config = config,
            key = kwargs['convergence_criterion']['key']
        );

        # Initiate mean function
        kwargs['mean_function'] = config_factory(
            classes = mean_classes,
            config = config,
            key = kwargs['mean_function']['key']
        );

        # Initiate mean function
        kwargs['covariance_function'] = config_factory(
            classes = covariance_classes,
            config = config,
            key = kwargs['covariance_function']['key']
        );

        return kwargs

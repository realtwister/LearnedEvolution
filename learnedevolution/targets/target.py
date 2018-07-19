import numpy as np;

from ..utils.signals import method_event;

class Target(object):
    def __init__(self):
        self.seed(None);
        self._params = dict();

    def _reset(self, initial_mean, initial_covariance):
        pass;

    @method_event('reset')
    def reset(self, initial_mean, initial_covariance,*args):
        self._mean = np.array(initial_mean);
        self._covariance = np.array(initial_covariance);
        self._reset(initial_mean, initial_covariance);

    def _seed(self, seed):
        pass;

    @method_event('seed')
    def seed(self, seed):
        self._random_state = np.random.RandomState(seed);
        self._seed(seed);

    def _calculate(self, population, evaluated_fitness):
        raise NotImplementedError("Method _calculate is not implemented for {}".format(self.__name__));

    def _calculate_deterministic(self, population,evaluated_fitness):
        return self._calculate(population, evaluated_fitness);

    @method_event('call')
    def __call__(self, population, evaluated_fitness = None, deterministic=False):
        calculate = self._calculate;
        if deterministic:
            calculate = self._calculate_deterministic;
        if evaluated_fitness is None:
            return calculate(population[:,:-1], population[:,-1]);
        else:
            return calculate(population, evaluated_fitness);

    def _update_covariance(self, covariance):
        pass;

    @method_event('update_covariance')
    def update_covariance(self, covariance):
        self._update_covariance(covariance);

    def _update_mean(self, mean):
        pass;

    @method_event('update_mean')
    def update_mean(self, mean):
        self._update_mean(mean);

    def _terminating(self, population, evaluated_fitness):
        pass;

    def _terminating_deterministic(self, population, evaluated_fitness):
        self._terminating(population, evaluated_fitness);

    @method_event('terminating')
    def terminating(self, population, evaluated_fitness = None, deterministic=False):
        if deterministic:
            self._terminating_deterministic(population, evaluated_fitness);
            return;
        if evaluated_fitness is None:
            self._terminating(population[:,:-1], population[:,-1]);
        else:
            self._terminating(population, evaluated_fitness);

    @property
    def proto(self):
        #TODO: should implement this;
        raise NotImplementedError();


    @property
    def p(self):
        return self._params;

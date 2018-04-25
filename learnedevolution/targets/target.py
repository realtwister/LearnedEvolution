import numpy as np;

from ..utils.signals import method_event;

class Target(object):
    def __init__(self):
        self.seed(None);
        self._params = dict();

    def _reset(self, initial_mean, initial_covariance):
        pass;

    @method_event('reset')
    def reset(self, initial_mean, initial_covariance):
        self._mean = np.array(initial_mean);
        self._covariance = np.array(initial_covariance);
        self._reset(initial_mean, initial_covariance);

    @method_event('seed')
    def seed(self, seed):
        self._random_state = np.random.RandomState(seed);

    def _calculate(self, population, evaluated_fitness):
        raise NotImplementedError("Method _calculate is not implemented for {}".format(self.__name__));

    @method_event('call')
    def __call__(self, population, evaluated_fitness):
        return self._calculate(population, evaluated_fitness);

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

    def _terminating(self, population):
        pass;

    @method_event('terminating')
    def terminating(self, population, evaluated_fitness):
        self._terminating(population);

    @property
    def proto(self):
        #TODO: should implement this;
        raise NotImplementedError();


    @property
    def p(self):
        return self._params;

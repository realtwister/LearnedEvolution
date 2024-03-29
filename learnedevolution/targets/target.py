import numpy as np;

from ..utils.signals import method_event
from ..utils.parse_config import ParseConfig

class Target(ParseConfig):
    _API = 1;
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

    def _calculate(self, population):
        raise NotImplementedError("Method _calculate is not implemented for {}".format(self.__name__));

    def _calculate_deterministic(self, population):
        return self._calculate(population);

    @method_event('call')
    def __call__(self, population, deterministic=False):
        calculate = self._calculate;
        if deterministic:
            calculate = self._calculate_deterministic;
        return calculate(population);

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

    def _terminating_deterministic(self, population):
        return self._terminating(population);

    @method_event('terminating')
    def terminating(self, population, deterministic=False):
        if deterministic:
            calculate = self._terminating_deterministic
        else:
            calculate = self._terminating;
        return calculate(population);

    def close(self):
        pass;

    @property
    def proto(self):
        #TODO: should implement this;
        raise NotImplementedError();


    @property
    def p(self):
        return self._params;

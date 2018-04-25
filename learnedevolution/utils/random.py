

import numpy as np;

class RandomGeneratable(object):
    @classmethod
    def generator(cls, **kwargs):
        return RandomGenerator(cls, **kwargs);

    @staticmethod
    def random(random_state, **kwargs):
        raise NotImplementedError("random trait not implemented for {}.".format(__class__.__name__));

class RandomGenerator(object):
    def __init__(self, cls, **kwargs):
        assert issubclass(cls, RandomGeneratable);
        self._class = cls;
        self._kwargs = kwargs;
        self.seed();

    def seed(self, seed = None):
        self._random_state = np.random.RandomState(seed);

    def generate(self):
        return self._class.random(self._random_state, **self._kwargs);

    def iter(self, n = 0):
        i = 0;
        while n == 0 or i<n:
            yield self.generate();
            i +=1;

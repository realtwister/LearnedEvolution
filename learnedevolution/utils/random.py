
import collections;
import numpy as np;

class RandomGeneratable(object):
    @classmethod
    def generator(cls, **kwargs):
        return RandomGenerator(cls, **kwargs);

    @staticmethod
    def random(random_state, **kwargs):
        raise NotImplementedError("random trait not implemented for {}.".format(__class__.__name__));

class RandomGenerator(object):
    def __init__(self, clss, **kwargs):
        if not isinstance(clss,collections.Iterable ):
            clss = [clss];
        for cls in clss:
            assert hasattr(cls, 'random');
        self._classes = clss;
        self._kwargs = kwargs;
        self.seed();

    def copy(self):
        return self.__class__(self._classes,**self._kwargs);

    def seed(self, seed = None):
        self._random_state = np.random.RandomState(seed);

    def generate(self):
        cls = self._random_state.choice(self._classes);
        self._current = cls.random(self._random_state, **self._kwargs);
        return self._current;

    def append(self, cls):
        assert hasattr(cls, 'random');
        self._classes.append(cls);

    def iter(self, n = 0):
        i = 0;
        while n == 0 or i<n:
            yield self.generate();
            i +=1;

    def get_state(self):
        return self._random_state.get_state();

    def set_state(self, tuple):
        self._random_state.set_state(tuple);

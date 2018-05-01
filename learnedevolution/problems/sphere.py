
import numpy as np;

from .problem import Problem;
from ..utils.random import RandomGeneratable;

class Sphere(Problem, RandomGeneratable):
    """ Implementation of the sphere problem (f1 of BBOB)
    params:
        a: The optimum value (np.array(dim))
        b: The fitness translation (float)
    """
    _type = "Sphere";

    def __init__(self, a, b):
        super().__init__(len(a));
        self._evaluations = 0;
        self._params = dict(
            a = np.array(a),
            b = np.array([b])
        )

    @staticmethod
    def random(random_state, dimension, position_spread = 100, offset_spread = 10 ):
        a = random_state.rand(dimension) * position_spread;
        b = random_state.rand() * offset_spread;
        return Sphere(a,b);

    def fitness(self, xs):
        super().fitness(xs);
        xs = xs-self._params['a'];
        s = np.add.reduce(xs*xs, axis = 1, keepdims = False) + self._params['b'];
        return s;

    @property
    def optimum(self):
        return self._params['a'];

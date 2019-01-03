
import numpy as np;

from .problem import Problem;
from ..utils.random import RandomGeneratable;

class Ellipsoid(Problem, RandomGeneratable):
    """ Implementation of the ellipsoid problem (f2 of BBOB)
    params:
        a: stretch factor
    """
    _type = "Sphere";

    def __init__(self, a):
        super().__init__(len(a));
        self._evaluations = 0;
        self._params = dict(
            a = np.array(a)
        )

    @staticmethod
    def random(random_state, dimension, spread = 100. ):
        a = random_state.rand(dimension) * spread;
        return Ellipsoid(a);

    def fitness(self, xs):
        super().fitness(xs);
        xs = self._params['a']*xs;
        s = -np.linalg.norm(xs, axis = -1);
        return s;

    @property
    def optimum(self):
        return np.array([0]*dimension);


import numpy as np;

from .problem import Problem;
from ..utils.random import RandomGeneratable;

class Rosenbrock(Problem, RandomGeneratable):
    """ Implementation of the rosenbrock problem (f1 of BBOB)
    params:
        a: The optimum value (np.array(dim))
        b: The fitness translation (float)
    """
    _type = "Rosenbrock";

    def __init__(self,dimension):
        super().__init__(dimension);
        self._evaluations = 0;
        self._params = dict(
        )

    @staticmethod
    def random(random_state, dimension):
        return Rosenbrock(dimension);

    def fitness(self, xs):
        super().fitness(xs);
        xs = xs/100+np.array([1]*self.dimension);
        x = xs[...,:-1]
        y = xs[...,1:]
        a = 1. - x
        b = y - x*x

        s = np.sum(a**2 + b**2*100, axis=-1);
        return -s;

    @property
    def optimum(self):
        return np.array([0]*self.dimension);

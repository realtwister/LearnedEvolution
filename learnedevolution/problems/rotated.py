import numpy as np;
from scipy.stats import special_ortho_group

from .problem import Problem;
from ..utils.random import RandomGeneratable, RandomGenerator;

class RotateProblem:
    def __init__(self, problem_cls):
        self._problem = problem_cls;

    def random(self, random_state, dimension,**kwargs):
        return RotatedProblem.random(random_state, dimension, self._problem, **kwargs)

class RotatedProblem(Problem, RandomGeneratable):

    def __init__(self, problem, rotation):
        assert isinstance(problem, Problem);

        super().__init__(problem.dimension);
        self._problem = problem;
        self._type = "Rotated_"+problem.type;
        self._params = dict(
            rotation = rotation,
            problem = problem
        );

    @staticmethod
    def random(random_state, dimension, problem_cls, **kwargs):
        problem = problem_cls.random(random_state, dimension, **kwargs);

        rotation = special_ortho_group.rvs(dimension, random_state = random_state);

        return RotatedProblem(problem, rotation);

    def fitness(self, xs):
        super().fitness(xs);
        xs_rotated = xs@self._params['rotation'];
        return self._params['problem'].fitness(xs_rotated);

    @property
    def optimum(self):
        problem_opt = self._params['problem'].optimum;
        return problem_opt@self._params['rotation'].T;

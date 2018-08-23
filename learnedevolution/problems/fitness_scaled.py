import numpy as np;

from .problem import Problem;
from ..utils.random import RandomGeneratable, RandomGenerator;

class FitnessScaleProblem:
    def __init__(self, problem_cls):
        self._problem = problem_cls;

    def random(self, random_state, dimension,**kwargs):
        return FitnessScaledProblem.random(random_state, dimension, self._problem, **kwargs)

class FitnessScaledProblem(Problem, RandomGeneratable):

    def __init__(self, problem, scale):
        assert isinstance(problem, Problem);

        super().__init__(problem.dimension);
        self._problem = problem;
        self._type = "Scaled_"+problem.type;
        self._params = dict(
            scale = scale,
            problem = problem
        );

    @staticmethod
    def random(random_state, dimension, problem_cls, **kwargs):
        problem = problem_cls.random(random_state, dimension, **kwargs);

        scale = 100*random_state.rand();

        return FitnessScaledProblem(problem, scale);

    def fitness(self, xs):
        super().fitness(xs);
        return self._params['scale'] * self._params['problem'].fitness(xs);

    @property
    def optimum(self):
        return self._params['problem'].optimum;

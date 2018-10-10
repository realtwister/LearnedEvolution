import numpy as np;
from scipy.stats import special_ortho_group

from .problem import Problem;
from ..utils.random import RandomGeneratable, RandomGenerator;

class DiscretizeProblem:
    def __init__(self,* ,
        initial_point = 0.,
        width = 0.2):
        self._initial_point = initial_point;
        self._width = width;

    def __call__(self, problem_cls):
        self._problem = problem_cls;
        return self;

    def random(self, random_state, dimension,**kwargs):
        return DiscretizedProblem.random(random_state, dimension, self._problem, self._initial_point, self._width, **kwargs)

class DiscretizedProblem(Problem, RandomGeneratable):

    def __init__(self, parent, initial_point, width):
        assert isinstance(parent, Problem);

        super().__init__(parent.dimension);
        self._parent = parent;
        self._type = "Discretized_"+parent.type;
        self._params = dict(
            initial_point = initial_point,
            width = width
        );

    @staticmethod
    def random(random_state, dimension, problem_cls, initial_point = 0., width=0.2,  **kwargs):
        problem = problem_cls.random(random_state, dimension, **kwargs);

        if isinstance(initial_point, tuple):
            delta = (initial_point[1]- initial_point[0]);

            initial_point = delta*random_state.rand(dimension)+initial_point[0];

        if isinstance(width, tuple):
            delta = (width[1]- width[0]);

            width = delta*random_state.rand()+width[0];

        return DiscretizedProblem(problem, initial_point, width);

    def fitness(self, xs):
        super().fitness(xs);
        xs= xs-self._params['initial_point'];
        xs = np.round(xs/self._params['width'])*self._params['width'];
        xs = xs + self._params['initial_point'];
        return self._parent.fitness(xs);

    @property
    def optimum(self):
        return self._parent.optimum;
